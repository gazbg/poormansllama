# T-GPLR-52 arm A2: the warm/eviction probe. --pp-repeat 3 at 200k and 50k,
# copy-source mmap-registered EXPLICIT at both shapes (H2 is about the
# registered mapping's file pages; the recorded 50k baseline ran pinned, which
# per T-52 F5 may never have engaged - A2 must exercise the file-read path at
# both shapes to test it).
#
# Pre-registered H2 predictions (written BEFORE the arms run; see the T-52
# block): pass-1 faults bank pages at disk-class rate (~73 MB/s effective);
# the Windows file cache evicts far-ahead pages before reuse when the pass is
# long, but retains them between passes when the pass is short.
#   200k: pass-3 ~ pass-2  (|p3-p2|/p2 <= 10%)      -> H2 CONFIRMED (evicted regime)
#         pass-3 <= 0.80 * pass-2                   -> H2 FALSIFIED (banks survive)
#   50k : pass-3 <= 0.85 * pass-2                   -> H2 CONFIRMED (survivor regime)
#         pass-3 ~ pass-2  (|p3-p2|/p2 <= 10%)      -> H2 survivor prediction
#                                                      FALSIFIED (no surviving regime
#                                                      at any tested shape; the 4k
#                                                      pass-2 halving is first-touch
#                                                      warm-up, not survival)
# Shapes cloned verbatim from run-gplr50-arm.ps1 (4k/50k/200k blocks) except:
# --pp-repeat 3 (the probe) and --copy-source mmap-registered at 50k.
param(
    [Parameter(Mandatory = $true)][ValidateSet('200k', '50k')][string] $Shape,
    [string] $Tag = ''
)
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base  = Join-Path $root 'build\llama-cuda'
$exe   = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$verdictDir = Join-Path $base 'verdicts'
if (-not (Test-Path $exe)) { Write-Host "MISSING EXE: $exe"; exit 1 }
. (Join-Path $root 'tools\arm-verdict.ps1')

if (-not $Tag) { $Tag = "gplr52-a2-$Shape" }
$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'bin\Release\llama.dll')).Hash.Substring(0, 16)
# same-binary identity vs the A1-baseline verdict (gplr50-200k-on)
$baselinePath = Join-Path $verdictDir 'gplr50-200k-on.verdict.json'
$binaryMatchesBaseline = $false
if (Test-Path $baselinePath) {
    $baseline = (Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json).fields
    $binaryMatchesBaseline = ($exeSha -eq $baseline.exe_sha256_16 -and $dllSha -eq $baseline.llama_dll_sha256_16)
}
Write-Host ("[{0}] exe {1} dll {2} matches-A1-binary: {3}" -f $Tag, $exeSha, $dllSha, $binaryMatchesBaseline)

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
Remove-Item $trace, $log -ErrorAction SilentlyContinue

$anchor = Join-Path $base ('gplr07-qa-prompts\anchor-{0}.txt' -f ($(if ($Shape -eq '200k') { '200000' } else { '50000' })))
if (-not (Test-Path -LiteralPath $anchor)) { Write-Host "MISSING ANCHOR: $anchor"; exit 1 }

$a = @('--model', $model, '--prompt-file', $anchor,
       '--pp-tokens', '200000', '--ctx-size', '200704',
       '--ubatch', '512', '--pp-no-logits',
       '--tg-tokens', '3',
       '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
       '--act-slab-dtype', 'bf16',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '3',
       '--copy-source', 'mmap-registered',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)
if ($Shape -eq '50k') {
    # verbatim from the T-50 50k shape: no ctx-size override
    $a = @('--model', $model, '--prompt-file', $anchor,
           '--pp-tokens', '50000',
           '--ubatch', '512', '--pp-no-logits',
           '--tg-tokens', '3',
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '3', '--allow-full-residency',
           '--residency-budget-mib', '0',
           '--pp-repeat', '3',
           '--copy-source', 'mmap-registered',
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', $trace)
}

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0} (shape {1})" -f $Tag, $Shape)

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log; arm = "A2-repeat3-$Shape"; shape = $Shape
             binary_matches_a1 = [bool]$binaryMatchesBaseline }
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode 124 -StartedUtc $started `
            -LogPath $log -ErrorText 'mutex timeout' -Fields $fields | Out-Null
        exit 124
    }
    Write-Host '  mutex held'

    $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
    $fields['gpu_pre_run'] = $gpuPre
    Write-Host ("  gpu_pre_run = {0}" -f $gpuPre)
    if ($gpuPre -ne 'clean') {
        if ($gpuPre -eq 'probe-unavailable') {
            Write-Warning 'GPU busy probe unavailable; proceeding UNVERIFIED.'
        } else {
            Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode 125 -StartedUtc $started `
                -LogPath $log -ErrorText "GPU busy: $gpuPre" -Fields $fields | Out-Null
            Write-Host ("REFUSED: GPU busy ({0})" -f $gpuPre)
            exit 125
        }
    }

    $t0 = Get-Date
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['arm_exit'] = $armExit
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])

    $fail = @()
    if ($armExit -ne 0) { $fail += "arm exited $armExit" }
    else {
        $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
        $fields['gpu_post_run'] = $gpuPost
        if ($gpuPost -ne 'clean') { Write-Warning ("GPU post-run probe: {0} - timing suspect." -f $gpuPost) }

        $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
        $fields['resolved_copy_source']       = $rs.copy_source
        $fields['resolved_copy_source_src']   = $rs.copy_source_source
        $fields['resolved_prebuild_per_pass'] = $rs.pp_prebuild_per_pass
        $fields['resolved_prebuild_src']      = $rs.pp_prebuild_per_pass_source
        if ($rs.copy_source -ne 'mmap-registered') { $fail += "copy_source $($rs.copy_source), want mmap-registered" }

        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

        # ---- the A2 table: pass-index vs median-copy, phase-0 (PP) copies only.
        # With repeat=3 the trace carries 3 x 40 phase-0 copies in time order;
        # split by index (the S2 census proved the copy stream is exactly
        # n_slots-fair and ordered; the count assertion catches any drift).
        $copies = @()
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"copy_complete"*' -and $ln -like '*"phase":0*') {
                $copies += ($ln | ConvertFrom-Json)
            }
        }
        $copies = @($copies | Sort-Object t_us)
        $fields['copy_count_phase0'] = $copies.Count
        if ($copies.Count -ne 120) {
            $fail += "phase-0 copy count $($copies.Count), want 120 (3 passes x 40 layers)"
        } else {
            $per = @()
            for ($p = 0; $p -lt 3; $p++) {
                $v = @($copies[($p * 40)..($p * 40 + 39)] | ForEach-Object { [double]$_.transfer_us / 1e3 } | Sort-Object)
                $p50 = [math]::Round($v[20], 1)
                $fields[("pass{0}_p50_ms" -f ($p + 1))] = $p50
                $fields[("pass{0}_min_ms" -f ($p + 1))] = [math]::Round($v[0], 1)
                $fields[("pass{0}_max_s" -f ($p + 1))]  = [math]::Round($v[39] / 1e3, 2)
                $per += $p50
            }
            $fields['drop_1to2_pct'] = [math]::Round(100.0 * ($per[0] - $per[1]) / $per[0], 1)
            $fields['drop_2to3_pct'] = [math]::Round(100.0 * ($per[1] - $per[2]) / $per[1], 1)

            # pre-registered verdict bands
            $rel23 = [math]::Abs($per[2] - $per[1]) / $per[1]
            if ($Shape -eq '200k') {
                if ($per[2] -le 0.80 * $per[1]) { $fields['h2_verdict'] = 'FALSIFIED: banks survive at 200k' }
                elseif ($rel23 -le 0.10)        { $fields['h2_verdict'] = 'CONFIRMED: evicted regime, pass-3 ~ pass-2' }
                else                            { $fields['h2_verdict'] = 'ANOMALOUS: pass-3 moved 10-20% without a prediction' }
            } else {
                if ($per[2] -le 0.85 * $per[1]) { $fields['h2_verdict'] = 'CONFIRMED: survivor regime, pass-3 drops' }
                elseif ($rel23 -le 0.10)        { $fields['h2_verdict'] = 'survivor prediction FALSIFIED: no surviving regime at 50k' }
                else                            { $fields['h2_verdict'] = 'ANOMALOUS: pass-3 rose >10%' }
            }

            # per-pass wall/compute from phase_summary (the 'pp' events, 3 of them)
            $phases = @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                        ForEach-Object { $_ | ConvertFrom-Json })
            $i = 0
            foreach ($p in $phases) {
                $i++
                $fields[("pass{0}_wall_s"  -f $i)] = [math]::Round($p.wall_us / 1e6, 2)
                $fields[("pass{0}_compute_s" -f $i)] = [math]::Round($p.total_compute_us / 1e6, 2)
                $fields[("pass{0}_tok_s"   -f $i)] = [math]::Round($p.tokens_per_s, 1)
                $fields[("pass{0}_evictions" -f $i)] = [int]$p.evictions
            }
            Write-Host ("  pass medians: {0} -> {1} -> {2} ms | H2: {3}" -f $per[0], $per[1], $per[2], $fields['h2_verdict'])
        }
    }
    if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
Write-ArmVerdict -Dir $verdictDir -Tag $Tag -ExitCode $exit -StartedUtc $started `
    -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) -> {2}" -f $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, (Join-Path $verdictDir ($Tag + '.done')))
exit $exit
