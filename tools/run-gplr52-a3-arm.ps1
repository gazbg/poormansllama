# T-GPLR-52 arm A3: the proven-active pinned probe. The 200k charter shape
# twice on the SAME binary - copy-source mmap-registered vs copy-source pinned
# - with the engine now emitting the A3/AC4 in-trace assertion:
#   run_start.staging_source        (engine state flags, not argv echo)
#   run_start.staging_source_match  (1 = the label matches engine state)
# --check-telemetry FAILS any trace whose pair disagrees, and the arm verdict
# carries both fields, so "pinned" can no longer be claimed without proof.
#
# Pre-registered H3 evaluation (written BEFORE the arms run; T-52 block):
#   H3 predicted: with in-trace proof of an active pinned pool, 200k per-copy
#   cost collapses toward the 4k regime (RAM-speed source, no file reads).
#   - pinned p50 <= 0.50 x mmap p50  -> H3 CONFIRMED: T-51 A2's pinned arm
#     effectively ran mmap (fallback) -> the T-51 A2 addendum must be corrected.
#   - pinned p50 ~ mmap p50 (within 20%) -> H3's collapse prediction FALSIFIED:
#     pinned WAS engaged in T-51 A2 (no fallback existed); the per-copy cost is
#     source-independent as concluded, now with engagement PROOF. F5 resolved.
#   - in between -> ANOMALOUS: record and escalate to the owner.
# Cross-binary note: the comparison to A1/A2 medians is CORROBORATING only
# (this is a new binary); the A/B verdict is within-A3, same-binary.
param(
    [Parameter(Mandatory = $true)][ValidateSet('mmap', 'pinned')][string] $Source,
    [string] $Tag = ''
)
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base  = Join-Path $root 'build\llama-cuda'
$exe   = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts\anchor-200000.txt'
$verdictDir = Join-Path $base 'verdicts'
if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }
. (Join-Path $root 'tools\arm-verdict.ps1')

if (-not $Tag) { $Tag = "gplr52-a3-$Source" }
$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'bin\Release\llama.dll')).Hash.Substring(0, 16)
Write-Host ("[{0}] exe {1} dll {2}" -f $Tag, $exeSha, $dllSha)

$trace = Join-Path $base "$Tag.jsonl"
$log   = Join-Path $base "$Tag.log"
Remove-Item $trace, $log -ErrorAction SilentlyContinue

$a = @('--model', $model, '--prompt-file', $prompt,
       '--pp-tokens', '200000', '--ctx-size', '200704',
       '--ubatch', '512', '--pp-no-logits',
       '--tg-tokens', '3',
       '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
       '--act-slab-dtype', 'bf16',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '2',
       '--copy-source', $(if ($Source -eq 'pinned') { 'pinned' } else { 'mmap-registered' }),
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0} (source {1})" -f $Tag, $Source)

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log; arm = "A3-$Source"; source = $Source }
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
        $fields['resolved_copy_source'] = $rs.copy_source
        $fields['resolved_copy_source_src'] = $rs.copy_source_source
        # THE A3/AC4 fields: engine-derived staging source + match. A missing
        # key on this binary is itself a failure (the engine must emit it).
        $fields['engine_staging_source']      = $rs.staging_source
        $fields['engine_staging_source_match'] = $rs.staging_source_match
        if (-not $rs.staging_source) { $fail += 'run_start missing staging_source (engine did not emit the A3 assertion)' }
        elseif ($rs.staging_source_match -ne 1) { $fail += "staging_source $($rs.staging_source) does not match $($rs.copy_source)" }
        else {
            $want = if ($Source -eq 'pinned') { 'pinned' } else { 'mmap-registered' }
            if ($rs.staging_source -ne $want) { $fail += "engine staging_source $($rs.staging_source), want $want" }
        }

        # check-telemetry now enforces the match==1 invariant; a failure here
        # is the loud-refusal path working.
        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

        # copy stats: PP phase-0 only (T-GPLR-52 F1/AC1)
        $t = @()
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"copy_complete"*' -and $ln -like '*"phase":0,*') {
                $t += [double](($ln | ConvertFrom-Json).transfer_us) / 1e3
            }
        }
        $fields['copy_count_phase0'] = $t.Count
        if ($t.Count -ne 80) { $fail += "phase-0 copy count $($t.Count), want 80 (2 passes x 40)" }
        else {
            $sorted = $t | Sort-Object
            $fields['copy_p50_ms'] = [math]::Round($sorted[40], 1)
            $fields['copy_p90_ms'] = [math]::Round($sorted[[int](80 * 0.90)], 1)
            $fields['copy_max_s']  = [math]::Round(($t | Measure-Object -Maximum).Maximum / 1e3, 2)
        }

        $phases = @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                    ForEach-Object { $_ | ConvertFrom-Json })
        $i = 0
        foreach ($p in $phases) {
            $i++
            $fields[("pass{0}_wall_s"  -f $i)] = [math]::Round($p.wall_us / 1e6, 2)
            $fields[("pass{0}_compute_s" -f $i)] = [math]::Round($p.total_compute_us / 1e6, 2)
            $fields[("pass{0}_tok_s"   -f $i)] = [math]::Round($p.tokens_per_s, 1)
        }

        $readyWait = 0.0
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"layer_ready_wait"*') {
                $e = $ln | ConvertFrom-Json
                $readyWait += [double]($(if ($e.wait_us) { $e.wait_us } else { 0 }))
            }
        }
        $fields['ready_wait_total_s'] = [math]::Round($readyWait / 1e6, 2)
        Write-Host ("  staging_source={0} match={1} | copy p50 {2} ms max {3} s | ready_wait {4} s" -f `
            $fields['engine_staging_source'], $fields['engine_staging_source_match'], `
            $fields['copy_p50_ms'], $fields['copy_max_s'], $fields['ready_wait_total_s'])
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
