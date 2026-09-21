# T-GPLR-51 arm S2: the default-combo confirmation leg the draft never measured.
# The four-arm RESULT compared 3 vs 4 slots (A3) but never 2 vs 3 on this binary;
# the owner's "keep 3-slot + async-feed + mmap as default if really best" needs
# the 2-slot datum. Shape VERBATIM vs A1 (gplr50-200k-on) except --layer-slots 2.
# Pre-registered P5 (written before the arm runs):
#   2 slots expose >= 2x A1's ready_wait (>= 7.3 s)  -> 3 slots confirmed as the
#       saturation floor; the default stands.
#   2 slots expose <= 4.0 s                          -> P5 FALSIFIED: 2 slots are
#       as good; the default question reopens for the owner.
#   in between (4.0-7.3 s)                            -> 3 slots still better;
#       default stands with the number recorded.
# tok/s band guard: outside 1464-1490 +/- 3% is recorded as a compute-side
# anomaly, not a staging result (P4 convention).
param(
    [string] $Tag = 'gplr51-s2-2slot'
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

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'bin\Release\llama.dll')).Hash.Substring(0, 16)
$baseline = (Get-Content -LiteralPath (Join-Path $verdictDir 'gplr50-200k-on.verdict.json') -Raw | ConvertFrom-Json).fields
$binaryMatchesBaseline = ($exeSha -eq $baseline.exe_sha256_16 -and $dllSha -eq $baseline.llama_dll_sha256_16)
Write-Host ("[{0}] exe {1} dll {2} matches-A1-binary: {3}" -f $Tag, $exeSha, $dllSha, $binaryMatchesBaseline)

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
       '--layer-slots', '2', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '2',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log; arm = 'S2-slots2'
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
        $fields['resolved_copy_source'] = $rs.copy_source
        if ($rs.copy_source -ne 'mmap-registered') { $fail += "copy_source $($rs.copy_source), want mmap-registered" }

        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

        # run_start carries no layer_slots field (measured on gplr50-200k-on);
        # the in-trace proof of the 2-slot shape is the distinct-slot census
        # over PP copy events instead.
        $t = @(); $slotSet = New-Object 'System.Collections.Generic.HashSet[int]'
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"copy_complete"*' -and $ln -like '*"phase":0*') {
                $e = $ln | ConvertFrom-Json
                $t += [double]$e.transfer_us
                [void]$slotSet.Add([int]$e.slot)
            }
        }
        $fields['distinct_slots_used'] = $slotSet.Count
        if ($slotSet.Count -ne 2) { $fail += "distinct slots used $($slotSet.Count), want 2" }

        $phases = @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                    ForEach-Object { $_ | ConvertFrom-Json })
        $i = 0
        foreach ($p in $phases) {
            $i++
            $fields[("pass{0}_wall_s" -f $i)]    = [math]::Round($p.wall_us / 1e6, 2)
            $fields[("pass{0}_compute_s" -f $i)] = [math]::Round($p.total_compute_us / 1e6, 2)
            $fields[("pass{0}_tok_s" -f $i)]     = [math]::Round($p.tokens_per_s, 1)
        }

        # copy quantiles + exposed wait, PP phase-0 only (T-GPLR-52 F1: never mix
        # phase-1 TG copy events into PP staging stats); $t and $slotSet were
        # filled by the census pass above
        if ($t.Count -gt 0) {
            $sorted = $t | Sort-Object
            $fields['copy_count']   = $t.Count
            $fields['copy_total_s'] = [math]::Round(($t | Measure-Object -Sum).Sum / 1e6, 2)
            $fields['copy_p50_ms']  = [math]::Round($sorted[[int]($t.Count * 0.50)] / 1e3, 2)
            $fields['copy_p90_ms']  = [math]::Round($sorted[[int]($t.Count * 0.90)] / 1e3, 2)
            $fields['copy_max_s']   = [math]::Round(($t | Measure-Object -Maximum).Maximum / 1e6, 2)
        }
        $readyWait = 0.0
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"layer_ready_wait"*') {
                $e = $ln | ConvertFrom-Json
                $readyWait += [double]($(if ($e.wait_us) { $e.wait_us } else { 0 }))
            }
        }
        $fields['ready_wait_total_s'] = [math]::Round($readyWait / 1e6, 2)

        # P5 evaluation against the A1 baseline (3 slots: 3.67 s / max 10.82 s)
        $a1Wait = 3.67
        $w = $fields['ready_wait_total_s']
        $fields['p5_wait_ge_2x']      = [bool]($w -ge (2 * $a1Wait))
        $fields['p5_falsified_le_4s'] = [bool]($w -le 4.0)
        if     ($w -ge (2 * $a1Wait)) { $fields['p5_verdict'] = 'confirmed: 3-slot floor stands' }
        elseif ($w -le 4.0)           { $fields['p5_verdict'] = 'FALSIFIED: 2 slots as good' }
        else                          { $fields['p5_verdict'] = 'partial: 3 slots still better' }
        # tok/s band guard (P4 convention): recorded, not failing
        foreach ($k in @('pass1_tok_s', 'pass2_tok_s')) {
            if ($fields[$k] -and ([math]::Abs($fields[$k] - 1475) / 1475) -gt 0.03) {
                $fields['toks_band_anomaly'] = "$($fields[$k]) outside +/-3% of 1475"
            }
        }
        Write-Host ("  ready_wait {0} s | copy p50 {1} ms p90 {2} ms max {3} s | P5: {4}" -f `
            $w, $fields['copy_p50_ms'], $fields['copy_p90_ms'], $fields['copy_max_s'], $fields['p5_verdict'])
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
