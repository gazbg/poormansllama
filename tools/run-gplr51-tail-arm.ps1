# T-GPLR-51 arms A3/A4: the copy-tail probes at the charter shape, against the
# A1 baseline (gplr50-200k-on: ready_wait 3.67 s, copy p50 31.5 ms, max 10.82 s).
#   A3: --layer-slots 4      (deeper staging pipeline; P2: exposed wait < 2 s
#                             AND tail max shrunk >= 30% -> <= 7.6 s)
#   A4: --pp-async-feed 0    (hiding ablation;      P3: exposed wait >= 10x A1,
#                             ~40 s class; falsified below 5 s)
# Shape otherwise VERBATIM vs A1 (mmap default, f16, 3-or-4 slots, repeat 2,
# tg 3). Same binary as the baseline (verified at launch). Predictions are
# EVALUATED in the verdict fields; the arm itself passes on clean data only.
param(
    [Parameter(Mandatory = $true)][ValidateSet('A3', 'A4')][string] $Arm,
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

if (-not $Tag) { $Tag = "gplr51-$($Arm.ToLower())" }
$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'bin\Release\llama.dll')).Hash.Substring(0, 16)
# baseline identity (A2's lesson: the comparison is only same-binary if the
# hashes match the A1 verdict's)
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
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '2',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)
if ($Arm -eq 'A3') { $a = @($a | Where-Object { $_ -ne '3' -or $true }); $a = @(); $a = $null }
# rebuild the argv explicitly per arm (simpler than splicing):
if ($Arm -eq 'A3') {
    $a = @('--model', $model, '--prompt-file', $prompt,
           '--pp-tokens', '200000', '--ctx-size', '200704',
           '--ubatch', '512', '--pp-no-logits',
           '--tg-tokens', '3',
           '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
           '--act-slab-dtype', 'bf16',
           '--pp-schedule', 'layer-major',
           '--layer-slots', '4', '--allow-full-residency',
           '--residency-budget-mib', '0',
           '--pp-repeat', '2',
           '--pp-async-feed', '1', '--pp-recon', '1',
           '--telemetry', $trace)
} else {
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
           '--pp-async-feed', '0', '--pp-recon', '1',
           '--telemetry', $trace)
}

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log; arm = $Arm
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
        # arm-specific resolution checks
        if ($Arm -eq 'A3' -and $rs.copy_source -ne 'mmap-registered') { $fail += "A3 copy_source $($rs.copy_source), want mmap-registered (default)" }

        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

        $phases = @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                    ForEach-Object { $_ | ConvertFrom-Json })
        $i = 0
        foreach ($p in $phases) {
            $i++
            $fields[("pass{0}_wall_s" -f $i)]    = [math]::Round($p.wall_us / 1e6, 2)
            $fields[("pass{0}_compute_s" -f $i)] = [math]::Round($p.total_compute_us / 1e6, 2)
            $fields[("pass{0}_transfer_s" -f $i)] = [math]::Round($p.total_transfer_us / 1e6, 2)
            $fields[("pass{0}_tok_s" -f $i)]     = [math]::Round($p.tokens_per_s, 1)
        }

        # copy distribution incl. the tail quantiles the draft's P2 needs
        $t = @(); $lat = @()   # T-GPLR-52 F1/AC1: PP phase-0 only - TG copy events must not pollute PP staging stats
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"copy_complete"*' -and $ln -like '*"phase":0,*') {
                $e = $ln | ConvertFrom-Json
                $t += [double]$e.transfer_us
                $lat += [double]$e.ready_latency_us
            }
        }
        if ($t.Count -gt 0) {
            $sorted = $t | Sort-Object
            $fields['copy_count']   = $t.Count
            $fields['copy_total_s'] = [math]::Round(($t | Measure-Object -Sum).Sum / 1e6, 2)
            $fields['copy_p50_ms']  = [math]::Round($sorted[[int]($t.Count * 0.50)] / 1e3, 2)
            $fields['copy_p90_ms']  = [math]::Round($sorted[[int]($t.Count * 0.90)] / 1e3, 2)
            $fields['copy_p99_ms']  = [math]::Round($sorted[[Math]::Min($t.Count - 1, [int]($t.Count * 0.99))] / 1e3, 2)
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

        # prediction evaluation (A1 baseline values from gplr50-200k-on)
        $a1Wait = 3.67; $a1Max = 10.82
        if ($Arm -eq 'A3') {
            $fields['p2_wait_under_2s']   = [bool]($fields['ready_wait_total_s'] -lt 2.0)
            $fields['p2_max_shrunk_30pc'] = [bool]($fields['copy_max_s'] -le ($a1Max * 0.70))
            $fields['p2_met'] = [bool]($fields['p2_wait_under_2s'] -and $fields['p2_max_shrunk_30pc'])
        } else {
            $fields['p3_wait_ge_10x'] = [bool]($fields['ready_wait_total_s'] -ge (10 * $a1Wait))
            $fields['p3_met'] = $fields['p3_wait_ge_10x']
            if ($fields['ready_wait_total_s'] -lt 5.0) { $fields['p3_falsified_below_5s'] = $true }
        }
        Write-Host ("  ready_wait {0} s | copy p50 {1} ms p99 {2} ms max {3} s" -f `
            $fields['ready_wait_total_s'], $fields['copy_p50_ms'], $fields['copy_p99_ms'], $fields['copy_max_s'])
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
