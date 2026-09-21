# T-GPLR-51 arm A2: the never-run same-binary copy-source A/B at the charter
# shape. Shape is gplr50-200k-on's VERBATIM (the remeasure shape, mmap default)
# with ONE deliberate difference: --copy-source pinned. The binary is unchanged
# since that baseline (nested 40bca9f44; no rebuild since), so any delta is the
# copy source, not the build.
#
# Gates applied here (AC1): the GPU busy gate runs via the shared
# Get-GpuComputeApps helper AFTER acquiring the mutex (refuse 125 on a busy
# GPU), and the verdict carries gpu_pre_run for the provenance rule.
#
# Pre-registered targets (from the draft):
#   P1 falsified if pinned ready_wait <= 3.67 s (mmap's recorded exposure)
#   P4 falsified if tok/s leaves 1464-1485 by more than 3%
param([string] $Tag = 'gplr51-a2-pinned')
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
Write-Host ("[a2] exe {0} dll {1}" -f $exeSha, $dllSha)

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
       '--copy-source', 'pinned',
       '--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $Tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host ("START {0}" -f $Tag)

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
             trace = $trace; log = $log; copy_source = 'pinned' }
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

    # GPU busy gate (the hardening from 001bf65): refuse to start on a busy GPU.
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
        if ($rs.copy_source -ne 'pinned') { $fail += "copy_source resolved to $($rs.copy_source), want pinned" }

        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }

        $phases = @(Get-Content $trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                    ForEach-Object { $_ | ConvertFrom-Json })
        $i = 0
        foreach ($p in $phases) {
            $i++
            $fields[("pass{0}_wall_s" -f $i)]      = [math]::Round($p.wall_us / 1e6, 2)
            $fields[("pass{0}_compute_s" -f $i)]   = [math]::Round($p.total_compute_us / 1e6, 2)
            $fields[("pass{0}_transfer_s" -f $i)]  = [math]::Round($p.total_transfer_us / 1e6, 2)
            $fields[("pass{0}_tok_s" -f $i)]       = [math]::Round($p.tokens_per_s, 1)
            $fields[("pass{0}_wait_s" -f $i)]      = [math]::Round($p.total_wait_us / 1e6, 2)
        }

        # copy distribution + the exposed-wait metric (P1's object)
        $t = @(); $lat = @()
        foreach ($ln in (Get-Content $trace)) {
            if ($ln -like '*"event":"copy_complete"*') {
                $e = $ln | ConvertFrom-Json
                $t += [double]$e.transfer_us
                $lat += [double]$e.ready_latency_us
            }
        }
        if ($t.Count -gt 0) {
            $sorted = $t | Sort-Object
            $fields['copy_count'] = $t.Count
            $fields['copy_total_s'] = [math]::Round(($t | Measure-Object -Sum).Sum / 1e6, 2)
            $fields['copy_p50_ms']  = [math]::Round($sorted[[int]($t.Count / 2)] / 1e3, 2)
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
        Write-Host ("  ready_wait total {0:N2} s | copy p50 {1} ms max {2} s" -f `
            $fields['ready_wait_total_s'], $fields['copy_p50_ms'], $fields['copy_max_s'])
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
