# T-GPLR-36-shape 200k RE-MEASUREMENT on the current binary (post T-45/47/48/49/37/39).
#
# Shape: byte-identical flags to the recorded T-36 hostmem arm
# (tools/run-gplr36-hostmem.ps1: 200k tokens, 262,144 F16 KV config, layer-major,
# ubatch 512, af 1, bf16 slabs, 3 slots, --pp-repeat 2, --pp-recon 1) with ONE
# deliberate difference: NO --copy-source flag, so the arm measures the tree's
# own state (resolves to mmap-registered since T-GPLR-37). That difference is
# the point: the recorded 3,402.3 ms/pass patch figure came from the pinned
# pre-T-45 tree; this arm answers "what is left of the per-launch patch lever
# TODAY" on the binary the owner would actually run.
#
# Recorded references for this shape:
#   T-36 ledger (2026-09-18, pinned, pre-T-45): warm wall 141,847.9 ms,
#     patch 3,402.3 ms/pass (2.40% of the wall; 2,487-2,555 ms in the hostmem rerun)
#   T-45 changed arm (2026-09-19, pinned): patch 18.8-19.0 ms/pass (0.01%)
#   T-47 v3 arm (2026-09-20, pinned, rebuild ON): patch 17.7-18.0 ms/pass (0.01%)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$exe    = Join-Path $base 'bin\Release\llama-layer-streaming.exe'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-200000.txt'
$verdictDir = Join-Path $base 'verdicts'
if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }
if (-not (Test-Path $exe))    { Write-Host "MISSING EXE: $exe"; exit 1 }
. (Join-Path $root 'tools\arm-verdict.ps1')

$tag   = 'gplr36-remeasure-200k'
$trace = Join-Path $base "$tag.jsonl"
$log   = Join-Path $base "$tag.log"
Remove-Item $trace, $log -ErrorAction SilentlyContinue

$exeSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $exe).Hash.Substring(0, 16)
$dllSha = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $base 'bin\Release\llama.dll')).Hash.Substring(0, 16)
Write-Host ("[remeasure] exe {0} dll {1}" -f $exeSha, $dllSha)

$a = @('--model', $model, '--prompt-file', $prompt,
       '--pp-tokens', '262144', '--ubatch', '512', '--pp-no-logits',
       '--tg-tokens', '0',
       '--residency', 'moe-expert-bank', '--kv-cache', 'f16',
       '--act-slab-dtype', 'bf16',
       '--pp-schedule', 'layer-major',
       '--layer-slots', '3', '--allow-full-residency',
       '--residency-budget-mib', '0',
       '--pp-repeat', '2',
       '--pp-async-feed', '1', '--pp-recon', '1',
       '--telemetry', $trace)

Clear-ArmVerdict -Dir $verdictDir -Tag $tag | Out-Null
$started = (Get-Date).ToUniversalTime().ToString('o')
Write-ArmStart -Dir $verdictDir -Tag $tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $log | Out-Null
Write-Host 'START gplr36-remeasure-200k'

$fields = @{ exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha; trace = $trace; log = $log }
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
$exit = 1
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) {
        Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode 124 -StartedUtc $started -LogPath $log -ErrorText 'mutex timeout' -Fields $fields | Out-Null
        exit 124
    }
    Write-Host '  mutex held'
    $fields['command'] = ($exe + ' ' + ($a -join ' '))
    $t0 = Get-Date
    & $exe @a > $log 2>&1
    $armExit = $LASTEXITCODE
    $fields['arm_exit'] = $armExit
    $fields['duration_s'] = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $fields['duration_s'])

    if ($armExit -ne 0) {
        $fields['fail_reason'] = ("arm exited {0}" -f $armExit)
    } else {
        # resolved copy source (must be the mmap-registered default on this tree)
        $rs = @(Get-Content $trace | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
        $fields['copy_source'] = $rs.copy_source
        Write-Host ("  copy_source = {0}" -f $rs.copy_source)

        $ledgers = @(Get-Content $trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                     ForEach-Object { $_ | ConvertFrom-Json })
        $fields['pass_count'] = $ledgers.Count
        $patches = @()
        for ($i = 0; $i -lt $ledgers.Count; $i++) {
            $l = $ledgers[$i]
            $pm = [math]::Round($l.patch_us / 1000.0, 1)
            $patches += $pm
            $fields[("pass{0}_patch_ms" -f ($i + 1))] = $pm
            $fields[("pass{0}_sweep_ms" -f ($i + 1))] = [math]::Round($l.sweep_us / 1000.0, 1)
            $fields[("pass{0}_compute_ms" -f ($i + 1))] = [math]::Round($l.compute_us / 1000.0, 1)
            $fields[("pass{0}_launches" -f ($i + 1))] = $l.launches
            Write-Host ("  pass {0}: sweep {1:N1} ms  patch {2} ms ({3:P2})  launches {4}" -f `
                ($i + 1), ($l.sweep_us / 1000.0), $pm, ($l.patch_us / $l.sweep_us), $l.launches)
        }
        $fields['warm_patch_ms'] = $patches[-1]
        & $exe --check-telemetry $trace
        $fields['check_telemetry_exit'] = $LASTEXITCODE
        Write-Host ("  CHECK EXIT {0}" -f $LASTEXITCODE)

        $fail = @()
        if ($fields.ContainsKey('fail_reason')) { $fail += $fields['fail_reason'] }
        if ($ledgers.Count -ne 2) { $fail += "host_ledger count {0} != 2" -f $ledgers.Count }
        if ($fields['check_telemetry_exit'] -ne 0) { $fail += ("check-telemetry exit {0}" -f $fields['check_telemetry_exit']) }
        if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}
Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $exit -StartedUtc $started -LogPath $log -Fields $fields | Out-Null
Write-Host ("VERDICT {0} ({1}) warm_patch={2} ms -> {3}" -f `
    $(if ($exit -eq 0) { 'PASS' } else { 'FAIL' }), $exit, $fields['warm_patch_ms'], (Join-Path $verdictDir ($tag + '.done')))
exit $exit
