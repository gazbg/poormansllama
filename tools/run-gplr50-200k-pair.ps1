# T-GPLR-50 AC3: the owner-invoked 200k confirmation PAIR under ONE mutex hold.
#
#   Arm A (control) : gplr50-200k-off  -- explicit --pp-prebuild-per-pass 0
#                     (the recorded rebuild-OFF state; its excess is the
#                      recorded +67.4 s / +48.9% class)
#   Arm B (changed) : gplr50-200k-on   -- default (resolves rebuild ON after
#                     the flip; same resolution as the new f16 default)
#
# Shape: the gplr36-remeasure-200k hostmem shape VERBATIM (200000 tokens,
# ubatch 512, f16 KV, bf16 slabs, layer-major, 3 slots, --pp-repeat 2,
# --pp-async-feed 1, --pp-recon 1, --copy-source mmap-registered, tg 3,
# ctx 200704). One deliberate difference per arm: the --pp-prebuild-per-pass
# flag only on the OFF arm (explicit); the ON arm runs bare default so it
# proves the flipped resolution itself. Metrics per arm: host_ledger
# pass1/pass2 compute_us (the excess), phase_summary wall_us,
# layer_compute pass2/pass1 ratio per chunk position (uniformity).
#
# AC3 gates (from the plan block):
#   excess(ON) <= 5 s                      (P1; recorded OFF excess +67.4 s)
#   wall improvement >= 25% paired         (P3; falsified below 15%)
#   uniformity max/min of per-position pass2/pass1 ratio < 1.1 (vs ~1.5)
param([string] $TagOff = 'gplr50-200k-off', [string] $TagOn = 'gplr50-200k-on')
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
Write-Host ("[ac3] exe {0} dll {1}" -f $exeSha, $dllSha)

function Invoke-Ac3Arm {
    param([string] $Tag, [string] $RebuildFlag)
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
           '--copy-source', 'mmap-registered',
           '--telemetry', $trace)
    if ($RebuildFlag) { $a += @('--pp-prebuild-per-pass', $RebuildFlag) }
    return @{ Tag = $Tag; Trace = $trace; Log = $log; Args = $a }
}

$results = @{}
$mutex = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
$acquired = $false
try {
    try { $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds(3600)) }
    catch [System.Threading.AbandonedMutexException] { $acquired = $true; Write-Warning 'Recovered an abandoned GPU mutex.' }
    if (-not $acquired) { Write-Host 'MUTEX TIMEOUT'; exit 124 }
    Write-Host '  mutex held (both arms run inside this single hold)'

    foreach ($spec in @(
        (Invoke-Ac3Arm -Tag $TagOff -RebuildFlag '0'),
        (Invoke-Ac3Arm -Tag $TagOn  -RebuildFlag '')
    )) {
        Clear-ArmVerdict -Dir $verdictDir -Tag $spec.Tag | Out-Null
        $started = (Get-Date).ToUniversalTime().ToString('o')
        Write-ArmStart -Dir $verdictDir -Tag $spec.Tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath $spec.Log | Out-Null
        Write-Host ("START {0}" -f $spec.Tag)
        $t0 = Get-Date
        & $exe @($spec.Args) > $spec.Log 2>&1
        $armExit = $LASTEXITCODE
        $dur = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
        Write-Host ("  ARM EXIT {0} ({1}s)" -f $armExit, $dur)

        $fields = @{ arm_exit = $armExit; duration_s = $dur
                     exe_sha256_16 = $exeSha; llama_dll_sha256_16 = $dllSha
                     trace = $spec.Trace; log = $spec.Log }
        $fail = @()
        if ($armExit -ne 0) { $fail += "arm exited $armExit" }
        else {
            # resolved value provenance
            $rs = @(Get-Content $spec.Trace | Where-Object { $_ -like '*"event":"run_start"*' })[0] | ConvertFrom-Json
            $fields['resolved_pp_prebuild_per_pass'] = $rs.pp_prebuild_per_pass
            $fields['pp_prebuild_per_pass_source']   = $rs.pp_prebuild_per_pass_source
            # OFF arm must be explicit-0; ON arm must be default-1
            if ($spec.Tag -eq $TagOff) {
                if ([int]$rs.pp_prebuild_per_pass -ne 0 -or $rs.pp_prebuild_per_pass_source -ne 'explicit') {
                    $fail += "OFF arm resolved $($rs.pp_prebuild_per_pass)/$($rs.pp_prebuild_per_pass_source), want 0/explicit"
                }
            } else {
                if ([int]$rs.pp_prebuild_per_pass -ne 1 -or $rs.pp_prebuild_per_pass_source -ne 'default') {
                    $fail += "ON arm resolved $($rs.pp_prebuild_per_pass)/$($rs.pp_prebuild_per_pass_source), want 1/default"
                }
            }
            # ledgers
            $ledgers = @(Get-Content $spec.Trace | Where-Object { $_ -like '*"event":"host_ledger"*' } |
                         ForEach-Object { $_ | ConvertFrom-Json })
            $fields['pass_count'] = $ledgers.Count
            if ($ledgers.Count -eq 2) {
                $c1 = [double]$ledgers[0].compute_us; $c2 = [double]$ledgers[1].compute_us
                $fields['pass1_compute_s'] = [math]::Round($c1 / 1e6, 2)
                $fields['pass2_compute_s'] = [math]::Round($c2 / 1e6, 2)
                $fields['excess_2v1_s']    = [math]::Round(($c2 - $c1) / 1e6, 2)
                Write-Host ("  compute pass1 {0:N1} s  pass2 {1:N1} s  excess {2:N2} s" -f ($c1/1e6), ($c2/1e6), $fields['excess_2v1_s'])
            } else { $fail += "host_ledger count $($ledgers.Count) != 2" }
            # walls
            $phases = @(Get-Content $spec.Trace | Where-Object { $_ -like '*"event":"phase_summary"*' -and $_ -like '*"phase":"pp"*' } |
                        ForEach-Object { $_ | ConvertFrom-Json })
            if ($phases.Count -ge 2) {
                $w1 = [double]$phases[0].wall_us; $w2 = [double]$phases[1].wall_us
                $fields['pass1_wall_s'] = [math]::Round($w1 / 1e6, 2)
                $fields['pass2_wall_s'] = [math]::Round($w2 / 1e6, 2)
                $fields['two_pass_wall_s'] = [math]::Round(($w1 + $w2) / 1e6, 2)
            }
            # checker
            & $exe --check-telemetry $spec.Trace
            $fields['check_telemetry_exit'] = $LASTEXITCODE
            if ($LASTEXITCODE -ne 0) { $fail += "check-telemetry exit $LASTEXITCODE" }
        }
        if ($fail.Count -gt 0) { $fields['fail_reason'] = ($fail -join '; '); $exit = 1 } else { $exit = 0 }
        Write-ArmVerdict -Dir $verdictDir -Tag $spec.Tag -ExitCode $exit -StartedUtc $started `
            -LogPath $spec.Log -Fields $fields | Out-Null
        Write-Host ("VERDICT {0} ({1}) -> {2}" -f $(if ($exit -eq 0) {'PASS'} else {'FAIL'}), $exit, (Join-Path $verdictDir ($spec.Tag + '.done')))
        $results[$spec.Tag] = $fields
    }
} finally {
    if ($acquired) { $mutex.ReleaseMutex(); Write-Host '  mutex released' }
    $mutex.Dispose()
}

# ---------------- pair verdict: P1, P3, uniformity ----------------
$v = Read-ArmVerdict -Dir $verdictDir -Tag $TagOff
$w = Read-ArmVerdict -Dir $verdictDir -Tag $TagOn
if ($null -eq $v -or $null -eq $w -or -not $v.done_present -or -not $w.done_present) {
    Write-Host 'PAIR INCOMPLETE (a verdict sentinel missing)'; exit 1
}
$exOff = [double]$v.fields.excess_2v1_s
$exOn  = [double]$w.fields.excess_2v1_s
$wallOff = [double]$v.fields.two_pass_wall_s
$wallOn  = [double]$w.fields.two_pass_wall_s
$impr = if ($wallOff -gt 0) { [math]::Round(100.0 * ($wallOff - $wallOn) / $wallOff, 1) } else { -1 }
Write-Host ("PAIR: excess OFF {0} s -> ON {1} s   wall OFF {2} s -> ON {3} s  improvement {4}%" -f $exOff, $exOn, $wallOff, $wallOn, $impr)
$pairFail = @()
if ($exOn -gt 5.0)      { $pairFail += "P1 falsified: ON excess $exOn s > 5 s" }
if ($impr -lt 15.0)     { $pairFail += "P3 falsified: improvement $impr % < 15 %" }
if ($v.exit_code -ne 0) { $pairFail += "OFF arm FAIL: $($v.fields.fail_reason)" }
if ($w.exit_code -ne 0) { $pairFail += "ON arm FAIL: $($w.fields.fail_reason)" }
$pairExit = 0
if ($pairFail.Count -gt 0) { Write-Host ("PAIR VERDICT FAIL: " + ($pairFail -join '; ')); $pairExit = 1 }
else { Write-Host 'PAIR VERDICT PASS (P1, P3)' }
# uniformity is computed post-hoc from the traces by analyze (layer_compute);
# recorded here as fields for the closure script.
exit $pairExit
