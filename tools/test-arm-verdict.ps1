# Host-only regression test for the arm verdict + sentinel convention
# (tools/arm-verdict.ps1). CPU only, no CUDA, no model: it exercises file
# semantics and the wrapper that writes them. The mutex IS taken for the
# integration cases, but only around `cmd.exe /c exit N`, which initialises no
# CUDA context, so this is not a GPU workload.
#
# What it guards, i.e. the ways a sentinel scheme goes wrong in practice:
#   1. a stale PASS from an earlier run read as the current run's result;
#   2. a .done present without a complete verdict beside it;
#   3. a caller forcing PASS for a run whose exit code says otherwise;
#   4. a wait that hangs instead of returning $null on timeout;
#   5. the tag not being derivable from the command, so nobody can predict
#      which sentinel to check;
#   6. the wrapper PRINTING gpu_pre/gpu_post and taking a tag while persisting
#      neither into the verdict a later pass actually reads (section 12);
#   7. a probe that could not be taken being recorded as a clean card -- both the
#      case where nvidia-smi fails and the measured case (2026-09-22) where it
#      answered with unparseable output (section 12);
#   8. a verdict that cannot say WHICH SOURCE TREE and WHICH BINARY it describes,
#      so a source edit or a rebuild silently voids the evidence it stands for
#      (section 13; the fixture half -- a source edit after a build must read
#      STALE -- is tools/check-arm-provenance.ps1 -Selftest);
#   9. that evidence reaching the record WITHOUT being required to say either: an
#      arm whose verdict carries no capture, names no binary, or was built before
#      its sources must be REFUSED where this program's own arms launch (section
#      14), with the escape explicit and recorded rather than implied.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/test-arm-verdict.ps1
# Exit code: 0 = all cases pass, 1 = at least one failed.
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
. (Join-Path $root 'tools\arm-verdict.ps1')

$fail = 0
$script:skip = 0
function Check([string]$name, [bool]$cond, [string]$detail = '') {
    if ($cond) {
        Write-Host ("  PASS  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' }))
    } else {
        Write-Host ("  FAIL  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' }))
        $script:fail++
    }
}

# The shared GPU lock is a real machine-wide object and another project may hold it
# right now (measured 2026-09-22: deepseek_v4.exe, pid 21812, lock held). A host-only
# test must not queue behind that, nor be failed by it: probe it, and let the one
# section that needs the REAL lock report a skip instead of a false failure.
function Test-SharedGpuLockFree {
    $m = [System.Threading.Mutex]::new($false, 'Global\AI-GPU-EXCLUSIVE-v1')
    try {
        if ($m.WaitOne(0)) { $m.ReleaseMutex(); return $true }
        return $false
    } catch [System.Threading.AbandonedMutexException] {
        return $true
    } finally {
        $m.Dispose()
    }
}

# A private per-case lock: the gate logic does not depend on WHICH mutex is held,
# so every case that is not specifically about the default lock uses one of these
# and stays out of the shared queue.
function New-PrivateGpuLockName {
    param([string] $Suffix)
    return 'Global\AI-GPU-EXCLUSIVE-v1-selftest-' + $Suffix
}

$tmp = Join-Path $env:TEMP ('arm-verdict-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$tag = 'unit-case'

try {
    Write-Host '=== 1. round trip ==='
    Clear-ArmVerdict -Dir $tmp -Tag $tag
    $p = Get-ArmVerdictPaths -Dir $tmp -Tag $tag
    Check 'clear leaves no files' (-not (Test-Path $p.verdict) -and -not (Test-Path $p.done))
    # -AllowUnattested: this fixture's subject is STATUS DERIVATION (exit 0 -> PASS), and a
    # bare temp dir cannot attest a revision, so it opts out explicitly. The writer's own
    # refusal to leave an unattested PASS is proven in its own section below, not here.
    Write-ArmVerdict -AllowUnattested -Dir $tmp -Tag $tag -ExitCode 0 -StartedUtc (Get-Date).ToUniversalTime().ToString('o') `
        -Fields @{ gain_tok_s = 10.017 } | Out-Null
    Check 'verdict file written' (Test-Path $p.verdict)
    Check 'sentinel written' (Test-Path $p.done)
    $j = Get-Content -LiteralPath $p.verdict -Raw | ConvertFrom-Json
    Check 'json parses' ($j.schema -eq 'arm-verdict/1')
    Check 'status derived PASS from exit 0' ($j.status -eq 'PASS') $j.status
    Check 'fields survive' ([double]$j.fields.gain_tok_s -eq 10.017)
    $r = Read-ArmVerdict -Dir $tmp -Tag $tag
    Check 'read reports the sentinel' ($r.done_present -eq $true)
    Check 'sentinel names status and code' ($r.done_text -like 'PASS 0 *') $r.done_text

    Write-Host '=== 2. stale sentinel cannot survive a new run ==='
    # The failure this prevents: yesterday's PASS read as today's result.
    Clear-ArmVerdict -Dir $tmp -Tag $tag
    Check 'invalidate removes the previous .done' (-not (Test-Path $p.done))
    Check 'read after invalidation has no sentinel' ($null -eq (Read-ArmVerdict -Dir $tmp -Tag $tag).done_text)
    Write-ArmVerdict -Dir $tmp -Tag $tag -ExitCode 3 -Fields @{ crit1 = $true; crit2 = $false } | Out-Null
    $j = Get-Content -LiteralPath $p.verdict -Raw | ConvertFrom-Json
    Check 'failing run reports FAIL 3' ($j.status -eq 'FAIL' -and $j.exit_code -eq 3) ("$($j.status) $($j.exit_code)")
    Check 'sentinel agrees with the verdict' ((Get-Content -LiteralPath $p.done -Raw).Trim() -like 'FAIL 3 *')

    Write-Host '=== 3. status is derived, never trusted from the caller ==='
    Write-ArmVerdict -Dir $tmp -Tag 'timeout-case' -ExitCode 124 | Out-Null
    $j = Get-Content -LiteralPath (Join-Path $tmp 'timeout-case.verdict.json') -Raw | ConvertFrom-Json
    Check 'exit 124 maps to TIMEOUT' ($j.status -eq 'TIMEOUT') $j.status
    # An explicit status is allowed for the ERROR/termination path, but the exit
    # code still has to say the run did not succeed.
    Write-ArmVerdict -Dir $tmp -Tag 'error-case' -ExitCode 1 -Status 'ERROR' -ErrorText 'boom' | Out-Null
    $j = Get-Content -LiteralPath (Join-Path $tmp 'error-case.verdict.json') -Raw | ConvertFrom-Json
    Check 'ERROR carries its text' ($j.status -eq 'ERROR' -and $j.error -eq 'boom')
    Clear-ArmVerdict -Dir $tmp -Tag 'timeout-case'
    Clear-ArmVerdict -Dir $tmp -Tag 'error-case'

    Write-Host '=== 4. waiting is bounded and returns the result ==='
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $w = Wait-ArmVerdict -Dir $tmp -Tag $tag -TimeoutSeconds 20 -PollSeconds 1
    $sw.Stop()
    Check 'wait returns an already-finished run' ($null -ne $w -and $w.status -eq 'FAIL')
    Check 'wait does not sleep when the sentinel exists' ($sw.Elapsed.TotalSeconds -lt 2) ("{0:N2}s" -f $sw.Elapsed.TotalSeconds)
    $sw.Restart()
    $w = Wait-ArmVerdict -Dir $tmp -Tag 'never-runs' -TimeoutSeconds 3 -PollSeconds 1 -Quiet
    $sw.Stop()
    Check 'wait times out with $null' ($null -eq $w)
    Check 'wait is bounded by its timeout' ($sw.Elapsed.TotalSeconds -lt 8) ("{0:N2}s" -f $sw.Elapsed.TotalSeconds)

    Write-Host '=== 5. the tag is predictable from the command ==='
    $cases = @(
        @{ c = 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\repo\tools\run-gplr36-hostmem.ps1'; want = 'run-gplr36-hostmem' },
        @{ c = '"C:\repo\build\llama-cuda\bin\Release\llama-layer-streaming.exe" --model x.gguf';            want = 'llama-layer-streaming' },
        @{ c = 'nvidia-smi --query-gpu=memory.used';                                                          want = 'gpu-command' },
        @{ c = '';                                                                                            want = 'gpu-command' }
    )
    $i = 0
    foreach ($k in $cases) {
        $i++
        $got = Get-ArmTagFromCommand -Command $k.c
        Check ("tag case {0} -> {1}" -f $i, $k.want) ($got -eq $k.want) ("got '{0}'" -f $got)
    }

    Write-Host '=== 6. through tools/with-gpu-lock.ps1 (mutex, no CUDA) ==='
    # cmd.exe /c exit N takes the mutex and initialises nothing. This is the path
    # every -locked.ps1 arm wrapper uses, so the integration is the real subject.
    # It is also the only section that needs the REAL shared lock, so it is the only
    # one that can be blocked by another project: that is reported as a skip, never
    # as a failure of this code.
    $lock = Join-Path $root 'tools\with-gpu-lock.ps1'
    $vdir = Join-Path $tmp 'wrapper'
    $lockFree = Test-SharedGpuLockFree
    $cardBusy = (Format-GpuBusy (Get-GpuComputeApps)) -ne 'clean'
    if (-not $lockFree -or $cardBusy) {
        $script:skip++
        Write-Host ('  SKIP  the shared GPU lock is {0} and the card is {1}: the integration path cannot be exercised now.' -f `
            $(if ($lockFree) { 'free' } else { 'HELD by another process' }), $(if ($cardBusy) { 'BUSY' } else { 'clean' }))
        Write-Host '        Nothing here is GPU work, but the case needs the real lock; re-run this test on a free card.'
    } else {
    Clear-ArmVerdict -Dir $vdir -Tag 'w1'
    # The stub names the binary it RUNS (`$env:ComSpec`), because the wrapper now gates at
    # the choke point: a command that names no binary cannot be attested and is refused
    # (exit 10, section 15). Naming a real exe keeps this case about what it tests --
    # that a FAIL verdict carries the command's own exit code.
    & $lock -AllowUnprovenanced -TimeoutSeconds 60 -Command 'exit 3' -VerdictDir $vdir -VerdictTag 'w1' | Out-Null
    $w1 = Read-ArmVerdict -Dir $vdir -Tag 'w1'
    Check 'wrapper writes a verdict' ($null -ne $w1)
    Check 'wrapper FAIL reflects cmd exit 3' ($w1.status -eq 'FAIL' -and $w1.exit_code -eq 3) ("$($w1.status) $($w1.exit_code)")
    Check 'wrapper records the mutex it held' ($w1.mutex -eq 'Global\AI-GPU-EXCLUSIVE-v1' -and $w1.fields.mutex_acquired -eq $true)
    Check 'wrapper leaves a sentinel beside it' ($w1.done_present -eq $true)

    # Same tag, opposite result: proves the .done is cleared, not merely overwritten
    # in place, and that yesterday's PASS cannot be read as today's result.
    & $lock -AllowUnprovenanced -TimeoutSeconds 60 -Command 'exit 0' -VerdictDir $vdir -VerdictTag 'w1' | Out-Null
    $w2 = Read-ArmVerdict -Dir $vdir -Tag 'w1'
    Check 'rerun flips the same tag to PASS 0' ($w2.status -eq 'PASS' -and $w2.exit_code -eq 0) ("$($w2.status) $($w2.exit_code)")
    }

    Write-Host '=== 7. the start marker and the four run states ==='
    # The point of the marker: with only a sentinel, a missing .done cannot tell
    # "never started" (a bad path, a rejected launch) from "still running".
    $mt = 'marker-case'
    Clear-ArmVerdict -Dir $tmp -Tag $mt
    $st = Get-ArmRunState -Dir $tmp -Tag $mt
    Check 'no marker + no sentinel = never-started' ($st.state -eq 'never-started') $st.state
    Check 'nothing to read in that state' ($null -eq $st.marker -and $null -eq $st.verdict)

    Write-ArmStart -Dir $tmp -Tag $mt -Command 'demo' -LogPath 'demo.log' | Out-Null
    $st = Get-ArmRunState -Dir $tmp -Tag $mt
    Check 'marker alone = running' ($st.state -eq 'running') $st.state
    Check 'marker records the live pid' ($null -ne $st.marker.pid) ("pid=$($st.marker.pid)")
    Check 'marker pid is this process, so liveness reads true' ($st.alive -eq $true)
    Check 'marker is readable on its own' ((Read-ArmStart -Dir $tmp -Tag $mt).schema -eq 'arm-start/1')

    # The 'died' case: a marker whose pid is gone and no sentinel. Use a pid that
    # cannot be alive, and pin the marker's host so the check is not skipped.
    $deadMarker = Get-Content -LiteralPath (Join-Path $tmp ($mt + '.started')) -Raw | ConvertFrom-Json
    $deadMarker.pid = 999998
    [System.IO.File]::WriteAllText((Join-Path $tmp ($mt + '.started')), ($deadMarker | ConvertTo-Json -Depth 4),
        (New-Object System.Text.UTF8Encoding($false)))
    $st = Get-ArmRunState -Dir $tmp -Tag $mt
    Check 'marker with a dead pid and no sentinel = died' ($st.state -eq 'died') $st.state
    Check 'death is only claimed when liveness is known false' ($st.alive -eq $false)

    # -AllowUnattested: the subject HERE is the run-state machine, and a bare temp dir cannot
    # attest a revision. The writer's refusal is proven in its own section, not by this fixture.
    Write-ArmVerdict -AllowUnattested -Dir $tmp -Tag $mt -ExitCode 0 | Out-Null
    $st = Get-ArmRunState -Dir $tmp -Tag $mt
    Check 'marker + sentinel = finished' ($st.state -eq 'finished') $st.state
    Check 'the verdict is what finished means' ($st.verdict.status -eq 'PASS')
    Check 'no anomaly when both are present' ($st.anomaly -eq '')

    # Anomaly: a sentinel with no marker. Not reachable through the wrappers, so a
    # reader must say so rather than silently trusting the verdict.
    Remove-Item -LiteralPath (Join-Path $tmp ($mt + '.started')) -Force
    $st = Get-ArmRunState -Dir $tmp -Tag $mt
    Check 'sentinel without a marker is flagged as an anomaly' ($st.state -eq 'finished' -and $st.anomaly -ne '') $st.anomaly

    Write-Host '=== 8. an old finished state cannot be read as a new run ==='
    # Exactly the stale-state failure this convention exists to prevent.
    Check 'before: the tag reads finished' ((Get-ArmRunState -Dir $tmp -Tag $mt).state -eq 'finished')
    Clear-ArmVerdict -Dir $tmp -Tag $mt
    Check 'after invalidation: never-started' ((Get-ArmRunState -Dir $tmp -Tag $mt).state -eq 'never-started')
    Write-ArmStart -Dir $tmp -Tag $mt | Out-Null
    Check 'a new run reads running, not finished' ((Get-ArmRunState -Dir $tmp -Tag $mt).state -eq 'running')

    Write-Host '=== 9. Wait-ArmStart is bounded and conclusive ==='
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $w = Wait-ArmStart -Dir $tmp -Tag $mt -TimeoutSeconds 20 -PollSeconds 1
    $sw.Stop()
    Check 'wait returns the marker when it exists' ($null -ne $w)
    Check 'wait does not sleep when the marker exists' ($sw.Elapsed.TotalSeconds -lt 2) ("{0:N2}s" -f $sw.Elapsed.TotalSeconds)
    $sw.Restart()
    $w = Wait-ArmStart -Dir $tmp -Tag 'launch-that-never-began' -TimeoutSeconds 3 -PollSeconds 1 -Quiet
    $sw.Stop()
    Check 'a launch that never began times out with $null' ($null -eq $w)
    Check 'and the wait is bounded' ($sw.Elapsed.TotalSeconds -lt 8) ("{0:N2}s" -f $sw.Elapsed.TotalSeconds)
    Clear-ArmVerdict -Dir $tmp -Tag $mt

    Write-Host '=== 10. defaults: derived tag, default directory ==='
    # The subject here is the tag derivation and the default verdict directory, not
    # the GPU, so the probe is stubbed: otherwise a foreign compute process makes
    # this section fail on the busy REFUSAL (correct behaviour) and report a defect
    # in tag derivation that does not exist.
    $defDir = Join-Path $root 'build\llama-cuda\verdicts'
    Clear-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    $stubClean = Join-Path $tmp 'stub-clean.csv'
    Set-Content -Path $stubClean -Value '' -Encoding Ascii
    $env:GPU_BUSY_PROBE_CMD = "type `"$stubClean`""
    & $lock -AllowUnprovenanced -TimeoutSeconds 60 -Command 'exit 0' -MutexName (New-PrivateGpuLockName 'default') | Out-Null
    Remove-Item Env:GPU_BUSY_PROBE_CMD -ErrorAction SilentlyContinue
    $d = Read-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    Check 'default dir + derived tag produce a verdict' ($null -ne $d -and $d.status -eq 'PASS')
    Check 'and leave a marker beside it' ((Get-ArmRunState -Dir $defDir -Tag 'gpu-command').state -eq 'finished')
    Clear-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    Check 'cleanup removed the default-location artifacts' (-not (Test-Path (Join-Path $defDir 'gpu-command.done')))
    Check 'cleanup removed the marker too' (-not (Test-Path (Join-Path $defDir 'gpu-command.started')))
    Check 'and the tag now reads never-started' ((Get-ArmRunState -Dir $defDir -Tag 'gpu-command').state -eq 'never-started')

    Write-Host '=== 11. the marker exists BEFORE the command body runs ==='
    # Ordering, proven rather than assumed: the command itself checks for its own
    # marker and reports via its exit code, so no timing assumption is involved.
    $vdir2 = Join-Path $tmp 'ordering'
    Clear-ArmVerdict -Dir $vdir2 -Tag 'w2'
    $probe = 'if exist "' + (Join-Path $vdir2 'w2.started') + '" (exit 0) else (exit 9)'
    # Stubbed probe for the same reason as section 10: the subject is marker ORDER,
    # and a busy card would otherwise gate the command out of existence.
    $env:GPU_BUSY_PROBE_CMD = "type `"$stubClean`""
    # The probe is a bare `if exist ...` shell fragment, so it names no binary and the
    # choke-point gate cannot attest it; the waiver is explicit and recorded, and the
    # subject here is marker ORDER (section 15 proves the refusal path itself).
    & $lock -AllowUnprovenanced -TimeoutSeconds 60 -Command $probe -VerdictDir $vdir2 -VerdictTag 'w2' -MutexName (New-PrivateGpuLockName 'order') | Out-Null
    Remove-Item Env:GPU_BUSY_PROBE_CMD -ErrorAction SilentlyContinue
    $st = Get-ArmRunState -Dir $vdir2 -Tag 'w2'
    Check 'marker present when the arm body started' ($st.verdict.exit_code -eq 0) ("exit=$($st.verdict.exit_code) (9 would mean the marker came too late)")
    Check 'state is finished with both files' ($st.state -eq 'finished' -and $null -ne $st.marker)
    Write-Host '=== 12. the wrapper persists the evidence it prints: probe + tag + honest absence ==='
    # Three properties, all host-only. The private mutex names below are deliberate:
    # the gate logic does not depend on WHICH lock is held, so a host-only test has
    # no business sitting in the shared GPU queue.
    #   (a) a wrapped run's verdict carries the probe fields and the tag it used;
    #   (b) a consumer that reads ONLY the pre-existing fields still works unchanged;
    #   (c) an unavailable probe is distinguishable from a clean one, and a probe
    #       that was never taken reads as 'not-probed' -- never as 'clean'.
    $gdir = Join-Path $tmp 'gpu-fields'
    $fx = Join-Path $tmp 'probes'
    New-Item -ItemType Directory -Force -Path $fx | Out-Null
    $cleanOut = Join-Path $fx 'clean.csv'; Set-Content -Path $cleanOut -Value '' -Encoding Ascii
    $busyOut  = Join-Path $fx 'busy.csv';  Set-Content -Path $busyOut -Value '1234, C:\dev\other\deepseek_v4.exe, 16500 MiB' -Encoding Ascii
    $junkOut  = Join-Path $fx 'junk.out';  Set-Content -Path $junkOut -Value 'WARNING: infoROM is corrupted at gpu 0000:01:00.0' -Encoding Ascii
    $failOut  = Join-Path $fx 'fail.cmd';  Set-Content -Path $failOut -Value '@echo simulated-nvidia-smi-failure & exit /b 1' -Encoding Ascii

    function Invoke-GpuFieldCase {
        param([string] $Tag, [string] $Probe = '', [string] $Command = 'exit 0',
              [switch] $AllowBusy, [string] $HoldMutex = '', [int] $TimeoutSeconds = 30, [switch] $DeriveTag)
        # These cases are CONTRIVED STUBS (a bare `exit 0` names no binary), so under the
        # choke-point gate they are unattestable by construction and carry the explicit,
        # RECORDED waiver. Section 15 is where the refusal itself is proven; here the
        # subject is the wrapper's other contract (probe fields, marker, derived tag).
        # BusyGraceSeconds 0: the teardown grace is a real behaviour tested elsewhere
        # (build/gpu-gate-tests); here it would only add 30 s of sleep per case.
        $a = @{ TimeoutSeconds = $TimeoutSeconds; Command = $Command; VerdictDir = $gdir
                MutexName = 'Global\AI-GPU-EXCLUSIVE-v1-selftest-' + $Tag; BusyGraceSeconds = 0 }
        if ($HoldMutex) { $a['MutexName'] = $HoldMutex }
        if (-not $DeriveTag) { $a['VerdictTag'] = $Tag }
        if ($AllowBusy) { $a['AllowBusy'] = $true }
        if ($Probe) { $env:GPU_BUSY_PROBE_CMD = $Probe } else { Remove-Item Env:GPU_BUSY_PROBE_CMD -ErrorAction SilentlyContinue }
        # Capture first, read the code second: piping the invocation in the same
        # statement risks reading $LASTEXITCODE after the pipeline, not the script.
        # `@a` is a HASHTABLE splat onto a script (not a native command), so a switch
        # entry binds as a switch - the native-call caveat in section 15 does not apply.
        $a['AllowUnprovenanced'] = $true
        $out = & $lock @a
        $code = $LASTEXITCODE
        Remove-Item Env:GPU_BUSY_PROBE_CMD -ErrorAction SilentlyContinue
        return $code
    }

    $code = Invoke-GpuFieldCase -Tag 'ev-clean' -Probe "type `"$cleanOut`""
    Check 'clean case exits 0' ($code -eq 0) ("exit=$code")
    $vc = Read-ArmVerdict -Dir $gdir -Tag 'ev-clean'
    Check '(a) the verdict was written' ($null -ne $vc)
    Check '(a) the probe evidence is persisted, not merely printed' `
        ($vc.fields.gpu_pre_run -eq 'clean' -and $vc.fields.gpu_post_run -eq 'clean') `
        ("pre=$($vc.fields.gpu_pre_run) post=$($vc.fields.gpu_post_run)")
    Check '(a) the statuses use the fixed vocabulary' `
        ($vc.fields.gpu_pre_status -eq 'clean' -and $vc.fields.gpu_post_status -eq 'clean')
    Check '(a) the tag is in the verdict, with how it was chosen' `
        ($vc.fields.tag -eq 'ev-clean' -and $vc.fields.tag_source -eq 'explicit') `
        ("tag=$($vc.fields.tag) source=$($vc.fields.tag_source)")
    Check '(a) the probe source is named, so a stubbed probe is not read as evidence' `
        ($vc.fields.gpu_probe_source -eq 'stub') ("source=$($vc.fields.gpu_probe_source)")
    Check '(a) the pre-existing fields keep their values' `
        ($vc.fields.mutex_acquired -eq $true -and $vc.fields.timeout_seconds -eq 30)

    # (b) A consumer written before this change knows only these. Nothing may be
    # renamed, re-typed or moved out from under it by an additive change.
    $legacyKeys = @('schema', 'arm', 'status', 'exit_code', 'started_utc', 'ended_utc',
                    'duration_s', 'mutex', 'command', 'log', 'error', 'fields')
    $gotKeys = @($vc.PSObject.Properties.Name | Where-Object { $_ -ne 'done_present' -and $_ -ne 'done_text' })
    Check '(b) no top-level field was added, removed or renamed' `
        (($gotKeys -join ',') -eq ($legacyKeys -join ',')) ($gotKeys -join ',')
    $legacyView = '{0} {1} {2} gpu_pre={3} gpu_post={4} mutex_held={5} timeout={6}' -f `
        $vc.status, $vc.exit_code, $vc.arm, $vc.fields.gpu_pre_run, $vc.fields.gpu_post_run,
        $vc.fields.mutex_acquired, $vc.fields.timeout_seconds
    Check '(b) the legacy projection reads exactly as it did before' `
        ($legacyView -eq 'PASS 0 ev-clean gpu_pre=clean gpu_post=clean mutex_held=True timeout=30') $legacyView

    $code = Invoke-GpuFieldCase -Tag 'ev-busy' -Probe "type `"$busyOut`""
    Check 'busy refusal is still exit 125' ($code -eq 125) ("exit=$code")
    $vb = Read-ArmVerdict -Dir $gdir -Tag 'ev-busy'
    Check '(c) busy is distinguishable from clean' `
        ($vb.fields.gpu_pre_status -eq 'busy' -and $vb.fields.gpu_pre_run -like '*deepseek_v4.exe*')
    Check '(c) refused_busy still names the process' ($vb.fields.refused_busy -like '*deepseek_v4.exe*')
    Check '(c) a refused run records ABSENCE, not a clean post-probe' `
        ($vb.fields.gpu_post_status -eq 'not-probed' -and $vb.fields.gpu_post_run -eq '__unprobed__') `
        ("post_status=$($vb.fields.gpu_post_status)")

    $code = Invoke-GpuFieldCase -Tag 'ev-noprobe' -Probe "`"$failOut`""
    Check 'a probe that cannot run does not block the arm' ($code -eq 0) ("exit=$code")
    $vn = Read-ArmVerdict -Dir $gdir -Tag 'ev-noprobe'
    Check '(c) a failed probe is unavailable, not clean' `
        ($vn.fields.gpu_pre_status -eq 'unavailable' -and $vn.fields.gpu_pre_run -eq 'probe-unavailable')

    # The measured defect this section exists for: a probe that ANSWERED with
    # something unparseable used to persist as `clean` (empty row list), which is
    # indistinguishable from an idle card in every arm verdict ever written.
    $code = Invoke-GpuFieldCase -Tag 'ev-junk' -Probe "type `"$junkOut`""
    $vj = Read-ArmVerdict -Dir $gdir -Tag 'ev-junk'
    Check '(c) UNPARSEABLE probe output is unavailable, NOT clean' `
        ($vj.fields.gpu_pre_status -eq 'unavailable' -and $vj.fields.gpu_pre_run -ne 'clean') `
        ("pre_status=$($vj.fields.gpu_pre_status) pre_run=$($vj.fields.gpu_pre_run)")
    Check '(c) and the arm still ran UNVERIFIED rather than being blocked' `
        ($code -eq 0 -and $vj.status -eq 'PASS')

    # (c) the never-taken case. A run that ended before any probe must read
    # 'not-probed': absence is a state, not a default of clean.
    $mtxHold = 'Global\AI-GPU-EXCLUSIVE-v1-selftest-hold-' + [guid]::NewGuid().ToString('N').Substring(0, 8)
    $holder = Start-Job -ScriptBlock {
        param($n)
        $m = [System.Threading.Mutex]::new($false, $n)
        [void]$m.WaitOne()
        Start-Sleep -Seconds 90
    } -ArgumentList $mtxHold
    $held = $false
    $pm = [System.Threading.Mutex]::new($false, $mtxHold)
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $deadline) {
        if ($pm.WaitOne(0)) { $pm.ReleaseMutex(); Start-Sleep -Milliseconds 250 } else { $held = $true; break }
    }
    $pm.Dispose()
    Check 'the private holder mutex is held by another process' $held
    if ($held) {
        $code = Invoke-GpuFieldCase -Tag 'ev-nofree' -HoldMutex $mtxHold -TimeoutSeconds 0
        Check 'an unacquirable mutex is still TIMEOUT 124' ($code -eq 124) ("exit=$code")
        $vt = Read-ArmVerdict -Dir $gdir -Tag 'ev-nofree'
        Check '(c) a run that never probed reads not-probed, never clean' `
            ($vt.fields.gpu_pre_status -eq 'not-probed' -and $vt.fields.gpu_post_status -eq 'not-probed') `
            ("pre=$($vt.fields.gpu_pre_status) post=$($vt.fields.gpu_post_status)")
        Check '(c) and it names the mutex it did not get' `
            ($vt.fields.mutex_acquired -eq $false -and $vt.mutex -eq $mtxHold)
    }
    Stop-Job -Job $holder -ErrorAction SilentlyContinue
    Remove-Job -Job $holder -Force -ErrorAction SilentlyContinue

    $code = Invoke-GpuFieldCase -Tag 'ignored' -Probe "type `"$cleanOut`"" -DeriveTag
    $vd = Read-ArmVerdict -Dir $gdir -Tag 'gpu-command'
    Check '(a) a DERIVED tag is recorded as derived, and names itself' `
        ($null -ne $vd -and $vd.fields.tag -eq 'gpu-command' -and $vd.fields.tag_source -eq 'derived') `
        ("tag=$($vd.fields.tag) source=$($vd.fields.tag_source)")

    Write-Host '=== 13. provenance: which tree, and which binary, a verdict describes ==='
    # Every arm's evidence is only as good as its answer to those two questions.
    # Measured 2026-09-22: a control arm passed at 16:28 against the 16:07 build,
    # the engine sources were edited at 16:31 and the tree rebuilt at 16:32, and
    # nothing in the record noticed -- the verdict named an exe hash and no
    # consumer compared it to anything. The fixture half of this (a source edit
    # AFTER a build must read STALE, not CURRENT) lives in the checker's own
    # selftest, tools/check-arm-provenance.ps1 -Selftest, over a throwaway repo;
    # the real tree is never mutated by a test.
    $realBin = Join-Path $root 'build\llama-cuda\bin\Release'
    $realExe = Join-Path $realBin 'llama-layer-streaming.exe'
    $realDll = Join-Path $realBin 'llama.dll'
    $nested  = Join-Path $root 'third_party\llama.cpp'
    if (-not (Test-Path -LiteralPath $realExe)) {
        $script:skip++
        Write-Host '  SKIP  no built arm exe on disk; the binary-identity cases need one.'
    } else {
        # (a) the binary is identified FROM THE COMMAND, so a verdict names the
        # binary it actually ran rather than whatever is in the build tree.
        $arts = @(Get-ProvenanceArtifacts -Command ('"{0}" -m model.gguf --pp-schedule layer-major' -f $realExe))
        $leafs = @($arts | ForEach-Object { Split-Path -Leaf $_ })
        Check '(a) the command resolves to its exe plus the engine modules' `
            ((@($arts) -contains $realExe) -and ($leafs -contains 'llama.dll' -or -not (Test-Path -LiteralPath $realDll))) `
            ($leafs -join ', ')

        # (b) a verdict written through the SHARED writer carries the real tree.
        # The equality below is only meaningful while the tree holds still, and
        # this checkout has a concurrently working actor (measured 2026-09-22: the
        # nested digest moved twice inside an hour), so quiescence is TESTED
        # rather than assumed - a moving tree reports a SKIP with its reason, not
        # a failure of this code.
        $pdir = Join-Path $tmp 'provenance'
        $preDigest = (Get-ArmProvenance -SourceRoot $nested -RepoRoot $root).source.digest
        Write-ArmVerdict -Dir $pdir -Tag 'prov-code' -ExitCode 0 `
            -StartedUtc (Get-Date).ToUniversalTime().ToString('o') `
            -ArtifactPaths @($realExe, $realDll) -SourceRoot $nested -RepoRoot $root | Out-Null
        $pv = Read-ArmVerdict -Dir $pdir -Tag 'prov-code'
        Check '(b) the verdict carries provenance' ($null -ne $pv.fields.provenance)
        Check '(b) it names the engine tree and its HEAD' `
            ($pv.fields.provenance.source_state -eq 'git' -and $pv.fields.provenance.source_scope -eq 'third_party/llama.cpp' -and $pv.fields.provenance.source_head) `
            ("state={0} head={1}" -f $pv.fields.provenance.source_state, $pv.fields.provenance.source_head)
        $fresh = Get-ArmProvenance -SourceRoot $nested -RepoRoot $root -ArtifactPaths @($realExe, $realDll)
        if ($preDigest -eq $fresh.source.digest) {
            Check '(b) the digest is the tree as it stands, not a placeholder' `
                ($pv.fields.provenance.source_digest -eq $fresh.source.digest -and $pv.fields.provenance.source_digest.Length -eq 16) `
                $pv.fields.provenance.source_digest
        } else {
            $script:skip++
            Write-Host ('  SKIP  the nested tree moved during this case ({0} -> {1}): the digest equality is not testable while another actor edits it.' -f $preDigest, $fresh.source.digest)
            Check '(b) the digest is well formed even so' ($pv.fields.provenance.source_digest.Length -eq 16) $pv.fields.provenance.source_digest
        }
        Check '(b) the streaming library hash is recorded beside the exe hash' `
            ($pv.fields.provenance.streaming_lib -eq 'llama.dll' -and $pv.fields.provenance.streaming_lib_sha -eq $fresh.binary.streaming_lib_sha) `
            ("lib={0}" -f $pv.fields.provenance.streaming_lib_sha)

        # (c) the checker certifies it -- through the CLI, as a reader would run it.
        # Same quiescence rule as (b): a rebuild landing mid-case is a real event
        # the mechanism is SUPPOSED to catch, so it is reported as a skip of this
        # host-only case rather than as a defect in it.
        $chk = Join-Path $root 'tools\check-arm-provenance.ps1'
        $null = & powershell -NoProfile -ExecutionPolicy Bypass -File $chk -Dir $pdir -Tag 'prov-code' -Quiet
        $certExit = $LASTEXITCODE
        $postBin = Get-BinaryIdentity -Paths @($realExe, $realDll) -RepoRoot $root
        $sameBin = (($postBin.modules | ConvertTo-Json -Compress) -eq ($fresh.binary.modules | ConvertTo-Json -Compress))
        if ($sameBin) {
            Check '(c) an arm of the current tree certifies: checker exit 0' ($certExit -eq 0) ("exit=$certExit")
        } else {
            $script:skip++
            Write-Host '  SKIP  the binary was rebuilt during this case; the certification result is not attributable to this code.'
            Check '(c) the checker still rendered a decision' ($certExit -eq 0 -or $certExit -eq 1 -or $certExit -eq 2) ("exit=$certExit")
        }

        # (d) a verdict with no provenance is NOT a pass, and says so out loud
        $legacy = Join-Path $pdir 'legacy.verdict.json'
        [System.IO.File]::WriteAllText($legacy, '{"schema":"arm-verdict/1","arm":"legacy","status":"PASS","exit_code":0,"fields":{"mutex_acquired":true}}')
        $out = (& powershell -NoProfile -ExecutionPolicy Bypass -File $chk -Verdict $legacy 2>&1) -join "`n"
        Check '(d) a pre-provenance verdict exits 2, not 0' ($LASTEXITCODE -eq 2) ("exit=$LASTEXITCODE")
        Check '(d) and it is labelled NOT A PASS' ($out -like '*NO-PROVENANCE*NOT A PASS*')
    }

    # (e) through the wrapper: the whole run is bracketed, and honest absence is
    # still recorded when the command names no binary.
    # A command that names NO binary is exactly what this case is about: the capture must
    # say `not-supplied` rather than guess, and the WRAPPER gate then refuses the run
    # (section 15). The helper carries the explicit waiver so the case can still observe
    # the wrapper's own contract (probe fields, exit code) rather than the refusal.
    $code = Invoke-GpuFieldCase -Tag 'prov-wrap' -Probe "type `"$cleanOut`"" -Command 'exit 0'
    $vw = Read-ArmVerdict -Dir $gdir -Tag 'prov-wrap'
    Check '(e) a wrapped run records provenance taken at launch' `
        ($null -ne $vw.fields.provenance -and $vw.fields.provenance.origin -eq 'launch') `
        ("origin={0}" -f $vw.fields.provenance.origin)
    Check '(e) the wrapper captures the run twice' ($null -ne $vw.fields.provenance_post -and $vw.fields.provenance_post.origin -eq 'post-run')
    Check '(e) the delta between the two captures is computed, and agrees with them' `
        ($vw.fields.provenance_delta.source_changed_during_run -eq ($vw.fields.provenance.source_digest -ne $vw.fields.provenance_post.source_digest)) `
        ("delta={0}" -f $vw.fields.provenance_delta.source_changed_during_run)
    Check '(e) a command that names no binary records THAT, rather than a guess' `
        ($vw.fields.provenance.binary_state -eq 'not-supplied') ("binary_state={0}" -f $vw.fields.provenance.binary_state)
    Check '(e) and a source file edited mid-run would have been visible' `
        ($vw.fields.provenance.source_newest_change_path -ne '' -or $vw.fields.provenance.source_changed_tracked -eq 0) `
        ("newest={0}" -f $vw.fields.provenance.source_newest_change_path)

    # (f) THE DRIVER PATH, which is the whole reason the capture is not in the
    # wrapper: a driver writes the arm's verdict itself, for its own tag, with no
    # artifact arguments at all. Its binary comes from the run's own start marker.
    if (Test-Path -LiteralPath $realExe) {
        $ddir = Join-Path $tmp 'driver'
        Write-ArmStart -Dir $ddir -Tag 'drv' -Command (('"{0}" -m model.gguf --pp-schedule layer-major' -f $realExe)) | Out-Null
        Write-ArmVerdict -Dir $ddir -Tag 'drv' -ExitCode 0 `
            -StartedUtc (Get-Date).ToUniversalTime().ToString('o') -Fields @{ pass1_wall_s = 3.7 } | Out-Null
        $dv = Read-ArmVerdict -Dir $ddir -Tag 'drv'
        Check '(f) a driver verdict with no artifact args still names its binary' `
            ($dv.fields.provenance.binary_exe -eq 'llama-layer-streaming.exe' -and $dv.fields.provenance.binary_state -eq 'hashed') `
            ("exe={0} lib={1}" -f $dv.fields.provenance.binary_exe_sha, $dv.fields.provenance.streaming_lib_sha)
        Check '(f) it is the ENGINE library that was hashed, not just the launcher' `
            ($dv.fields.provenance.streaming_lib -eq 'llama.dll' -and $dv.fields.provenance.streaming_lib_sha.Length -eq 16) `
            $dv.fields.provenance.streaming_lib_sha
        Check '(f) it is marked as the writer''s own capture, not the wrapper''s' `
            ($dv.fields.provenance.origin -eq 'verdict-write') ("origin={0}" -f $dv.fields.provenance.origin)
        Check '(f) the driver''s own fields survive beside it' ($dv.fields.pass1_wall_s -eq 3.7)
        $null = & powershell -NoProfile -ExecutionPolicy Bypass -File $chk -Dir $ddir -Tag 'drv' -Quiet
        $drvExit = $LASTEXITCODE
        if ($sameBin -and $preDigest -eq (Get-ArmProvenance -SourceRoot $nested -RepoRoot $root).source.digest) {
            Check '(f) and the checker certifies the driver-written arm: exit 0' ($drvExit -eq 0) ("exit=$drvExit")
        } else {
            $script:skip++
            Write-Host '  SKIP  the tree moved during this case; certification is not attributable to this code.'
            Check '(f) the checker still rendered a decision' ($drvExit -eq 0 -or $drvExit -eq 1 -or $drvExit -eq 2) ("exit=$drvExit")
        }
    }

    Write-Host '=== 14. the provenance GATE: an arm that cannot attest its revision is refused ==='
    # What turns a capture into a requirement. The gate is called by the GPLR-53
    # drivers at their exit, by the session launcher per step, and by the smoke
    # before it writes its chain verdict; the refusal path is proven here, host-only.
    $gdir2 = Join-Path $tmp 'gate'
    $realBin2 = Join-Path $root 'build\llama-cuda\bin\Release'
    $gExe = Join-Path $realBin2 'llama-layer-streaming.exe'
    $gDll = Join-Path $realBin2 'llama.dll'
    $gNested = Join-Path $root 'third_party\llama.cpp'

    # A verdict that CAN attest: complete capture, real tree, real binary.
    Write-ArmVerdict -Dir $gdir2 -Tag 'gate-good' -ExitCode 0 `
        -StartedUtc (Get-Date).ToUniversalTime().ToString('o') `
        -ArtifactPaths @($gExe, $gDll) -SourceRoot $gNested -RepoRoot $root | Out-Null
    $gate = Get-ArmProvenanceGate -Dir $gdir2 -Tags @('gate-good') -RepoRoot $root
    Check '(a) a complete verdict passes the gate' ($gate.checked -eq 1 -and $gate.failures.Count -eq 0) `
        ("checked={0} failures={1}" -f $gate.checked, $gate.failures.Count)
    Check '(a) the gate names the digest and library it accepted' ($gate.arms[0].digest.Length -eq 16 -and $gate.arms[0].lib_sha.Length -eq 16) `
        ("digest={0} lib={1}" -f $gate.arms[0].digest, $gate.arms[0].lib_sha)
    Check '(a) and Assert-ArmProvenance returns true, the pass the drivers rely on' `
        (Assert-ArmProvenance -Dir $gdir2 -Tags @('gate-good') -RepoRoot $root -Quiet)

    # THE REFUSAL PATH: the same arm with its capture stripped. This is the state
    # every one of the 101 saved verdicts is in, and it must never pass.
    $gj = Get-Content -LiteralPath (Join-Path $gdir2 'gate-good.verdict.json') -Raw | ConvertFrom-Json
    $gj.fields.PSObject.Properties.Remove('provenance')
    $gj | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $gdir2 'gate-stripped.verdict.json') -Encoding UTF8
    # The sentinel too: without it the gate (correctly) refuses for INCOMPLETENESS
    # first, which would hide the capture-specific reason this case is about.
    Set-Content -LiteralPath (Join-Path $gdir2 'gate-stripped.done') -Value 'PASS 0 (case b) gate-stripped' -Encoding Ascii
    $g1 = Get-ArmProvenanceGate -Dir $gdir2 -Tags @('gate-stripped') -RepoRoot $root
    Check '(b) a verdict with no capture is REFUSED' ($g1.failures.Count -eq 1 -and ($g1.arms[0].reasons -join ' ') -like '*no provenance capture*') `
        ($g1.arms[0].reasons -join '; ')
    Check '(b) and Assert-ArmProvenance returns false' (-not (Assert-ArmProvenance -Dir $gdir2 -Tags @('gate-stripped') -RepoRoot $root -Quiet))

    # A capture that exists but names no binary: an arm that ran SOMETHING.
    Write-ArmVerdict -Dir $gdir2 -Tag 'gate-noexe' -ExitCode 0 -Command 'exit 0' | Out-Null
    $g2 = Get-ArmProvenanceGate -Dir $gdir2 -Tags @('gate-noexe') -RepoRoot $root
    Check '(c) a capture with no binary identity is REFUSED' ($g2.failures.Count -eq 1 -and ($g2.arms[0].reasons -join ' ') -like '*no binary identity*') `
        ($g2.arms[0].reasons -join '; ')

    # A capture that says the binary predates its sources: the run itself recorded
    # that its evidence cannot describe the code.
    $gj2 = Get-Content -LiteralPath (Join-Path $gdir2 'gate-good.verdict.json') -Raw | ConvertFrom-Json
    $gj2.fields.provenance.sources_newer_than_binary = $true
    $gj2.fields.provenance.sources_newer_detail = 'src/x.cpp edited after the build'
    $gj2 | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $gdir2 'gate-lag.verdict.json') -Encoding UTF8
    Set-Content -LiteralPath (Join-Path $gdir2 'gate-lag.done') -Value 'PASS 0 (case d) gate-lag' -Encoding Ascii
    $g3 = Get-ArmProvenanceGate -Dir $gdir2 -Tags @('gate-lag') -RepoRoot $root
    Check '(d) an arm whose binary predates its sources is REFUSED' ($g3.failures.Count -eq 1 -and ($g3.arms[0].reasons -join ' ') -like '*PREDATES its sources*') `
        ($g3.arms[0].reasons -join '; ')

    # A tag with no verdict at all: absence is not a pass.
    $g4 = Get-ArmProvenanceGate -Dir $gdir2 -Tags @('never-ran') -RepoRoot $root
    Check '(e) a tag with no verdict is REFUSED, not skipped' ($g4.failures.Count -eq 1 -and ($g4.arms[0].reasons -join ' ') -like '*no verdict file*') `
        ($g4.arms[0].reasons -join '; ')
    # ... and an empty SELECTION refuses too (the silent-pass hole in the scan path).
    $future = (Get-Date).AddHours(1).ToUniversalTime().ToString('o')
    Check '(e) an empty selection refuses rather than passing vacuously' `
        (-not (Assert-ArmProvenance -Dir $gdir2 -SinceUtc $future -RepoRoot $root -Quiet))

    # THE ESCAPE, explicit and recorded.
    $wlog = Join-Path $gdir2 'provenance-waivers.log'
    if (Test-Path -LiteralPath $wlog) { Remove-Item -LiteralPath $wlog -Force }
    $waived = Assert-ArmProvenance -Dir $gdir2 -Tags @('gate-stripped', 'never-ran') -RepoRoot $root -AllowUnprovenanced -WaiverNote 'unit-case' -Quiet
    Check '(f) the escape lets the run continue' ($waived -eq $true)
    Check '(f) and it is RECORDED, not quiet' (Test-Path -LiteralPath $wlog)
    if (Test-Path -LiteralPath $wlog) {
        $wl = Get-Content -LiteralPath $wlog -Raw
        Check '(f) the waiver line names every uncertifiable arm' ($wl -like '*gate-stripped*' -and $wl -like '*never-ran*' -and $wl -like '*unit-case*') `
            ($wl.Trim())
    }

    # The wiring itself, guarded so it cannot be deleted silently. Static on
    # purpose (a driver cannot be run host-only), and labelled as static: it
    # asserts the CALL SITES exist, not that a GPU run reaches them.
    $wired = @(
        @{ f = 'tools\run-gplr53-a3-arm.ps1'; want = @('Assert-ArmProvenance', 'AllowUnprovenanced', '$sessionExit = 10') },
        @{ f = 'tools\run-gplr53-a4-arm.ps1'; want = @('Assert-ArmProvenance', 'AllowUnprovenanced', '$sessionExit = 10') },
        @{ f = 'tools\run-gplr53-session.ps1'; want = @('Assert-ArmProvenance', 'AllowUnprovenanced', 'exit 10') },
        @{ f = 'tools\smoke-gpu-lock-ctxshell.ps1'; want = @('Assert-ArmProvenance', 'AllowUnprovenanced', 'arm_provenance_complete') }
    )
    foreach ($w in $wired) {
        $src = Get-Content -LiteralPath (Join-Path $root $w.f) -Raw
        $missingBits = @($w.want | Where-Object { $src -notlike ('*' + $_ + '*') })
        Check (('(g) [static] {0} calls the gate and propagates a refusal' -f (Split-Path -Leaf $w.f))) `
            ($missingBits.Count -eq 0) (($missingBits -join ', '))
    }

    Write-Host '=== 15. the gate at the CHOKE POINT: the WRAPPER itself refuses an unattested run ==='
    # The audit's finding (2026-09-22): capture was systematic (the writer always records
    # it) while enforcement lived only in the CALLERS, so a driver that forgot its own
    # gate shipped uncertifiable evidence SILENTLY. The wrapper is the shared path, so the
    # assertion lives there now too - same taxonomy (exit 10), same explicit waiver.
    #
    # Hermetic and cheap: a throwaway copy of the three tools at a temp root with NO
    # engine tree, so the capture cannot name a revision and the gate must refuse. No real
    # file is touched, and the private mutex name keeps this out of the shared GPU queue.
    $croot = Join-Path $tmp 'choke'
    New-Item -ItemType Directory -Force -Path (Join-Path $croot 'tools') | Out-Null
    foreach ($f in @('with-gpu-lock.ps1', 'arm-verdict.ps1', 'source-revision.ps1')) {
        Copy-Item -LiteralPath (Join-Path $root ('tools\' + $f)) -Destination (Join-Path $croot ('tools\' + $f))
    }
    $cvDir = Join-Path $croot 'verdicts'
    New-Item -ItemType Directory -Force -Path $cvDir | Out-Null
    $wlock = Join-Path $croot 'tools\with-gpu-lock.ps1'

    function Invoke-ChokeCase {
        param([string] $Tag, [string] $MutexTag, [switch] $Waive)
        # An ARGUMENT ARRAY, not a hashtable splat: splatting a hashtable onto a NATIVE
        # command (powershell.exe) expands every entry as `-Key value`, so a switch entry
        # becomes the literal positional argument `True` and the child fails to bind it
        # (measured 2026-09-22, this case). A bare `-AllowUnprovenanced` token is correct.
        $argList = @('-TimeoutSeconds', '30', '-Command', "`"$env:ComSpec`" /d /c exit 0", '-VerdictDir', $cvDir,
                     '-VerdictTag', $Tag, '-BusyGraceSeconds', '0',
                     '-MutexName', ('Global\AI-GPU-EXCLUSIVE-v1-selftest-choke-' + $MutexTag))
        if ($Waive) { $argList += '-AllowUnprovenanced' }
        $env:GPU_BUSY_PROBE_CMD = "type `"$cleanOut`""
        $null = & powershell -NoProfile -ExecutionPolicy Bypass -File $wlock @argList 2>&1
        $code = $LASTEXITCODE
        Remove-Item Env:GPU_BUSY_PROBE_CMD -ErrorAction SilentlyContinue
        return $code
    }

    $chokeExit = Invoke-ChokeCase -Tag 'choke-refused' -MutexTag 'refused'
    Check '(a) the WRAPPER refuses (exit 10) a run it cannot attribute - not only its caller' `
        ($chokeExit -eq 10) ("exit=$chokeExit")
    $cv = Read-ArmVerdict -Dir $cvDir -Tag 'choke-refused'
    Check '(b) the verdict AGREES with the refusal (no fabricated PASS beside it)' `
        ($null -ne $cv -and "$($cv.status)" -ne 'PASS') ("status=$(if ($cv) { $cv.status } else { '<none>' })")
    Check '(c) the command''s own exit code is preserved, not lost to the refusal' `
        ($null -ne $cv -and "$($cv.fields.command_exit)" -eq '0') ("command_exit=$(if ($cv) { $cv.fields.command_exit } else { '<none>' })")
    Check '(d) the refusal is named in the verdict' ($null -ne $cv -and $cv.fields.refused_unprovenanced -eq $true)
    Check '(e) the refusal NAMES what could not be attested' (`
        (Get-Content -LiteralPath (Join-Path $cvDir 'choke-refused.verdict.json') -Raw) -match 'source_state|no provenance capture')

    $waiveExit = Invoke-ChokeCase -Tag 'choke-waived' -MutexTag 'waived' -Waive
    Check '(f) -AllowUnprovenanced still passes AT THE WRAPPER (exit 0)' ($waiveExit -eq 0) ("exit=$waiveExit")
    $waiveLog = Join-Path $cvDir 'provenance-waivers.log'
    Check '(g) and the waiver is RECORDED, not silent' (Test-Path -LiteralPath $waiveLog)
    $cvw = Read-ArmVerdict -Dir $cvDir -Tag 'choke-waived'
    Check '(h) a waived run reads as PASS, and the waiver file names the arm' `
        ($null -ne $cvw -and "$($cvw.status)" -eq 'PASS' -and
         ((Get-Content -LiteralPath $waiveLog -Raw) -like '*choke-waived*'))
    Write-Host '=== 17. the WRITER refuses to leave an unattested PASS (the last structural hole) ==='
    # The inventory left 20 historical runners that never call the gate. Because the writer
    # captures automatically, a verdict from one of them LOOKED certifiable to a later
    # reader. The rule now lives in the writer, at the one place every verdict is created,
    # and it is about ATTESTABILITY: a verdict that cannot name its tree and binary is
    # recorded UNCERTIFIED under the same exit contract the live gates use (10), with the
    # evidence kept and the reasons named - never dropped, never a pass.
    $adir = Join-Path $tmp 'attest'
    New-Item -ItemType Directory -Force -Path $adir | Out-Null

    # (a) an unattested run that WOULD end PASS
    $uv = Write-ArmVerdict -Dir $adir -Tag 'unattested' -ExitCode 0 -Command 'exit 0' | Out-Null
    $uv = Read-ArmVerdict -Dir $adir -Tag 'unattested'
    Check '(a) an unattested PASS is recorded UNCERTIFIED, not PASS' `
        ($null -ne $uv -and "$($uv.status)" -eq 'UNCERTIFIED') ("status=$(if ($uv) { $uv.status } else { '<none>' })")
    Check '(b) under the SAME exit contract the live gates use (10)' ($null -ne $uv -and $uv.exit_code -eq 10) ("exit=$(if ($uv) { $uv.exit_code } else { '' })")
    Check '(c) the status it WOULD have had, and the command''s own code, are preserved' `
        ($null -ne $uv -and "$($uv.fields.would_be_status)" -eq 'PASS' -and "$($uv.fields.command_exit)" -eq '0')
    Check '(d) the reasons are NAMED in the verdict' `
        ($null -ne $uv -and @($uv.fields.attestation_reasons).Count -gt 0) ("$(if ($uv) { @($uv.fields.attestation_reasons) -join '; ' } else { '' })")
    Check '(e) the capture failure is what is named - the EVIDENCE is kept, not dropped' `
        ($null -ne $uv -and $null -ne $uv.fields.provenance -and
         ((@($uv.fields.attestation_reasons) -join ' ') -match 'digested|binary identity'))
    Check '(f) the sentinel agrees with the verdict (a reader of .done sees the same thing)' `
        ((Get-Content -LiteralPath (Join-Path $adir 'unattested.done') -Raw) -match 'UNCERTIFIED 10') `
        ((Get-Content -LiteralPath (Join-Path $adir 'unattested.done') -Raw).Trim())

    # (g) a verdict that CAN attest is not touched - the rule must not break good runs
    # The artifact set must be the REAL one (the exe AND the dlls beside it), and this case
    # proved why: passing the exe alone made the capture read "the binary PREDATES its
    # sources", because the engine code lands in llama.dll while the exe relinks on its own
    # schedule - the repo's own documented gotcha, caught here by the rule itself.
    $attCmd = ('"{0}" -m model.gguf' -f $realExe)
    $gv = Write-ArmVerdict -Dir $adir -Tag 'attested' -ExitCode 0 `
        -Command $attCmd -ArtifactPaths (Get-ProvenanceArtifacts -Command $attCmd) `
        -SourceRoot (Join-Path $root 'third_party\llama.cpp') -RepoRoot $root
    Check '(g) an attestable run is still written PASS (no collateral damage)' `
        ($null -ne $gv -and "$($gv.status)" -eq 'PASS') ("status=$(if ($gv) { $gv.status } else { '<none>' })")
    Check '(h) and it carries no uncertified marker' (-not $gv.fields.uncertified)

    # (i) the explicit opt-out, for a verdict that is not an arm result at all
    $av = Write-ArmVerdict -AllowUnattested -Dir $adir -Tag 'optedout' -ExitCode 0
    Check '(i) -AllowUnattested is the escape and it is explicit' `
        ($null -ne $av -and "$($av.status)" -eq 'PASS') ("status=$(if ($av) { $av.status } else { '<none>' })")

    Write-Host '=== 16. every verdict WRITER is gated, or is a named historical exception ==='
    # The audit counted ten-plus older runners that hash inline and take the mutex
    # themselves, i.e. paths that can record evidence WITHOUT a gate. Rather than rewrite
    # historical runners wholesale, the rule is made enforceable: any file that writes a
    # verdict of its own must also call the gate. The files that predate the mechanism are
    # listed BY NAME below, dated, so the gap is explicit and any NEW ungated writer FAILS
    # this case instead of joining a silent set.
    $historicalUngated = @(
        'run-gplr36-remeasure-200k.ps1', 'run-gplr37-arm.ps1', 'run-gplr37-set.ps1',
        'run-gplr39-arm.ps1', 'run-gplr44-ab-50k.ps1', 'run-gplr45-arm.ps1',
        'run-gplr46-rebuild-ab.ps1', 'run-gplr47-arm.ps1', 'run-gplr48-arm.ps1',
        'run-gplr49-arm.ps1', 'run-gplr49b-echo-attrib.ps1', 'run-gplr50-200k-pair.ps1',
        'run-gplr50-arm.ps1', 'run-gplr51-a2-pinned.ps1', 'run-gplr51-s2-arm.ps1',
        'run-gplr51-tail-arm.ps1', 'run-gplr52-a2-arm.ps1', 'run-gplr52-a3-arm.ps1',
        # Found by THIS census, not by the first hand-scan: both invoke the wrapper AND
        # write a verdict of their own, so the wrapper's gated verdict is not the one
        # they leave behind. Completed tasks (T-GPLR-19, T-GPLR-44).
        'run-gplr19-rowcache-ab.ps1', 'run-gplr44-dump-50k.ps1'
    )
    # Tooling, not arms: the writer/definition site, the checker (its verdict records a
    # CHECK, not an arm) and this harness. The wrapper needs no exemption - it calls the gate.
    $toolingExempt = @('arm-verdict.ps1', 'check-arm-provenance.ps1', 'test-arm-verdict.ps1')
    function Get-UngatedVerdictWriter {
        param([string] $Dir)
        $out = @()
        foreach ($f in @(Get-ChildItem -LiteralPath $Dir -Filter '*.ps1' -File -ErrorAction SilentlyContinue | Sort-Object Name)) {
            $src = Get-Content -LiteralPath $f.FullName -Raw
            if ($src -notmatch 'Write-ArmVerdict') { continue }
            if ($src -match 'Assert-ArmProvenance') { continue }
            if ($toolingExempt -contains $f.Name) { continue }
            $out += $f.Name
        }
        return $out
    }

    $ungatedNow = @(Get-UngatedVerdictWriter -Dir (Join-Path $root 'tools'))
    $unexpected = @($ungatedNow | Where-Object { $historicalUngated -notcontains $_ })
    Check '(a) no ungated verdict writer exists beyond the named historical set' `
        ($unexpected.Count -eq 0) ("unexpected: " + ($unexpected -join ', '))
    $stale = @($historicalUngated | Where-Object { $ungatedNow -notcontains $_ })
    Check '(b) the named set is not stale (a gated writer must be removed from it)' `
        ($stale.Count -eq 0) ("now gated, remove from the list: " + ($stale -join ', '))
    Write-Host ("  NOTE   {0} historical ungated writer(s), all from completed tasks (T-GPLR-36..52), " -f $ungatedNow.Count)
    Write-Host '         none referenced by the active plan or this suite: named in the T-GPLR-57 block, not silently tolerated.'

    # (c) the rule must actually FAIL on a new ungated writer - proven hermetically, so a
    # green case can never mean "the census matched nothing".
    $fakeDir = Join-Path $tmp 'census'
    New-Item -ItemType Directory -Force -Path $fakeDir | Out-Null
    Set-Content -LiteralPath (Join-Path $fakeDir 'run-gplr99-new.ps1') `
        -Value 'Write-ArmVerdict -Dir x -Tag y -ExitCode 0 -Command z' -Encoding Ascii
    Set-Content -LiteralPath (Join-Path $fakeDir 'run-gplr98-gated.ps1') `
        -Value 'Write-ArmVerdict -Dir x -Tag y -ExitCode 0; Assert-ArmProvenance -Dir x -Tags @("y")' -Encoding Ascii
    Set-Content -LiteralPath (Join-Path $fakeDir 'run-gplr97-noevid.ps1') `
        -Value 'Write-Host "no verdict here"' -Encoding Ascii
    $fakeUngated = @(Get-UngatedVerdictWriter -Dir $fakeDir)
    Check '(c) a NEW ungated writer is caught by the census (fails the case)' `
        ($fakeUngated.Count -eq 1 -and $fakeUngated[0] -eq 'run-gplr99-new.ps1') ($fakeUngated -join ', ')
    Check '(d) a gated writer and a no-evidence runner are both correctly ignored' `
        ($fakeUngated -notcontains 'run-gplr98-gated.ps1' -and $fakeUngated -notcontains 'run-gplr97-noevid.ps1')
} finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction Continue }
}

Write-Host ''
if ($fail -eq 0) {
    if ($script:skip -gt 0) {
        Write-Host ("ALL ARM-VERDICT CASES PASS ({0} section(s) SKIPPED - see the SKIP lines above)" -f $script:skip)
    } else {
        Write-Host 'ALL ARM-VERDICT CASES PASS'
    }
    exit 0
}
Write-Host ("ARM-VERDICT CASES FAILED: {0}" -f $fail)
exit 1
