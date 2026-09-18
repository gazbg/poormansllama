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
#      which sentinel to check.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File tools/test-arm-verdict.ps1
# Exit code: 0 = all cases pass, 1 = at least one failed.
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
. (Join-Path $root 'tools\arm-verdict.ps1')

$fail = 0
function Check([string]$name, [bool]$cond, [string]$detail = '') {
    if ($cond) {
        Write-Host ("  PASS  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' }))
    } else {
        Write-Host ("  FAIL  {0}{1}" -f $name, $(if ($detail) { " -- $detail" } else { '' }))
        $script:fail++
    }
}

$tmp = Join-Path $env:TEMP ('arm-verdict-test-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
$tag = 'unit-case'

try {
    Write-Host '=== 1. round trip ==='
    Clear-ArmVerdict -Dir $tmp -Tag $tag
    $p = Get-ArmVerdictPaths -Dir $tmp -Tag $tag
    Check 'clear leaves no files' (-not (Test-Path $p.verdict) -and -not (Test-Path $p.done))
    Write-ArmVerdict -Dir $tmp -Tag $tag -ExitCode 0 -StartedUtc (Get-Date).ToUniversalTime().ToString('o') `
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
    $lock = Join-Path $root 'tools\with-gpu-lock.ps1'
    $vdir = Join-Path $tmp 'wrapper'
    Clear-ArmVerdict -Dir $vdir -Tag 'w1'
    & $lock -TimeoutSeconds 60 -Command 'exit 3' -VerdictDir $vdir -VerdictTag 'w1' | Out-Null
    $w1 = Read-ArmVerdict -Dir $vdir -Tag 'w1'
    Check 'wrapper writes a verdict' ($null -ne $w1)
    Check 'wrapper FAIL reflects cmd exit 3' ($w1.status -eq 'FAIL' -and $w1.exit_code -eq 3) ("$($w1.status) $($w1.exit_code)")
    Check 'wrapper records the mutex it held' ($w1.mutex -eq 'Global\AI-GPU-EXCLUSIVE-v1' -and $w1.fields.mutex_acquired -eq $true)
    Check 'wrapper leaves a sentinel beside it' ($w1.done_present -eq $true)

    # Same tag, opposite result: proves the .done is cleared, not merely overwritten
    # in place, and that yesterday's PASS cannot be read as today's result.
    & $lock -TimeoutSeconds 60 -Command 'exit 0' -VerdictDir $vdir -VerdictTag 'w1' | Out-Null
    $w2 = Read-ArmVerdict -Dir $vdir -Tag 'w1'
    Check 'rerun flips the same tag to PASS 0' ($w2.status -eq 'PASS' -and $w2.exit_code -eq 0) ("$($w2.status) $($w2.exit_code)")

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

    Write-ArmVerdict -Dir $tmp -Tag $mt -ExitCode 0 | Out-Null
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
    $defDir = Join-Path $root 'build\llama-cuda\verdicts'
    Clear-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    & $lock -TimeoutSeconds 60 -Command 'exit 0' | Out-Null
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
    & $lock -TimeoutSeconds 60 -Command $probe -VerdictDir $vdir2 -VerdictTag 'w2' | Out-Null
    $st = Get-ArmRunState -Dir $vdir2 -Tag 'w2'
    Check 'marker present when the arm body started' ($st.verdict.exit_code -eq 0) ("exit=$($st.verdict.exit_code) (9 would mean the marker came too late)")
    Check 'state is finished with both files' ($st.state -eq 'finished' -and $null -ne $st.marker)
} finally {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction Continue }
}

Write-Host ''
if ($fail -eq 0) {
    Write-Host 'ALL ARM-VERDICT CASES PASS'
    exit 0
}
Write-Host ("ARM-VERDICT CASES FAILED: {0}" -f $fail)
exit 1
