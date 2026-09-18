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

    Write-Host '=== 7. defaults: derived tag, default directory ==='
    $defDir = Join-Path $root 'build\llama-cuda\verdicts'
    Clear-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    & $lock -TimeoutSeconds 60 -Command 'exit 0' | Out-Null
    $d = Read-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    Check 'default dir + derived tag produce a verdict' ($null -ne $d -and $d.status -eq 'PASS')
    Clear-ArmVerdict -Dir $defDir -Tag 'gpu-command'
    Check 'cleanup removed the default-location artifacts' (-not (Test-Path (Join-Path $defDir 'gpu-command.done')))
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
