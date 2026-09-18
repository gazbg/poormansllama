# Shared verdict + sentinel for GPU arm runs. DOT-SOURCE this file; it defines
# functions only and has no side effects of its own.
#
# CONVENTION (2026-09-18): every GPU arm run leaves exactly two files in a
# predictable directory:
#   <dir>/<tag>.verdict.json  machine-readable: status, exit code, timings, fields
#   <dir>/<tag>.done          SENTINEL, written LAST; its presence means the
#                             verdict beside it is COMPLETE
# So a single check for `<tag>.done` replaces polling a log, and a run that dies
# early cannot masquerade as "still running": the wrappers write a verdict from
# their termination handler as well as on the normal path.
#
# Two rules make the sentinel trustworthy rather than decorative:
#   1. ORDER: verdict first, .done second. A .done therefore implies a complete
#      verdict, never a half-written one.
#   2. INVALIDATE FIRST: Clear-ArmVerdict deletes the .done BEFORE the run, so a
#      stale sentinel from a previous run can never be read as this run's result.
#      This is the failure mode that makes sentinel files untrustworthy in
#      practice -- a PASS from yesterday read as a PASS from today.
#
# Why not just read the log: the log is text, unbounded, and says nothing about
# whether the process finished. The verdict is small, parseable, and its existence
# is the completion signal.

function Get-ArmVerdictPaths {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    return [pscustomobject]@{
        dir     = $Dir
        verdict = (Join-Path $Dir ($Tag + '.verdict.json'))
        done    = (Join-Path $Dir ($Tag + '.done'))
    }
}

function Clear-ArmVerdict {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    foreach ($f in @($p.verdict, $p.done)) {
        if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force -ErrorAction Continue }
    }
}

function Write-ArmVerdict {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag,
        [Parameter(Mandatory = $true)][int]    $ExitCode,
        [string]   $Status = '',
        [hashtable]$Fields = @{},
        [string]   $StartedUtc = '',
        [string]   $LogPath = '',
        [string]   $Command = '',
        [string]   $ErrorText = '',
        [string]   $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1'
    )
    # Status is DERIVED from the exit code when not given explicitly, so a caller
    # cannot accidentally report PASS for a failing run.
    if ([string]::IsNullOrWhiteSpace($Status)) {
        $Status = if ($ExitCode -eq 0) { 'PASS' } elseif ($ExitCode -eq 124) { 'TIMEOUT' } else { 'FAIL' }
    }
    if (-not (Test-Path -LiteralPath $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }

    $ended = (Get-Date).ToUniversalTime()
    $dur   = $null
    if ($StartedUtc) {
        try { $dur = [math]::Round(($ended - [datetime]::Parse($StartedUtc).ToUniversalTime()).TotalSeconds, 1) } catch { $dur = $null }
    }

    $v = [ordered]@{
        schema      = 'arm-verdict/1'
        arm         = $Tag
        status      = $Status
        exit_code   = $ExitCode
        started_utc = $StartedUtc
        ended_utc   = $ended.ToString('o')
        duration_s  = $dur
        mutex       = $MutexName
        command     = $Command
        log         = $LogPath
        error       = $ErrorText
        fields      = $Fields
    }

    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    # Verdict first, sentinel second: .done means "the verdict beside it is complete".
    [System.IO.File]::WriteAllText($p.verdict, ($v | ConvertTo-Json -Depth 6),
        (New-Object System.Text.UTF8Encoding($false)))
    $doneLine = ('{0} {1} {2} {3}' -f $Status, $ExitCode, $ended.ToString('o'), $Tag)
    [System.IO.File]::WriteAllText($p.done, $doneLine + [Environment]::NewLine,
        (New-Object System.Text.UTF8Encoding($false)))
    return [pscustomobject]$v
}

# Returns the verdict object, or $null when no verdict file exists. Callers that
# need "is this run finished" should test the .done sentinel, not the JSON.
function Read-ArmVerdict {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    if (-not (Test-Path -LiteralPath $p.verdict)) { return $null }
    $o = Get-Content -LiteralPath $p.verdict -Raw | ConvertFrom-Json
    Add-Member -InputObject $o -NotePropertyName 'done_present' -NotePropertyValue (Test-Path -LiteralPath $p.done) -Force
    if (Test-Path -LiteralPath $p.done) {
        Add-Member -InputObject $o -NotePropertyName 'done_text' -NotePropertyValue ((Get-Content -LiteralPath $p.done -Raw).Trim()) -Force
    }
    return $o
}

# Blocks until the sentinel appears or the timeout expires. Returns the verdict
# or $null on timeout -- a bounded WAIT, which is what a caller wants instead of
# an unconditional sleep: it returns as soon as the observable condition flips.
function Wait-ArmVerdict {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag,
        [int] $TimeoutSeconds = 600,
        [int] $PollSeconds = 5,
        [switch] $Quiet
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $p.done) { return (Read-ArmVerdict -Dir $Dir -Tag $Tag) }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $Quiet) { Write-Warning ("Wait-ArmVerdict timed out after {0}s waiting for {1}" -f $TimeoutSeconds, $p.done) }
    return $null
}

# Derives a stable tag from the command a wrapper is about to run, so the sentinel
# path is predictable without every caller having to name it: the inner script's
# leaf name (run-gplr36-hostmem.ps1 -> run-gplr36-hostmem), else the exe's
# (llama-layer-streaming), else 'gpu-command'.
function Get-ArmTagFromCommand {
    param([string] $Command)
    if ([string]::IsNullOrWhiteSpace($Command)) { return 'gpu-command' }
    foreach ($ext in @('ps1', 'exe')) {
        $m = [regex]::Matches($Command, ('([^\\/\s"'']+)\.' + $ext + '\b'), 'IgnoreCase')
        if ($m.Count -gt 0) {
            $leaf = $m[$m.Count - 1].Groups[1].Value
            $safe = ($leaf -replace '[^A-Za-z0-9._-]', '_')
            if ($safe) { return $safe }
        }
    }
    return 'gpu-command'
}
