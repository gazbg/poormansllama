# Shared verdict + sentinel + start marker for GPU arm runs. DOT-SOURCE this
# file; it defines functions only and has no side effects of its own.
#
# CONVENTION (2026-09-18): every GPU arm run leaves up to three files in a
# predictable directory:
#   <dir>/<tag>.started       START MARKER, written at launch, before the heavy work
#   <dir>/<tag>.verdict.json  machine-readable: status, exit code, timings, fields
#   <dir>/<tag>.done          SENTINEL, written LAST; its presence means the
#                             verdict beside it is COMPLETE
#
# Which file exists IS the run's state, so no polling and no log tailing:
#   neither                 -> never started (the launch never got this far)
#   .started only           -> started, not finished (running, or died -- see below)
#   .started + .done        -> finished; read the verdict
#   .done without .started  -> an ANOMALY (the marker was deleted mid-run); the
#                              verdict is still authoritative for how it ended
#
# The marker exists because absence is otherwise ambiguous: a missing .done alone
# cannot distinguish "never started" from "still running", and a detached job
# given a bad path (a relative -File against $HOME, measured 2026-09-18) fails in
# under a second while looking exactly like a run in progress. The job id answers
# whether the LAUNCH was accepted; the marker answers whether the SCRIPT BODY
# BEGAN. Both are needed and neither replaces the other.
#
# Three rules make these files trustworthy rather than decorative:
#   1. ORDER: marker at launch, verdict on exit, .done last. A .done therefore
#      implies a complete verdict, and a .started implies the work actually began.
#   2. INVALIDATE FIRST: Clear-ArmVerdict deletes ALL of them BEFORE the run, so a
#      state from a previous run can never be read as this run's. This is the
#      failure mode that makes hand-rolled completion files untrustworthy -- a PASS
#      from yesterday read as a PASS from today.
#   3. NEVER INFER COMPLETION FROM THE MARKER: a killed process leaves .started and
#      no .done. Get-ArmRunState reports that as 'died' using a BEST-EFFORT pid
#      liveness check, and the answer for a finished run is always the verdict.
#
# Why not just read the log: the log is text, unbounded, and says nothing about
# whether the process finished.

function Get-ArmVerdictPaths {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    return [pscustomobject]@{
        dir     = $Dir
        started = (Join-Path $Dir ($Tag + '.started'))
        verdict = (Join-Path $Dir ($Tag + '.verdict.json'))
        done    = (Join-Path $Dir ($Tag + '.done'))
    }
}

# Removes the marker AND the verdict AND the sentinel. Call this BEFORE a run
# starts, never during one: it is what stops a previous run's state being read as
# the current run's.
function Clear-ArmVerdict {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    foreach ($f in @($p.started, $p.verdict, $p.done)) {
        if (Test-Path -LiteralPath $f) { Remove-Item -LiteralPath $f -Force -ErrorAction Continue }
    }
}

# Marks a run as STARTED. Call this at the launch boundary -- before the mutex
# wait and before any model load -- so the marker distinguishes "never got going"
# from "still working" for the whole duration of the run, not just at the end.
function Write-ArmStart {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag,
        [string] $Command = '',
        [string] $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1',
        [string] $LogPath = ''
    )
    if (-not (Test-Path -LiteralPath $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }
    $m = [ordered]@{
        schema      = 'arm-start/1'
        arm         = $Tag
        pid         = $PID
        host        = $env:COMPUTERNAME
        started_utc = (Get-Date).ToUniversalTime().ToString('o')
        mutex       = $MutexName
        command     = $Command
        log         = $LogPath
    }
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    [System.IO.File]::WriteAllText($p.started, ($m | ConvertTo-Json -Depth 4),
        (New-Object System.Text.UTF8Encoding($false)))
    return [pscustomobject]$m
}

function Read-ArmStart {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    if (-not (Test-Path -LiteralPath $p.started)) { return $null }
    return (Get-Content -LiteralPath $p.started -Raw | ConvertFrom-Json)
}

# Best-effort liveness for a marker. $null means "cannot tell" (no pid recorded,
# or the marker came from another host), NOT "dead" -- a nil answer must never be
# read as a death certificate.
function Test-ArmMarkerAlive {
    param($Marker)
    if ($null -eq $Marker) { return $null }
    if (-not $Marker.pid) { return $null }
    if ($Marker.host -and $env:COMPUTERNAME -and $Marker.host -ne $env:COMPUTERNAME) { return $null }
    $proc = Get-Process -Id ([int]$Marker.pid) -ErrorAction SilentlyContinue
    return [bool]$proc
}

# The one call a caller needs for "what happened to this run": no polling, no log
# tailing. State is one of:
#   never-started  no marker and no sentinel
#   running        marker present, no sentinel, and the recorded pid is alive
#   died           marker present, no sentinel, and the recorded pid is GONE --
#                  started and vanished without writing a verdict (hard kill, or a
#                  crash outside the termination handler)
#   finished       sentinel present; $result.verdict carries how it ended. If the
#                  marker is missing alongside it, $result.anomaly says so.
# `alive` is $null when liveness cannot be determined, in which case a marker with
# no sentinel reads as 'running' -- the conservative reading: pending, not failed.
function Get-ArmRunState {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    $marker  = Read-ArmStart -Dir $Dir -Tag $Tag
    $verdict = if (Test-Path -LiteralPath $p.done) { Read-ArmVerdict -Dir $Dir -Tag $Tag } else { $null }
    $alive   = Test-ArmMarkerAlive -Marker $marker

    $state = 'never-started'
    $anomaly = ''
    if ($null -ne $verdict) {
        $state = 'finished'
        if ($null -eq $marker) { $anomaly = 'sentinel present but start marker missing' }
    } elseif ($null -ne $marker) {
        $state = if ($alive -eq $false) { 'died' } else { 'running' }
    }

    return [pscustomobject]@{
        tag         = $Tag
        state       = $state
        marker      = $marker
        verdict     = $verdict
        alive       = $alive
        marker_path = $p.started
        done_path   = $p.done
        anomaly     = $anomaly
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

# Blocks until the start marker appears or the timeout expires. This is the
# launch-confirmation wait: for a job that was launched, a $null here means the
# script body never began within the window -- the signature of a bad path, a
# rejected permission, or a shell that could not start at all. Bounded on purpose;
# never replace it with an unconditional sleep.
function Wait-ArmStart {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [Parameter(Mandatory = $true)][string] $Tag,
        [int] $TimeoutSeconds = 60,
        [int] $PollSeconds = 1,
        [switch] $Quiet
    )
    $p = Get-ArmVerdictPaths -Dir $Dir -Tag $Tag
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path -LiteralPath $p.started) { return (Read-ArmStart -Dir $Dir -Tag $Tag) }
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not $Quiet) { Write-Warning ("Wait-ArmStart did not see the marker within {0}s ({1})" -f $TimeoutSeconds, $p.started) }
    return $null
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

# ---------------- GPU busy probe (T-GPLR-50 contention follow-up) -----------
# The GPU mutex is cooperative: a non-conforming project can hold VRAM and run
# compute DURING a mutex-held arm and poison it (uniformly-inflated per-layer
# compute, teardown cudaFree OOM, normal transfer ratios -- measured 2026-09-21).
# These helpers let a wrapper check the observable GPU state before trusting a
# hold and record what it saw in the verdict.

# One compute-apps query. Returns a list of { pid, name, short } rows, an EMPTY
# list when the GPU has no compute apps, or $null when the probe cannot run
# (nvidia-smi missing/errored). $null means UNVERIFIED -- never read it as
# clean. Tests can stub the probe by setting GPU_BUSY_PROBE_CMD to a command
# that prints nvidia-smi-compatible CSV (pid, process_name[, used_memory]).
function Get-GpuComputeApps {
    param([string] $NvidiaSmi = 'nvidia-smi')
    $out = $null
    try {
        if ($env:GPU_BUSY_PROBE_CMD) {
            $out = & cmd.exe /d /c $env:GPU_BUSY_PROBE_CMD 2>&1
            if ($LASTEXITCODE -ne 0) { return $null }
        } else {
            $out = & $NvidiaSmi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
        }
    } catch {
        return $null
    }
    if (-not $env:GPU_BUSY_PROBE_CMD -and $LASTEXITCODE -ne 0) { return $null }
    $apps = @()
    foreach ($line in @($out)) {
        $s = ("$line").Trim()
        if (-not $s) { continue }
        $parts = $s.Split(',')
        if ($parts.Count -lt 2) { continue }
        $procId = 0
        if (-not [int]::TryParse($parts[0].Trim(), [ref]$procId)) { continue }
        $name = $parts[1].Trim()
        if (-not $name) { continue }
        $apps += [pscustomobject]@{ pid = $procId; name = $name; short = (Split-Path -Leaf $name) }
    }
    return , $apps
}

# Human-readable one-line summary for logs and verdict fields.
function Format-GpuBusy {
    param($Apps)
    if ($null -eq $Apps) { return 'probe-unavailable' }
    $list = @($Apps)
    if ($list.Count -eq 0) { return 'clean' }
    return (($list | ForEach-Object { '{0} {1}' -f $_.pid, $_.short }) -join '; ')
}
