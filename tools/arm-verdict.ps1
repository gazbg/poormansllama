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

# ---- provenance wiring (2026-09-22) ----------------------------------------
# Every verdict carries `fields.provenance`: a digest of the engine source tree
# the binary was built from plus the identity of the binary this run used, so a
# later reader can tell whether the evidence still describes the current tree.
# The mechanism lives HERE, not only in tools/with-gpu-lock.ps1, because the
# wrapper is not the only writer: the arm drivers write their own verdicts for
# their own tags (`gplr53-a4-pin.verdict.json` carries none of the wrapper's
# fields -- measured 2026-09-22), and those ARE the arm verdicts the record cites.
# A writer's own capture is the fallback; a caller (the wrapper) that captured
# BEFORE and AFTER the run supplies the richer set instead.
#
# The tools directory is captured at DOT-SOURCE time on purpose: inside a
# function, $PSCommandPath is the CALLER's script, so resolving the repo root
# from a function body would silently point at the wrong tree.
$script:ArmVerdictToolsDir = if ($PSScriptRoot) { $PSScriptRoot } else { $null }
$script:ArmVerdictRepoRoot = if ($script:ArmVerdictToolsDir) { Split-Path $script:ArmVerdictToolsDir -Parent } else { $null }
$script:ArmProvenanceModule = if ($script:ArmVerdictToolsDir) { Join-Path $script:ArmVerdictToolsDir 'source-revision.ps1' } else { '' }
$script:ArmProvenanceAvailable = $false
if ($script:ArmProvenanceModule -and (Test-Path -LiteralPath $script:ArmProvenanceModule)) {
    . $script:ArmProvenanceModule
    $script:ArmProvenanceAvailable = $true
}

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

# The binary paths this RUN used. Explicit paths win; then an explicit command;
# then the run's own start marker, which records the command its launcher ran.
# That last fallback is why the arm drivers need no change: they already write a
# marker naming the arm's command (e.g. run-gplr53-a4-arm.ps1:131), so their
# verdicts get the same binary identity the wrapper gets. An empty result is
# recorded as binary_state=not-supplied -- absence, never a guess.
function Resolve-ArmArtifactPaths {
    param(
        [string]   $Dir = '',
        [string]   $Tag = '',
        [string]   $Command = '',
        [string[]] $ArtifactPaths = @()
    )
    if ($null -ne $ArtifactPaths -and @($ArtifactPaths).Count -gt 0) { return @($ArtifactPaths) }
    $candidateCommands = @()
    if (-not [string]::IsNullOrWhiteSpace($Command)) { $candidateCommands += $Command }
    if (-not [string]::IsNullOrWhiteSpace($Dir) -and -not [string]::IsNullOrWhiteSpace($Tag)) {
        $marker = Read-ArmStart -Dir $Dir -Tag $Tag
        if ($marker -and -not [string]::IsNullOrWhiteSpace("$($marker.command)")) {
            $candidateCommands += "$($marker.command)"
        }
    }
    foreach ($c in $candidateCommands) {
        $found = @(Get-ProvenanceArtifacts -Command $c)
        if ($found.Count -gt 0) { return $found }
    }
    return @()
}

# One capture in the flat verdict shape, with the failure to capture recorded as
# DATA rather than swallowed: a verdict whose provenance is missing must be
# readable as "not certifiable", never as "unchanged".
function Get-ArmVerdictProvenance {
    param(
        [string]   $Dir = '',
        [string]   $Tag = '',
        [string]   $Command = '',
        [string[]] $ArtifactPaths = @(),
        [string]   $SourceRoot = '',
        [string]   $RepoRoot = '',
        [string]   $Origin = 'verdict-write'
    )
    if (-not $script:ArmProvenanceAvailable) {
        return [ordered]@{ schema = 'arm-provenance/1'; origin = 'unavailable'
                           source_state = 'module-missing'; source_digest = ''
                           source_error = 'tools/source-revision.ps1 could not be loaded'
                           binary_state = 'not-supplied'; modules = [ordered]@{} }
    }
    $root = if ($RepoRoot) { $RepoRoot } else { $script:ArmVerdictRepoRoot }
    $src  = if ($SourceRoot) { $SourceRoot } elseif ($root) { Join-Path $root 'third_party\llama.cpp' } else { '' }
    $paths = Resolve-ArmArtifactPaths -Dir $Dir -Tag $Tag -Command $Command -ArtifactPaths $ArtifactPaths
    $cap = Get-ArmProvenance -SourceRoot $src -RepoRoot $root -ArtifactPaths $paths
    return (Format-ArmProvenance -Provenance $cap -Origin $Origin)
}

# ---- the provenance GATE (2026-09-22) --------------------------------------
# A capture that nothing requires is a suggestion. These functions make it
# binding at the point this program's own arms LAUNCH, so an arm that cannot
# attest which tree and which binary it describes is refused (or explicitly
# waived) instead of passing silently into the record.
#
# The gate tests what the RUN ITSELF can attest -- a capture exists, is
# well-formed, names the binary, and records no in-run anomaly. Whether the tree
# has moved since is a different question, asked of the current moment: that is
# the checker's verdict (tools/check-arm-provenance.ps1), printed here as a NOTE
# so a reader sees it at the end of the session, but NOT a reason to fail a run
# whose evidence was sound when it was written.
#
# Selection is by tag list or by `-SinceUtc` (the arms this launch produced),
# so a launcher never has to keep a tag list in sync with a driver.
# What makes a capture ATTESTABLE - one owner, used by the gate that refuses a run and
# by the writer that records one, so the two can never disagree about what "a verdict
# that can attest its tree and binary" means.
function Get-AttestationFailure {
    param([object] $Capture = $null)
    $fail = @()
    if ($null -eq $Capture) { return @('no provenance capture in the verdict') }
    if ("$($Capture.source_state)" -ne 'git') {
        $fail += ("the engine tree could not be digested at capture (source_state={0}{1})" -f $Capture.source_state, $(if ($Capture.source_error) { ": $($Capture.source_error)" } else { '' }))
    } elseif (">$($Capture.source_digest)".Length -ne 17) {
        $fail += 'the capture carries no 16-hex source digest'
    }
    if ("$($Capture.binary_state)" -ne 'hashed') {
        $fail += ("no binary identity was captured (binary_state={0})" -f $Capture.binary_state)
    }
    if ($Capture.sources_newer_than_binary -eq $true) { $fail += ("the binary PREDATES its sources: {0}" -f $Capture.sources_newer_detail) }
    if ($Capture.sources_changed_during_run -eq $true) { $fail += 'the sources changed WHILE this arm ran' }
    if ($Capture.binary_changed_during_run -eq $true) { $fail += 'the binary changed WHILE this arm ran' }
    return @($fail)
}

function Get-ArmProvenanceGate {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [string[]] $Tags = @(),
        [string] $SinceUtc = '',
        [string] $RepoRoot = ''
    )
    $root = if ($RepoRoot) { $RepoRoot } else { $script:ArmVerdictRepoRoot }
    $selected = @()
    if (@($Tags).Count -gt 0) {
        $selected = @($Tags | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    } elseif (Test-Path -LiteralPath $Dir) {
        foreach ($f in @(Get-ChildItem -LiteralPath $Dir -Filter '*.verdict.json' -File -ErrorAction SilentlyContinue | Sort-Object Name)) {
            $tag = $f.Name -replace '\.verdict\.json$', ''
            if ($SinceUtc) {
                $v = Read-ArmVerdict -Dir $Dir -Tag $tag
                if ($null -eq $v -or -not $v.started_utc) { continue }
                if ([datetime]$v.started_utc -lt [datetime]$SinceUtc) { continue }
            }
            $selected += $tag
        }
    }

    $arms = @()
    $failures = @()
    $notes = @()
    foreach ($tag in $selected) {
        $v = Read-ArmVerdict -Dir $Dir -Tag $tag
        $fail = @()
        if ($null -eq $v) { $fail += 'no verdict file' }
        elseif (-not $v.done_present) { $fail += 'verdict written but no sentinel beside it' }
        else {
            $fail += @(Get-AttestationFailure -Capture $v.fields.provenance)
        }
        if ($fail.Count -gt 0) { $failures += $tag }
        $arms += [pscustomobject]@{ tag = $tag; status = "$($v.status)"; reasons = $fail
                                    digest = if ($v) { "$($v.fields.provenance.source_digest)" } else { '' }
                                    lib_sha = if ($v) { "$($v.fields.provenance.streaming_lib_sha)" } else { '' }
                                    origin = if ($v) { "$($v.fields.provenance.origin)" } else { '' }
                                    complete = ($fail.Count -eq 0) }

        # Freshness vs the current moment: a NOTE, never a failure here.
        if ($fail.Count -eq 0 -and $root) {
            $cur = Format-ArmProvenance -Provenance (Get-ArmProvenance -SourceRoot (Join-Path $root 'third_party\llama.cpp') -RepoRoot $root)
            $cmp = Compare-ArmProvenance -Recorded $v.fields.provenance -Current $cur -RepoRoot $root
            if ($cmp.exit -ne 0) { $notes += ("{0}: {1} - {2}" -f $tag, $cmp.verdict, ($cmp.reasons -join '; ')) }
        }
    }
    return [pscustomobject]@{ dir = $Dir; arms = $arms; checked = @($arms).Count
                             failures = @($failures); notes = @($notes) }
}

# Print the gate and decide. Returns $true when the run may pass (everything
# complete, or the escape was taken explicitly). The escape appends a line to
# <Dir>/provenance-waivers.log so a waiver is a recorded act, not a quiet one.
function Assert-ArmProvenance {
    param(
        [Parameter(Mandatory = $true)][string] $Dir,
        [string[]] $Tags = @(),
        [string] $SinceUtc = '',
        [string] $RepoRoot = '',
        [string] $Label = 'arm',
        [string] $WaiverNote = '',
        [switch] $AllowUnprovenanced,
        [switch] $Quiet
    )
    $g = Get-ArmProvenanceGate -Dir $Dir -Tags $Tags -SinceUtc $SinceUtc -RepoRoot $RepoRoot
    if ($g.checked -eq 0) {
        # An empty selection is a FAILED attestation, not a vacuous pass: a
        # launcher that asks the gate about arms and finds no verdicts has no
        # evidence to certify, and a gate that shrugs there is exactly the silent
        # pass this whole mechanism exists to remove.
        Write-Host ("PROVENANCE GATE: REFUSING - no {0} verdict was found to attest in {1}." -f $Label, $Dir) -ForegroundColor Red
        if ($SinceUtc) { Write-Host ("  (selection was arms started at or after {0})" -f $SinceUtc) }
        Write-Host '  A run that produced no capturable verdict has no attributable evidence.'
        return $false
    }
    if (-not $Quiet) {
        Write-Host ("PROVENANCE GATE: {0} {1} verdict(s) checked - {2} complete, {3} refused" -f `
            $g.checked, $Label, ($g.checked - $g.failures.Count), $g.failures.Count)
        foreach ($a in $g.arms) {
            if ($a.complete) {
                Write-Host ("  PASS   {0,-26} digest {1}  {2} {3}  ({4})" -f $a.tag, $a.digest, 'llama.dll', $a.lib_sha, $a.origin)
            } else {
                Write-Host ("  REFUSE {0,-26} {1}" -f $a.tag, ($a.reasons -join '; '))
            }
        }
        foreach ($n in $g.notes) { Write-Host ("  NOTE   {0}" -f $n) }
    }
    if ($g.failures.Count -eq 0) { return $true }

    if ($AllowUnprovenanced) {
        $stamp = (Get-Date).ToUniversalTime().ToString('o')
        $line = ('{0} WAIVED label={1} arms={2} note={3}' -f $stamp, $Label, ($g.failures -join ','), $WaiverNote)
        if (-not (Test-Path -LiteralPath $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }
        Add-Content -LiteralPath (Join-Path $Dir 'provenance-waivers.log') -Value $line -Encoding UTF8
        Write-Host '  WAIVER: -AllowUnprovenanced given - the run may pass with uncertifiable arms.'
        Write-Host ("  WAIVER RECORDED: {0}" -f (Join-Path $Dir 'provenance-waivers.log'))
        Write-Host ("  WAIVER LINE: {0}" -f $line)
        return $true
    }

    Write-Host ("PROVENANCE GATE: REFUSING - {0} of {1} {2} verdict(s) cannot attest which tree and binary they describe." -f `
        $g.failures.Count, $g.checked, $Label) -ForegroundColor Red
    Write-Host '  This is not a performance judgement; it means the evidence is not attributable to a revision.'
    Write-Host '  Escape (explicit, recorded): re-run with -AllowUnprovenanced.'
    return $false
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
        [string]   $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1',
        [string[]] $ArtifactPaths = @(),
        [string]   $SourceRoot = '',
        [string]   $RepoRoot = '',
        # A verdict that is NOT an arm result (a tool's own bookkeeping verdict, or a
        # harness fixture the gate is about to judge) says so explicitly. Without this,
        # such a verdict would be downgraded below rather than the rule being weakened.
        [switch]   $AllowUnattested
    )
    # Status is DERIVED from the exit code when not given explicitly, so a caller
    # cannot accidentally report PASS for a failing run. Exit 10 is the provenance
    # contract the live gates use, and it reads UNCERTIFIED here so a refusal has ONE
    # name everywhere instead of FAIL-in-one-place and uncertified-in-another.
    if ([string]::IsNullOrWhiteSpace($Status)) {
        $Status = if ($ExitCode -eq 0) { 'PASS' } elseif ($ExitCode -eq 124) { 'TIMEOUT' } elseif ($ExitCode -eq 10) { 'UNCERTIFIED' } else { 'FAIL' }
    }
    if (-not (Test-Path -LiteralPath $Dir)) { New-Item -ItemType Directory -Force -Path $Dir | Out-Null }

    # A caller that captured provenance around the run (the wrapper) owns the
    # field; otherwise this writer captures its own, so no verdict -- from any
    # writer -- is left unable to say which sources and binary it describes.
    if (-not $Fields.ContainsKey('provenance')) {
        $Fields['provenance'] = Get-ArmVerdictProvenance -Dir $Dir -Tag $Tag -Command $Command `
            -ArtifactPaths $ArtifactPaths -SourceRoot $SourceRoot -RepoRoot $RepoRoot -Origin 'verdict-write'
    }

    # --- refuse to leave an UNATTESTED PASS ----------------------------------------
    # The structural hole the inventory left: a runner that never called the gate could
    # still record a PASS, and because the capture above is automatic such a verdict
    # LOOKED certifiable to a later reader. The rule is enforced HERE, at the one place
    # every verdict is written, and it is about ATTESTABILITY rather than about whether a
    # gate ran: a verdict that cannot name its tree and binary is recorded as UNCERTIFIED
    # under the SAME exit contract the live gates use (10), so the exit code, the status
    # and the .done sentinel all agree. The evidence is never dropped - it stays on disk
    # with `uncertified`, the reasons, the status it would have had, and the command's own
    # exit code - and `-AllowUnattested` is the explicit opt-out for a verdict that is not
    # an arm result at all.
    if (-not $AllowUnattested -and $Status -eq 'PASS') {
        $attFail = @(Get-AttestationFailure -Capture $Fields['provenance'])
        if ($attFail.Count -gt 0) {
            $Fields['uncertified']          = $true
            $Fields['would_be_status']      = 'PASS'
            $Fields['attestation_reasons']  = @($attFail)
            $Fields['command_exit']         = $ExitCode
            $ExitCode = 10
            $Status   = 'UNCERTIFIED'
            Write-Host ("VERDICT UNCERTIFIED: {0} cannot attest its tree and binary - recorded as uncertified (exit 10), not as a pass." -f $Tag) -ForegroundColor Red
            foreach ($r in $attFail) { Write-Host ("  - {0}" -f $r) }
            Write-Host '  The evidence is kept; only its status changed. Escape: -AllowUnattested for a verdict that is not an arm result.'
        }
    }

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
# (nvidia-smi missing/errored, or output that carries no parseable row -- see the
# guard at the end of the function). $null means UNVERIFIED -- never read it as
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
    $nonEmpty = 0
    foreach ($line in @($out)) {
        $s = ("$line").Trim()
        if (-not $s) { continue }
        $nonEmpty++
        $parts = $s.Split(',')
        if ($parts.Count -lt 2) { continue }
        $procId = 0
        if (-not [int]::TryParse($parts[0].Trim(), [ref]$procId)) { continue }
        $name = $parts[1].Trim()
        if (-not $name) { continue }
        $apps += [pscustomobject]@{ pid = $procId; name = $name; short = (Split-Path -Leaf $name) }
    }
    # Output that is NONEMPTY but yields no parseable row is an answered-nothing
    # probe, not an idle card: a dropped --format flag, a driver banner, an
    # nvidia-smi whose output shape changed. Measured 2026-09-22 BEFORE this
    # guard: that case returned an EMPTY list and Format-GpuBusy reported 'clean',
    # so it was persisted as `gpu_pre_run: clean` into every wrapped arm's verdict
    # and was indistinguishable from a genuinely idle GPU. An unparseable probe
    # must read as UNVERIFIED ($null), never as evidence of a clean card.
    if ($nonEmpty -gt 0 -and $apps.Count -eq 0) { return $null }
    return , $apps
}

# Machine-readable classification of a Format-GpuBusy summary, so a consumer does
# not have to substring-match a human sentence to tell an idle card from a probe
# that did not answer from a probe that was never taken. 'not-probed' is the state
# of a run that ended before the probe (mutex timeout, busy refusal): ABSENCE --
# never 'clean', and never a default.
function Get-GpuProbeStatus {
    param([string] $BusySummary = '')
    if ([string]::IsNullOrWhiteSpace($BusySummary)) { return 'not-probed' }
    switch ($BusySummary) {
        'clean'             { return 'clean' }
        'probe-unavailable' { return 'unavailable' }
        '__unprobed__'      { return 'not-probed' }
        default             { return 'busy' }
    }
}

# Human-readable one-line summary for logs and verdict fields.
function Format-GpuBusy {
    param($Apps)
    if ($null -eq $Apps) { return 'probe-unavailable' }
    $list = @($Apps)
    if ($list.Count -eq 0) { return 'clean' }
    return (($list | ForEach-Object { '{0} {1}' -f $_.pid, $_.short }) -join '; ')
}
