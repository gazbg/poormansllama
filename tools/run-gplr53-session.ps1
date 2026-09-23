# T-GPLR-53 session launcher: ONE documented command for the whole owner-invoked
# GPU spend. It runs the A3 arm driver (50k + 200k, one mutex hold) and the A4 arm
# driver (the pre-warm modality arms at 4k, one mutex hold) unattended.
#
#   tools/run-gplr53-session.ps1                 # A3 then A4, each in its own mutex hold
#   tools/run-gplr53-session.ps1 -Session a3     # A3 only
#   tools/run-gplr53-session.ps1 -ValidateOnly   # DRY: no mutex, no CUDA, prints the arms
#   tools/run-gplr53-session.ps1 -PreflightOnly  # probe the card only; launch nothing
#   tools/run-gplr53-session.ps1 -ArchiveOnly    # snapshot current evidence only; launch nothing
#
# EVIDENCE SAFETY (T-GPLR-53, 2026-09-22). The arm tags are FIXED, so a re-run
# overwrites the previous run's trace, log and verdict triple at the same paths.
# That is not theoretical: the second A3 run replaced the first run's artifacts,
# leaving the original FAIL evidence only in the record prose. Every real run
# therefore ARCHIVES the artifacts it is about to replace into a per-run directory
# (`build/llama-cuda/runs/<utc-stamp>/` + `manifest.json` with sizes and hashes),
# BEFORE invalidating anything. Copies only - nothing is moved, renamed or
# modified, so the accepted A3 evidence and every documented analysis path stay
# exactly where they are. The archive step sits AFTER the busy preflight, so a
# refused run writes nothing, and BEFORE Clear-ArmVerdict, so a real run cannot
# lose its predecessor.
#
# FAST PRE-LAUNCH BUSY PROBE (T-GPLR-53, 2026-09-22). The arm drivers keep the
# recorded T-GPLR-52 pattern: acquire the mutex, THEN probe the device. That order
# is a hazard for a one-command session, because the wait comes first
# (WaitOne(3600 s) at run-gplr53-a3-arm.ps1:103 precedes its probe at :115, and the
# A4 driver is identical) - so launching this script against a busy card would not
# refuse in seconds, it would block for up to an hour and only then exit 125. This
# launcher therefore does a READ-ONLY probe of its own (no mutex, no CUDA, no
# driver process) BEFORE it starts anything, and refuses fast:
#
#   any visible compute app -> print the list, exit 125, start no driver, write no
#                              arm artifact, acquire NO mutex
#   probe unavailable       -> WARN and launch; the drivers probe again themselves
#                              under the mutex, so the guarantee is never silently
#                              dropped (this mirrors with-gpu-lock.ps1's UNVERIFIED)
#   clean                   -> launch exactly as before
#
# The probe is stubbed deterministically by the hook the shared helper already
# documents: GPU_BUSY_PROBE_CMD (see Get-GpuComputeApps in tools/arm-verdict.ps1),
# which lets both the busy and the clean path be exercised host-side with no GPU.
#
# DRIVER OUTPUT + EXIT CODE (T-GPLR-53, 2026-09-22). Each arm driver used to be
# started with -WindowStyle Hidden and no redirection, so everything it printed was
# discarded and its exit code was never read: the per-arm verdict files were the
# only surviving output, which meant a driver's OWN computed verdict lines (the
# pre-registered removal table, the FALSIFIER line, the dose curve) existed nowhere
# on disk. That is a provenance hole, not a cosmetic one - a driver whose post-arm
# computation threw after writing PASS verdicts looked identical to a clean run.
# Each driver's stdout and stderr are now REDIRECTED to per-session files
# (`<session>-driver-<a3|a4>.out` / `.err`), the last lines are echoed into this
# log, BOTH files are archived with the rest of the run's evidence, and the driver's
# exit code is reported and folded into the session verdict (a non-zero driver exit
# fails the session even when every arm file says PASS).
#
# Contract: A3 and A4 are ONE session each (separate mutex holds), so this launcher
# never merges them into a single hold. Every arm script acquires and releases the
# shared mutex itself and refuses (exit 125) when the GPU busy probe is not clean;
# this launcher never passes -AllowBusy and never clears another holder's lock.
#
# Detached-launch rule (AGENTS.md): the LAUNCHER - not the arm - dot-sources
# tools/arm-verdict.ps1 and runs Clear-ArmVerdict for every tag it is about to
# launch, BEFORE Start-Process, so a previous run's marker/verdict/sentinel can
# never be read as this run's. Start-Process is used on purpose (never setsid).
param(
    [ValidateSet('a3', 'a4', 'both')][string] $Session = 'both',
    [int] $ArmTimeoutSeconds = 5400,
    [switch] $ValidateOnly,
    # Snapshot the artifacts this run would replace, then exit 0 without launching
    # anything: host-only, no mutex, no CUDA, so it works on a busy card too.
    [switch] $ArchiveOnly,
    # Probe the device and report, then exit 0 (clean) / 125 (busy) without
    # launching anything. The clean-card path is otherwise only reachable by
    # actually starting the drivers, which acquire the mutex.
    [switch] $PreflightOnly,
    # Provenance gate escape (2026-09-22). By default this session FAILS when an
    # arm's verdict cannot attest which source tree and which binary it describes
    # (the drivers gate too, at their own exit). Passed through to the drivers,
    # and the waiver is appended to verdicts/provenance-waivers.log so an escape
    # is a recorded act rather than a quiet one.
    [switch] $AllowUnprovenanced
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$verdictDir = Join-Path $base 'verdicts'
. (Join-Path $root 'tools\arm-verdict.ps1')

$a3Script = Join-Path $root 'tools\run-gplr53-a3-arm.ps1'
$a4Script = Join-Path $root 'tools\run-gplr53-a4-arm.ps1'
$a3Tags = @('gplr53-a3-50k', 'gplr53-a3-200k')
$a4Tags = @('gplr53-a4-pw0', 'gplr53-a4-pw1', 'gplr53-a4-pw2', 'gplr53-a4-pw3', 'gplr53-a4-delay', 'gplr53-a4-wrongtouch',
            'gplr53-a4-d000', 'gplr53-a4-d200', 'gplr53-a4-d600', 'gplr53-a4-d1200', 'gplr53-a4-d1700',
            'gplr53-a4-pin', 'gplr53-a4-r2')

$plan = @()
if ($Session -eq 'a3' -or $Session -eq 'both') {    $plan += @{ script = $a3Script; tags = $a3Tags; slug = 'a3'; name = 'A3 (decay discriminator: 50k + 200k)' } }
if ($Session -eq 'a4' -or $Session -eq 'both') { $plan += @{ script = $a4Script; tags = $a4Tags; slug = 'a4'; name = 'A4 (pre-warm modality: 4k)' } }

foreach ($step in $plan) {
    if (-not (Test-Path -LiteralPath $step.script)) { Write-Host "MISSING ARM DRIVER: $($step.script)"; exit 1 }
}

if ($ValidateOnly) {
    $rc = 0
    foreach ($step in $plan) {
        Write-Host ("==== DRY: {0} ====" -f $step.name)
        & powershell -NoProfile -ExecutionPolicy Bypass -File $step.script -ValidateOnly
        if ($LASTEXITCODE -ne 0) { $rc = 1 }
    }
    Write-Host ("DRY-RUN {0}" -f $(if ($rc -eq 0) { 'OK (inputs present; no mutex, no CUDA)' } else { 'FAILED' }))
    exit $rc
}

# ----------------------------------------------------- evidence archiving
# Copies (never moves) the artifacts the tags are about to replace, so each run's
# evidence survives next to the previous run's. Kept here rather than in the arm
# drivers so it applies to the single documented entry point, and so the drivers'
# behaviour is unchanged.
function Save-RunEvidence {
    param(
        [Parameter(Mandatory = $true)] $Steps,
        [string] $Root,
        [string] $Name,
        # Session prefix of the driver stdout/stderr files, so the PREVIOUS run's
        # driver output (the only place its computed verdict lines live) is kept
        # too, not just the per-tag artifacts.
        [string] $LogPrefix
    )
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $dir = Join-Path $Root $stamp
    $n = 0
    $entries = @()
    $srcsAll = @()
    foreach ($step in $Steps) {
        foreach ($t in $step.tags) {
            $srcsAll += @(
                (Join-Path $base ($t + '.jsonl')),
                (Join-Path $base ($t + '.log')),
                (Join-Path $verdictDir ($t + '.started')),
                (Join-Path $verdictDir ($t + '.verdict.json')),
                (Join-Path $verdictDir ($t + '.done'))
            )
        }
    }
    if ($LogPrefix) {
        $srcsAll += @(Get-ChildItem -LiteralPath $base -Filter ($LogPrefix + '-driver-*.out') -ErrorAction SilentlyContinue |
                      ForEach-Object { $_.FullName })
        $srcsAll += @(Get-ChildItem -LiteralPath $base -Filter ($LogPrefix + '-driver-*.err') -ErrorAction SilentlyContinue |
                      ForEach-Object { $_.FullName })
    }
    foreach ($s in $srcsAll) {
        if (-not (Test-Path -LiteralPath $s)) { continue }
        if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
        $dst = Join-Path $dir (Split-Path -Leaf $s)
        Copy-Item -LiteralPath $s -Destination $dst -Force
        $entries += [ordered]@{
            file      = (Split-Path -Leaf $s)
            bytes     = (Get-Item -LiteralPath $dst).Length
            sha256_16 = (Get-FileHash -Algorithm SHA256 -LiteralPath $dst).Hash.Substring(0, 16)
            source    = $s
        }
        $n++
    }
    if ($n -eq 0) { return $null }
    $manifest = [ordered]@{
        schema       = 'gplr-run-evidence/1'
        session      = $Name
        archived_utc = (Get-Date).ToUniversalTime().ToString('o')
        note         = 'copies of the artifacts this run was about to replace; originals untouched'
        files        = $entries
    }
    [System.IO.File]::WriteAllText((Join-Path $dir 'manifest.json'),
        ($manifest | ConvertTo-Json -Depth 5), (New-Object System.Text.UTF8Encoding($false)))
    return $dir
}

if ($ArchiveOnly) {
    $adir = Save-RunEvidence -Steps $plan -Root (Join-Path $base 'runs') -Name ('session ' + $Session) -LogPrefix $Session
    if ($null -eq $adir) {
        Write-Host 'ARCHIVE: nothing to archive (no artifact exists at these tags)'
    } else {
        Write-Host ('ARCHIVE OK: {0}' -f $adir)
    }
    exit 0
}

# --------------------------------------------- fast pre-launch busy probe
# Read-only: one nvidia-smi compute-apps query, no mutex, no CUDA, no driver.
# Placed AFTER the -ValidateOnly branch on purpose, so the dry run keeps working
# on a busy card exactly as it did before this probe existed.
# Get-GpuComputeApps returns an array of app rows, or $null when the probe cannot
# run; Format-GpuBusy maps those to 'clean' / 'probe-unavailable' / a list. Do NOT
# wrap the call in @(...): @($null) is a ONE-ELEMENT array and would read as an app.
$gpuProbe = Get-GpuComputeApps
$gpuState = Format-GpuBusy $gpuProbe
if ($gpuState -eq 'probe-unavailable') {
    Write-Warning 'GPU busy probe unavailable; launching anyway (the arm drivers probe again under the mutex).'
} elseif ($gpuState -ne 'clean') {
    Write-Host ('PREFLIGHT REFUSED (exit 125): GPU busy - {0}' -f $gpuState)
    Write-Host '  No driver was started, the shared mutex was NOT acquired, and no arm artifact was written.'
    Write-Host '  Re-run when nvidia-smi --query-compute-apps is empty; the busy gate in the arm drivers is unchanged.'
    exit 125
}
if ($PreflightOnly) {
    Write-Host ('PREFLIGHT OK: {0}; no driver launched, no mutex acquired.' -f $gpuState)
    Write-Host ('  would run: {0}' -f (($plan | ForEach-Object { $_.name }) -join ' then '))
    exit 0
}
Write-Host ('  preflight gpu state = {0}' -f $gpuState)

# Evidence safety: preserve what this run is about to replace, BEFORE clearing it.
$archived = Save-RunEvidence -Steps $plan -Root (Join-Path $base 'runs') -Name ('session ' + $Session) -LogPrefix $Session
if ($null -eq $archived) {
    Write-Host '  no predecessor artifacts to archive (nothing exists at these tags yet)'
} else {
    Write-Host ('  archived predecessor evidence -> {0}' -f $archived)
}

# Launcher-side invalidation, BEFORE any Start-Process.
foreach ($step in $plan) { foreach ($t in $step.tags) { Clear-ArmVerdict -Dir $verdictDir -Tag $t | Out-Null } }

$rolled = @()
$driverRows = @()
$provBad = @()
foreach ($step in $plan) {
    Write-Host ("==== SESSION: {0} ====" -f $step.name)
    $drvOut = Join-Path $base ("{0}-driver-{1}.out" -f $Session, $step.slug)
    $drvErr = Join-Path $base ("{0}-driver-{1}.err" -f $Session, $step.slug)
    $drvArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $step.script)
    if ($AllowUnprovenanced) { $drvArgs += '-AllowUnprovenanced' }
    $proc = Start-Process -FilePath 'powershell' `
        -ArgumentList $drvArgs `
        -RedirectStandardOutput $drvOut -RedirectStandardError $drvErr `
        -PassThru -WindowStyle Hidden
    # MEASURED TRAP (2026-09-22, host-only probe): with -RedirectStandardOutput/Error a
    # Start-Process -PassThru process object reports HasExited correctly but ExitCode as
    # NULL after a bounded WaitForExit - unless the handle is retained by setting
    # EnableRaisingEvents before the wait. Without this line the exit-code check below
    # would read null and fail every clean session ($null -ne 0 is true).
    $proc.EnableRaisingEvents = $true
    Write-Host ("  launched pid {0}" -f $proc.Id)

    foreach ($t in $step.tags) {
        $marker = Wait-ArmStart -Dir $verdictDir -Tag $t -TimeoutSeconds 120 -Quiet
        if ($null -eq $marker) { Write-Warning ("  {0}: no start marker within 120s" -f $t) }
        $v = Wait-ArmVerdict -Dir $verdictDir -Tag $t -TimeoutSeconds $ArmTimeoutSeconds -Quiet
        if ($null -eq $v) {
            $state = Get-ArmRunState -Dir $verdictDir -Tag $t
            Write-Warning ("  {0}: no verdict within {1}s (state {2})" -f $t, $ArmTimeoutSeconds, $state.state)
            $rolled += [pscustomobject]@{ tag = $t; status = 'NO-VERDICT'; exit = ''; state = $state.state }
        } else {
            Write-Host ("  {0}: {1} (exit {2})" -f $t, $v.status, $v.exit_code)
            $rolled += [pscustomobject]@{ tag = $t; status = $v.status; exit = $v.exit_code; state = 'finished' }
        }
    }
    # Capture the driver's own output and exit code (see the header note).
    $driverExit = 'not-exited'
    if (-not $proc.HasExited) { $proc.WaitForExit(15000) | Out-Null }
    if ($proc.HasExited) { $driverExit = $proc.ExitCode }
    else { Write-Warning ("  arm driver pid {0} still alive after its verdicts" -f $proc.Id) }
    $lines = @(Get-Content -LiteralPath $drvOut -ErrorAction SilentlyContinue | Where-Object { $_.Trim() -ne '' })
    Write-Host ("  driver exit = {0} | stdout {1} ({2} lines) | stderr {3}" -f $driverExit, $drvOut, $lines.Count, $drvErr)
    if ($lines.Count -gt 0) {
        Write-Host '  --- last driver lines (its own computed verdicts) ---'
        $lines | Select-Object -Last 20 | ForEach-Object { Write-Host ('  | ' + $_) }
    }
    $driverRows += [pscustomobject]@{ step = $step.slug; exit_code = $driverExit; stdout = (Split-Path -Leaf $drvOut) }

    # Provenance gate for the arms this step produced. The drivers gate their own
    # arms at their exit (exit 10), so this is the session-level statement of the
    # same fact - and it still bites if a driver was bypassed or replaced.
    if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags $step.tags `
            -Label ("{0} arm" -f $step.slug) -AllowUnprovenanced:$AllowUnprovenanced)) {
        $provBad += $step.slug
    }
}

Write-Host '==== ROLL-UP ===='
$rolled | Format-Table -AutoSize | Out-String | Write-Host
$driverRows | Format-Table -AutoSize | Out-String | Write-Host
$bad = @($rolled | Where-Object { $_.status -ne 'PASS' })
if ($bad.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0}/{1} arms not PASS)" -f $bad.Count, $rolled.Count)
    exit 1
}
# A driver that exited non-zero (or never exited) did not finish its own post-arm
# computation, so the session is not clean even when every arm file reads PASS.
if ($provBad.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0} step(s) produced arms with no attributable provenance: {1})" -f `
        $provBad.Count, ($provBad -join ', '))
    Write-Host '  The arms may still read PASS; what failed is that their evidence cannot be tied to a revision.'
    Write-Host '  Escape (explicit, recorded): re-run with -AllowUnprovenanced.'
    exit 10
}
$badDrivers = @($driverRows | Where-Object { $_.exit_code -ne 0 })
if ($badDrivers.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0} arm driver(s) exited {1}: see the driver stdout/stderr listed above)" -f `
        $badDrivers.Count, (($badDrivers | ForEach-Object { $_.exit_code }) -join '/'))
    exit 1
}
Write-Host ("SESSION VERDICT: PASS ({0}/{0} arms)" -f $rolled.Count)
exit 0
