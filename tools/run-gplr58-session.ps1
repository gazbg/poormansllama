# T-GPLR-58 A3/A4 session launcher: ONE documented command for each owner-invoked GPU
# spend. A3 runs five legs; A4 runs seven (off x2, a separate count-matched null/arm pair
# for bar09 and fusion, then retref@K4) at the declared 885-token short shape.
#
#   tools/run-gplr58-session.ps1                 # the A3 session (5 legs, one hold)
#   tools/run-gplr58-session.ps1 -Session p1     # A3 pilot family (backward-compatible alias)
#   tools/run-gplr58-session.ps1 -Session a3     # A3: off x2 + null/coact + retref@K4
#   tools/run-gplr58-session.ps1 -Session a4     # A4: off x2 + null/arm for bar09 and fusion + retref@K4
#   tools/run-gplr58-session.ps1 -ValidateOnly   # DRY: no mutex, no CUDA, prints the legs
#   tools/run-gplr58-session.ps1 -PreflightOnly  # probe the card only; launch nothing
#   tools/run-gplr58-session.ps1 -ArchiveOnly    # snapshot current evidence only; launch nothing
#
# EVIDENCE SAFETY (the T-GPLR-53 lesson, kept): the leg tags are FIXED, so a re-run
# overwrites the previous run's trace/log/verdict triple at the same paths. Every real run
# therefore ARCHIVES the artifacts it is about to replace into a per-run directory
# (`build/llama-cuda/runs/<utc-stamp>/` + `manifest.json` with sizes and hashes), and it
# does so AFTER the busy preflight (a refused run writes nothing) and BEFORE
# Clear-ArmVerdict (a real run cannot lose its predecessor). Copies only: nothing is moved,
# renamed or modified. The A0/A1/A2 recordings are NOT touched by this laundering: they live
# at different paths and are never overwritten by these tags.
#
# FAST PRE-LAUNCH BUSY PROBE (kept from the T-GPLR-53/T-56 launchers): the arm driver
# acquires the mutex and THEN probes the device, so its wait would come first (up to an hour)
# and only then exit 125. This launcher probes READ-ONLY first, with no mutex and no CUDA:
# any visible compute app -> print the list, exit 125, start nothing, take no lock, write no
# artifact; probe unavailable -> WARN and launch (the driver probes again under the mutex, so
# the guarantee is never silently dropped).
#
# DRIVER OUTPUT + EXIT CODE (kept): the driver's own computed band lines (the A2 report, the
# control spread, M3, the fill-room ceiling) exist only on its stdout, so they are redirected
# to `p1-driver-a3.out/.err`, echoed here, archived with the rest of the evidence, and the
# driver's exit code is folded into the session verdict.
#
# Detached-launch rule (AGENTS.md): the LAUNCHER - not the arm - dot-sources
# tools/arm-verdict.ps1 and runs Clear-ArmVerdict for every tag it is about to launch, BEFORE
# Start-Process, so a previous run's marker/verdict can never be read as this run's.
# Start-Process is used on purpose (never setsid).
param(
    # p1 is the historical A3 alias; a3 and a4 select the two pre-authorized holds.
    [ValidateSet('p1', 'a3', 'a4')][string] $Session = 'p1',
    [int] $ArmTimeoutSeconds = 5400,
    [switch] $ValidateOnly,
    [switch] $ArchiveOnly,
    [switch] $PreflightOnly,
    [switch] $AllowUnprovenanced,
    [switch] $ExternalMutex
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$verdictDir = Join-Path $base 'verdicts'
. (Join-Path $root 'tools\arm-verdict.ps1')

$armScript = Join-Path $root 'tools\run-gplr58-a3-arm.ps1'
$driverSession = if ($Session -eq 'p1') { 'a3' } else { $Session }
$sessionPrefix = "gplr58-$driverSession"
$driverPrefix = if ($Session -eq 'p1') { 'p1' } else { $Session }
if ($driverSession -eq 'a3') {
    $armTags = @('gplr58-a3-off1', 'gplr58-a3-off2', 'gplr58-a3-null', 'gplr58-a3-coact', 'gplr58-a3-retref4')
    $stepName = 'A3 pilot (off x2 + null/coact @K2 + retref@K4, one mutex hold)'
} else {
    $armTags = @('gplr58-a4-off1', 'gplr58-a4-off2',
                 'gplr58-a4-null-bar09', 'gplr58-a4-bar09',
                 'gplr58-a4-null-fusion', 'gplr58-a4-fusion',
                 'gplr58-a4-retref4')
    $stepName = 'A4 (off x2 + paired null/bar09 + paired null/fusion + retref@K4, one mutex hold)'
}
$plan = @(@{ script = $armScript; tags = $armTags; slug = $driverSession; name = $stepName })
foreach ($step in $plan) {
    if (-not (Test-Path -LiteralPath $step.script)) { Write-Host "MISSING ARM DRIVER: $($step.script)"; exit 1 }
}

if ($ValidateOnly) {
    $rc = 0
    foreach ($step in $plan) {
        Write-Host ("==== DRY: {0} ====" -f $step.name)
        & powershell -NoProfile -ExecutionPolicy Bypass -File $step.script -Session $driverSession -ValidateOnly
        if ($LASTEXITCODE -ne 0) { $rc = 1 }
    }
    Write-Host ("DRY-RUN {0}" -f $(if ($rc -eq 0) { 'OK (inputs present; no mutex, no CUDA)' } else { 'FAILED' }))
    exit $rc
}

function Save-RunEvidence {
    param(
        [Parameter(Mandatory = $true)] $Steps,
        [string] $Root,
        [string] $Name,
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
                (Join-Path $base ($t + '-bands.json')),
                (Join-Path $verdictDir ($t + '.started')),
                (Join-Path $verdictDir ($t + '.verdict.json')),
                (Join-Path $verdictDir ($t + '.done'))
            )
        }
    }
    $srcsAll += @((Join-Path $base ($sessionPrefix + '-summary.json')),
                  (Join-Path $base ($sessionPrefix + '-bands.json')))
    $srcsAll += @(Get-ChildItem -LiteralPath $base -Filter ($sessionPrefix + '-bands-*.json') -ErrorAction SilentlyContinue |
                  ForEach-Object { $_.FullName })
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
    $adir = Save-RunEvidence -Steps $plan -Root (Join-Path $base 'runs') -Name ('session ' + $driverSession) -LogPrefix $driverPrefix
    if ($null -eq $adir) { Write-Host 'ARCHIVE: nothing to archive (no artifact exists at these tags)' }
    else { Write-Host ('ARCHIVE OK: {0}' -f $adir) }
    exit 0
}

# --------------------------------------------- fast pre-launch busy probe
# Read-only: one nvidia-smi compute-apps query, no mutex, no CUDA, no driver. Do NOT wrap the
# call in @(...): @($null) is a ONE-ELEMENT array and would read as an app.
$gpuProbe = Get-GpuComputeApps
$gpuState = Format-GpuBusy $gpuProbe
if ($gpuState -eq 'probe-unavailable') {
    Write-Warning 'GPU busy probe unavailable; launching anyway (the arm driver probes again under the mutex).'
} elseif ($gpuState -ne 'clean') {
    Write-Host ('PREFLIGHT REFUSED (exit 125): GPU busy - {0}' -f $gpuState)
    Write-Host '  No driver was started, the shared mutex was NOT acquired, and no arm artifact was written.'
    Write-Host '  Re-run when nvidia-smi --query-compute-apps is empty; the busy gate in the arm driver is unchanged.'
    exit 125
}
if ($PreflightOnly) {
    Write-Host ('PREFLIGHT OK: {0}; no driver launched, no mutex acquired.' -f $gpuState)
    Write-Host ('  would run: {0}' -f (($plan | ForEach-Object { $_.name }) -join ' then '))
    exit 0
}
Write-Host ('  preflight gpu state = {0}' -f $gpuState)

# Evidence safety: preserve what this run is about to replace, BEFORE clearing it.
$archived = Save-RunEvidence -Steps $plan -Root (Join-Path $base 'runs') -Name ('session ' + $driverSession) -LogPrefix $driverPrefix
if ($null -eq $archived) { Write-Host '  no predecessor artifacts to archive (nothing exists at these tags yet)' }
else { Write-Host ('  archived predecessor evidence -> {0}' -f $archived) }

# Launcher-side invalidation, BEFORE any Start-Process.
foreach ($step in $plan) { foreach ($t in $step.tags) { Clear-ArmVerdict -Dir $verdictDir -Tag $t | Out-Null } }

$rolled = @()
$driverRows = @()
$provBad = @()
foreach ($step in $plan) {
    Write-Host ("==== SESSION: {0} ====" -f $step.name)
    $drvOut = Join-Path $base ("{0}-driver-{1}.out" -f $Session, $step.slug)
    $drvErr = Join-Path $base ("{0}-driver-{1}.err" -f $Session, $step.slug)
    $drvArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $step.script, '-Session', $driverSession)
    if ($AllowUnprovenanced) { $drvArgs += '-AllowUnprovenanced' }
    if ($ExternalMutex) { $drvArgs += '-ExternalMutex' }
    $proc = Start-Process -FilePath 'powershell' -ArgumentList $drvArgs `
        -RedirectStandardOutput $drvOut -RedirectStandardError $drvErr -PassThru -WindowStyle Hidden
    # MEASURED TRAP (2026-09-22): with redirection a Start-Process -PassThru object reports
    # HasExited correctly but ExitCode NULL after a bounded WaitForExit unless the handle is
    # retained by setting EnableRaisingEvents before the wait.
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
    $driverExit = 'not-exited'
    if (-not $proc.HasExited) { $proc.WaitForExit(15000) | Out-Null }
    if ($proc.HasExited) { $driverExit = $proc.ExitCode }
    else { Write-Warning ("  arm driver pid {0} still alive after its verdicts" -f $proc.Id) }
    $lines = @(Get-Content -LiteralPath $drvOut -ErrorAction SilentlyContinue | Where-Object { $_.Trim() -ne '' })
    Write-Host ("  driver exit = {0} | stdout {1} ({2} lines) | stderr {3}" -f $driverExit, $drvOut, $lines.Count, $drvErr)
    if ($lines.Count -gt 0) {
        Write-Host '  --- last driver lines (its own computed bands) ---'
        $lines | Select-Object -Last 26 | ForEach-Object { Write-Host ('  | ' + $_) }
    }
    $driverRows += [pscustomobject]@{ step = $step.slug; exit_code = $driverExit; stdout = (Split-Path -Leaf $drvOut) }

    if (-not (Assert-ArmProvenance -Dir $verdictDir -Tags $step.tags `
            -Label ("{0} leg" -f $step.slug) -AllowUnprovenanced:$AllowUnprovenanced)) {
        $provBad += $step.slug
    }
}

Write-Host '==== ROLL-UP ===='
$rolled | Format-Table -AutoSize | Out-String | Write-Host
$driverRows | Format-Table -AutoSize | Out-String | Write-Host
# A leg that was SKIPPED because a predecessor halted the series (exit 3) and a leg whose
# BANDS failed (still exit 0: a band is a result, not a session failure) are both reported as
# they are. What fails the session is a leg whose own integrity gates failed (exit 1: the leg
# is not usable evidence) or a REFUSED leg (exit 2: a refusal is not a measurement).
$bad = @($rolled | Where-Object { $_.exit -ne 0 -and $_.exit -ne 3 })
if ($bad.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0}/{1} legs not usable: {2})" -f `
        $bad.Count, $rolled.Count, (($bad | ForEach-Object { $_.tag + '/' + $_.exit }) -join ', '))
    Write-Host '  A refused leg or a leg that failed its integrity gates is NOT a measurement: report it as unusable, never as a band result.'
    exit 1
}
if ($provBad.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0} step(s) produced legs with no attributable provenance: {1})" -f `
        $provBad.Count, ($provBad -join ', '))
    Write-Host '  The legs may still read PASS; what failed is that their evidence cannot be tied to a revision.'
    Write-Host '  Escape (explicit, recorded): re-run with -AllowUnprovenanced.'
    exit 10
}
$badDrivers = @($driverRows | Where-Object { $_.exit_code -ne 0 })
if ($badDrivers.Count -gt 0) {
    Write-Host ("SESSION VERDICT: FAIL ({0} arm driver(s) exited {1}: see the driver stdout/stderr listed above)" -f `
        $badDrivers.Count, (($badDrivers | ForEach-Object { $_.exit_code }) -join '/'))
    exit 1
}
Write-Host ("SESSION VERDICT: PASS ({0} leg record(s) written; the measured bands are the driver's own RESULT lines above)" -f $rolled.Count)
exit 0
