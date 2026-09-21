[CmdletBinding()]
param(
    [int] $TimeoutSeconds = 7200,
    [Parameter(Mandatory = $true)]
    [string] $Command,
    [string] $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1',
    # Verdict/sentinel output. Defaulted below to <repo>/build/llama-cuda/verdicts
    # with the tag derived from the command, so every arm run through this wrapper
    # leaves a checkable result without the caller having to name anything.
    [string] $VerdictDir = '',
    [string] $VerdictTag = '',
    [switch] $NoVerdict,
    # T-GPLR-50 contention follow-up: a non-conforming project can hold VRAM and
    # run compute DURING a mutex-held arm (measured 2026-09-21: deepseek_v4.exe,
    # ~16.5 GiB, poisoned a host-suite arm and the recorded 200k remeasure). The
    # wrapper therefore probes nvidia-smi AFTER acquiring the mutex and REFUSES
    # to start the command while any compute process is visible, unless the
    # caller explicitly accepts the risk with -AllowBusy.
    [switch] $AllowBusy,
    # Seconds to wait for a just-exiting process to disappear (a pre-existing
    # compute app often tears down between mutex holders). Two probes, no sleep
    # when the first is already clean.
    [int] $BusyGraceSeconds = 30
)

if ($TimeoutSeconds -lt 0) {
    throw 'TimeoutSeconds must be zero or greater.'
}

if ([string]::IsNullOrWhiteSpace($Command)) {
    throw 'Command must not be empty.'
}

# --- verdict wiring ---------------------------------------------------------
# Written even when the command fails or the mutex times out, and the .done is
# DELETED before the run, so a stale sentinel from an earlier run can never be
# read as this run's result.
$verdictReady = $false
$vDir = ''
$vTag = ''
$vStarted = ''
if (-not $NoVerdict) {
    $root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
    $helper = Join-Path $root 'tools\arm-verdict.ps1'
    if (Test-Path -LiteralPath $helper) {
        . $helper
        $vTag = if ($VerdictTag) { $VerdictTag } else { Get-ArmTagFromCommand -Command $Command }
        $vDir = if ($VerdictDir) { $VerdictDir } else { Join-Path $root 'build\llama-cuda\verdicts' }
        $vStarted = (Get-Date).ToUniversalTime().ToString('o')
        Clear-ArmVerdict -Dir $vDir -Tag $vTag
        # Mark the run STARTED here -- before the mutex wait and before any model
        # load -- so "never started" is distinguishable from "still working" for
        # the whole duration of the run, not merely at the end.
        Write-ArmStart -Dir $vDir -Tag $vTag -Command $Command -MutexName $MutexName | Out-Null
        Write-Host ("Started: {0}" -f (Join-Path $vDir ($vTag + '.started')))
        $verdictReady = $true
    } else {
        Write-Warning "arm-verdict helper not found at $helper; no verdict file will be written."
    }
}

$mutex = [System.Threading.Mutex]::new($false, $MutexName)
$acquired = $false
$exitCode = 1
$gpuPre  = '__unprobed__'
$gpuPost = '__unprobed__'

try {
    Write-Host "Waiting for GPU mutex $MutexName"

    try {
        $acquired = $mutex.WaitOne([TimeSpan]::FromSeconds($TimeoutSeconds))
    } catch [System.Threading.AbandonedMutexException] {
        $acquired = $true
        Write-Warning 'Recovered an abandoned GPU mutex.'
    }

    if (-not $acquired) {
        [Console]::Error.WriteLine("Timed out waiting for GPU mutex after $TimeoutSeconds seconds.")
        $exitCode = 124
    } else {
        Write-Host "Acquired GPU mutex $MutexName"

        # --- GPU busy gate (post-acquisition) ---------------------------------
        # Probe only AFTER the hold is ours: a pre-existing compute app should
        # tear down once its holder loses the mutex, so give it a grace window,
        # then refuse loudly. Unverifiable probe state (no nvidia-smi) does NOT
        # block -- it is recorded and the run proceeds, annotated.
        $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
        if ($gpuPre -eq 'probe-unavailable') {
            Write-Warning 'GPU busy probe unavailable (nvidia-smi?); proceeding UNVERIFIED.'
        } else {
            Write-Host ("GPU pre-run probe: {0}" -f $gpuPre)
            if ($gpuPre -ne 'clean') {
                if ($BusyGraceSeconds -gt 0) {
                    Write-Host ("  busy; grace {0}s for a just-exiting holder..." -f $BusyGraceSeconds)
                    $deadline = (Get-Date).AddSeconds($BusyGraceSeconds)
                    while ((Get-Date) -lt $deadline) {
                        Start-Sleep -Seconds ([Math]::Min(5, [Math]::Max(1, $BusyGraceSeconds)))
                        $gpuPre = Format-GpuBusy (Get-GpuComputeApps)
                        if ($gpuPre -eq 'clean') { break }
                    }
                    Write-Host ("GPU pre-run probe after grace: {0}" -f $gpuPre)
                }
            }
            if ($gpuPre -ne 'clean' -and -not $AllowBusy) {
                [Console]::Error.WriteLine(("REFUSING to run: GPU busy with a non-conforming compute process ({0}). " -f $gpuPre) +
                    'Rerun with -AllowBusy to accept the contamination risk.')
                $exitCode = 125
            }
        }

        if ($exitCode -ne 125) {
            if ($AllowBusy -and $gpuPre -ne 'clean' -and $gpuPre -ne 'probe-unavailable') {
                Write-Warning ("-AllowBusy given: running DESPITE a busy GPU ({0}). Any timing from this run is suspect." -f $gpuPre)
            }
            & cmd.exe /d /s /c $Command
            $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }

            # --- GPU busy post-run probe --------------------------------------
            # A conforming process that ARRIVED mid-run still poisons the run;
            # catch it on the way out so the verdict says so.
            $gpuPost = Format-GpuBusy (Get-GpuComputeApps)
            if ($gpuPost -ne 'clean') {
                Write-Warning ("GPU post-run probe: {0} - a non-conforming compute process may have run alongside this arm; timing suspect." -f $gpuPost)
            }
        }
    }
} finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
        Write-Host "Released GPU mutex $MutexName"
    }

    $mutex.Dispose()

    if ($verdictReady) {
        $fields = @{ timeout_seconds = $TimeoutSeconds; mutex_acquired = $acquired
                     gpu_pre_run = $gpuPre; gpu_post_run = $gpuPost }
        if ($exitCode -eq 125) { $fields['refused_busy'] = $gpuPre }
        $v = Write-ArmVerdict -Dir $vDir -Tag $vTag -ExitCode $exitCode -StartedUtc $vStarted `
            -Command $Command -MutexName $MutexName -Fields $fields
        Write-Host ("Verdict: {0} ({1}) gpu_pre={2} gpu_post={3} -> {4}" -f `
            $v.status, $exitCode, $gpuPre, $gpuPost, (Join-Path $vDir ($vTag + '.done')))
    }
}

exit $exitCode