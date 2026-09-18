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
    [switch] $NoVerdict
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
        & cmd.exe /d /s /c $Command
        $exitCode = if ($null -eq $LASTEXITCODE) { 0 } else { $LASTEXITCODE }
    }
} finally {
    if ($acquired) {
        $mutex.ReleaseMutex()
        Write-Host "Released GPU mutex $MutexName"
    }

    $mutex.Dispose()

    if ($verdictReady) {
        $v = Write-ArmVerdict -Dir $vDir -Tag $vTag -ExitCode $exitCode -StartedUtc $vStarted `
            -Command $Command -MutexName $MutexName `
            -Fields @{ timeout_seconds = $TimeoutSeconds; mutex_acquired = $acquired }
        Write-Host ("Verdict: {0} ({1}) -> {2}" -f $v.status, $exitCode, (Join-Path $vDir ($vTag + '.done')))
    }
}

exit $exitCode