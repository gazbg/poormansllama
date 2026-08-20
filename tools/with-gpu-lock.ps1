[CmdletBinding()]
param(
    [int] $TimeoutSeconds = 7200,
    [Parameter(Mandatory = $true)]
    [string] $Command,
    [string] $MutexName = 'Global\AI-GPU-EXCLUSIVE-v1'
)

if ($TimeoutSeconds -lt 0) {
    throw 'TimeoutSeconds must be zero or greater.'
}

if ([string]::IsNullOrWhiteSpace($Command)) {
    throw 'Command must not be empty.'
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
}

exit $exitCode