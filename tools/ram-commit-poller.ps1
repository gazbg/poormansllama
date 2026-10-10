# T-GPLR-74: background commit/RAM poller.
#
# WHY: the 2026-10-02 K=2 arms drove Windows COMMITTED BYTES to ~71 GB
# transiently, on BOTH --copy-source mmap-registered (40 cudaHostRegister
# calls, 18,500,517,888 B pinned) and --copy-source direct (ZERO mmap_register
# events, host_input only 1,140,850,688 B). The first theory -- that
# cudaHostRegister page-locking explained it -- was FALSIFIED by the direct
# run. Committed bytes returned to 12.3 GiB afterwards, so nothing leaks; but
# no field in any trace explains ~60 GB of transient commit. This poller
# exists so the next run answers that by sampling instead of inference.
#
# It samples COMMITTED BYTES (the quantity the owner observed), not Working
# Set: commit counts RAM + pagefile backing, which is what a Task Manager
# "Committed" column shows and is NOT visible in any llama-layer-streaming
# telemetry field. Available Bytes and both kernel pools are recorded too, so
# a nonpaged-pool leak is distinguishable from a genuine allocation.
#
# USAGE -- start BEFORE launching the arm, stop after the arm exits:
#   powershell -NoProfile -File tools/ram-commit-poller.ps1
#   powershell -NoProfile -File tools/ram-commit-poller.ps1 -Out ram.csv
#
# Correlate by timestamp against the arm's telemetry:
#   run_start / phase_summary / run_end carry t_us on the same wall clock.
param(
    [string] $Out = '',
    [int]    $IntervalSeconds = 1,
    [int]    $TimeoutSeconds = 7200
)

# NOT 'Stop': PowerShell 5 treats a native command writing to stderr as a
# terminating error, and Get-Counter raises on a missing counter path.
$ErrorActionPreference = 'Continue'

if ([string]::IsNullOrWhiteSpace($Out)) {
    $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
    $outDir = Join-Path (Join-Path $PSScriptRoot '..\build\llama-cuda\logs\ram')
    $Out = Join-Path $outDir "ram-$stamp.csv"
}
$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

# Persistent PerformanceCounter objects, NOT Get-Counter. Measured: even a
# SINGLE multi-path Get-Counter call costs ~6 s on this host (samples landed
# ~6 s apart regardless of -IntervalSeconds), which is far too coarse to say
# which phase of an arm allocates. Constructed once and polled with
# .NextValue() the same counters cost milliseconds.
$counters = @{}
foreach ($spec in @(
    @('committed_bytes', 'Committed Bytes'),
    @('commit_limit',   'Commit Limit'),
    @('available',      'Available Bytes'),
    @('nonpaged',       'Pool Nonpaged Bytes'),
    @('paged',          'Pool Paged Bytes')
)) {
    try {
        $counters[$spec[0]] = New-Object System.Diagnostics.PerformanceCounter(
            'Memory', $spec[1], $null, [System.TimeSpan]::FromSeconds(1))
        # A rate counter's first NextValue() is meaningless; prime it.
        [void]$counters[$spec[0]].NextValue()
        [void]$counters[$spec[0]].NextValue()
    } catch {
        Write-Host ("  counter unavailable: " + $spec[1])
    }
}

if (-not (Test-Path $Out)) {
    'utc,committed_bytes,commit_limit_bytes,available_bytes,pool_nonpaged_bytes,pool_paged_bytes,free_physical_bytes,total_physical_bytes' |
        Set-Content -Path $Out
}

Write-Host ("RAM poller -> " + $Out)
Write-Host ("  every {0}s, timeout {1}s; committed bytes is the owner's 'Committed' figure" -f $IntervalSeconds, $TimeoutSeconds)

$started = Get-Date
while (((Get-Date) - $started).TotalSeconds -lt $TimeoutSeconds) {
    $now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
    if (-not $counters.ContainsKey('committed_bytes')) { Start-Sleep -Seconds 5; continue }
    $committed = [double]$counters['committed_bytes'].NextValue()
    $limit     = [double]$counters['commit_limit'].NextValue()
    $avail     = [double]$counters['available'].NextValue()
    $nonpaged  = [double]$counters['nonpaged'].NextValue()
    $paged     = [double]$counters['paged'].NextValue()
    $totalPhys = $limit
    $freePhys  = $avail

    Add-Content -Path $Out -Value ("{0},{1},{2},{3},{4},{5},{6},{7}" -f `
        $now, [int64]$committed, [int64]$limit, [int64]$avail, `
        [int64]$nonpaged, [int64]$paged, [int64]$freePhys, [int64]$totalPhys)

    Start-Sleep -Seconds $IntervalSeconds
}

Write-Host "RAM poller finished (timeout)."