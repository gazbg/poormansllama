# tools/gpu-mem-poll.ps1
#
# GPU+host memory poller for layer-residency validation.
#
# Samples, at a fixed interval, per GPU adapter present:
#   - dedicated (local) segment usage   ~ VRAM actually resident on the card
#   - shared (non-local) segment usage  ~ WDDM system-memory segment
#   - per-process dedicated usage       ~ attribution of the dedicated usage
# and host-wide:
#   - available physical RAM
#   - system commit (bytes / limit)
#
# Output: CSV, one row per sample, header row first. Incremental write: every
# sample batch is flushed immediately so a force-killed poller still leaves a
# complete CSV. Read-only: starts no processes, touches no locks, initializes
# no CUDA.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\gpu-mem-poll.ps1 `
#     -Out build\gpu-mem-poll-last.csv -IntervalMs 500 -DurationSeconds 120
#
# Defaults: 1000 ms interval, 300 s duration, build\gpu-mem-poll-last.csv.

param(
    [string] $Out = 'build\gpu-mem-poll-last.csv',
    [int] $IntervalMs = 1000,
    [int] $DurationSeconds = 300
)

if ($IntervalMs -lt 50)  { throw 'IntervalMs must be >= 50.' }
if ($DurationSeconds -lt 1) { throw 'DurationSeconds must be >= 1.' }

$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}

$ErrorActionPreference = 'Continue'

# Resolve -Out to an absolute path up front (relative resolves against the
# caller's cwd, which Start-Process children inherit).
$Out = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Out)

$writer = [System.IO.StreamWriter]::new($Out, $false, [System.Text.Encoding]::UTF8)
$writer.AutoFlush = $true
$writer.WriteLine('wrapper=gpu-mem-poll.ps1,interval_ms=' + $IntervalMs + ',duration_s=' + $DurationSeconds)

$line = [System.Collections.Generic.List[string]]::new()

function Add-Row([string] $row) {
    $script:line.Add($row)
}

function Flush-Line {
    if ($script:line.Count -gt 0) {
        foreach ($l in $script:line) { $script:writer.WriteLine($l) }
        $script:line.Clear()
    }
}

$deadline = (Get-Date).AddSeconds($DurationSeconds)
$sw = [System.Diagnostics.Stopwatch]::StartNew()

while ((Get-Date) -lt $deadline) {
    $t = $sw.Elapsed.TotalSeconds

    try {
        $adapters = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory -ErrorAction Stop
        foreach ($a in $adapters) {
            # Name looks like: luid_0x00000000_0x00015277_phys_0[_part_N]
            Add-Row ('gpu_adapter,t={0:F3},name={1},dedicated_bytes={2},shared_bytes={3},total_committed_bytes={4}' -f `
                $t, $a.Name, $a.DedicatedUsage, $a.SharedUsage, $a.TotalCommitted)
        }
    } catch {
        Add-Row ('gpu_adapter,t={0:F3},error={1}' -f $t, (($_.Exception.Message) -replace ',', ';'))
    }

    try {
        $procs = Get-CimInstance -ClassName Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory -ErrorAction Stop
        foreach ($p in $procs) {
            if ($p.DedicatedUsage -gt 0 -or $p.NonLocalUsage -gt 0) {
                Add-Row ('gpu_proc,t={0:F3},name={1},pid={2},dedicated_bytes={3},nonlocal_bytes={4}' -f `
                    $t, (($p.Name) -replace ',', ';'), $p.IDProcess, $p.DedicatedUsage, $p.NonLocalUsage)
            }
        }
    } catch {
        Add-Row ('gpu_proc,t={0:F3},error={1}' -f $t, (($_.Exception.Message) -replace ',', ';'))
    }

    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        Add-Row ('host,t={0:F3},avail_phys_bytes={1},total_phys_bytes={2}' -f `
            $t, ([uint64]$os.FreePhysicalMemory * 1024), ([uint64]$os.TotalVisibleMemorySize * 1024))
    } catch {
        Add-Row ('host,t={0:F3},error={1}' -f $t, (($_.Exception.Message) -replace ',', ';'))
    }

    try {
        $c = Get-Counter '\Memory\Committed Bytes','\Memory\Commit Limit' -ErrorAction Stop
        $commit = [uint64]($c.CounterSamples | Where-Object { $_.Path -like '*committed bytes' } | Select-Object -ExpandProperty CookedValue)
        $limit  = [uint64]($c.CounterSamples | Where-Object { $_.Path -like '*commit limit' }   | Select-Object -ExpandProperty CookedValue)
        Add-Row ('host_commit,t={0:F3},committed_bytes={1},commit_limit_bytes={2}' -f $t, $commit, $limit)
    } catch {
        Add-Row ('host_commit,t={0:F3},error={1}' -f $t, (($_.Exception.Message) -replace ',', ';'))
    }

    Flush-Line
    Start-Sleep -Milliseconds $IntervalMs
}

$writer.Flush()
$writer.Dispose()
Write-Host ("wrote {0}" -f $Out)
