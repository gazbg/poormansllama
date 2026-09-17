# T-GPLR-36 follow-up probe (CPU-only, no GPU work of its own): the ledger run
# reported peak VRAM 3.19 GiB, but the owner observed the process climbing from
# ~19 GB to ~30 GB during the run. There is no instrument for host RSS in the
# telemetry, so this samples it alongside the GPU counters and writes a CSV.
#
# Usage (while a layer-streaming arm runs in another window):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools/monitor-mem-gplr36.ps1
#   powershell ... -File tools/monitor-mem-gplr36.ps1 -Minutes 15 -IntervalSeconds 5
## Output: build/llama-cuda/gplr36-mem-monitor.csv with one row per sample:
#   t_s,proc_ws_mib,proc_private_mib,sys_free_mib,gpu_dedicated_mib,gpu_shared_mib,gpu_util_pct
# proc_ws/private = the runner's Working Set / Private Bytes (Task Manager's two
# "memory" numbers); sys_free = free physical RAM (the pressure signal pinned
# pages compete against); gpu_shared = the shared (host) GPU memory WDDM
# attributes to the GPU, which is where a host-spilled KV and pinned staging
# show up.
param(
  [string]$ProcessName = 'llama-layer-streaming',
  [string]$Out = 'build/llama-cuda/gplr36-mem-monitor.csv',
  [int]$IntervalSeconds = 5
)

$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$out  = Join-Path $root $Out
"t_s,proc_ws_mib,proc_private_mib,sys_free_mib,gpu_dedicated_mib,gpu_shared_mib,gpu_util_pct" | Set-Content $out

Write-Host ("[mem-monitor] waiting for process '$ProcessName'... (Ctrl+C to stop)")
$t0 = $null
while ($true) {
  if (-not $t0) {
    $p0 = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p0) { Start-Sleep -Seconds 2; continue }
    $t0 = Get-Date            # clock starts when the FIRST arm appears, so the
                              # whole measured window is covered (mutex waits,
                              # all arms, teardown) -- T-GPLR-37c fix: the old
                              # fixed-Minutes loop started before the mutex wait
                              # and missed the arms entirely.
    Write-Host ("[mem-monitor] process appeared, sampling...")
  }
  $t = [int]((Get-Date) - $t0).TotalSeconds
  $p = Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Select-Object -First 1
  $ws = 0; $priv = 0
  if ($p) {
    $ws   = [int]($p.WorkingSet64 / 1MB)
    $priv = [int]($p.PrivateMemorySize64 / 1MB)
  } else {
    # arm boundary: sample the gap so the CSV shows the teardown, then keep
    # watching for the next arm until a full minute passes with no process.
    $gap = (Get-Process -Name $ProcessName -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($gap -eq 0) {
      if (-not $script:lastSeen) { $script:lastSeen = Get-Date }
      if (((Get-Date) - $script:lastSeen).TotalSeconds -gt 60) { break }
    } else { $script:lastSeen = Get-Date }
  }
  $os = Get-CimInstance Win32_OperatingSystem
  $sysfree = [int]($os.FreePhysicalMemory / 1KB)
  $gpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits 2>$null
  $ded = 0; $util = 0
  if ($gpu) {
    $parts = ($gpu -split ',')
    $ded = [int]$parts[0].Trim(); $util = [int]$parts[1].Trim()
  }
  # WDDM shared GPU memory = the GPU's own view of host memory it can touch.
  $shared = 0
  $cnt = Get-Counter '\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
  if ($cnt) {
    $shared = [int]((($cnt.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum) / 1MB)
  }
  Add-Content $out ("{0},{1},{2},{3},{4},{5},{6}" -f $t, $ws, $priv, $sysfree, $ded, $shared, $util)
  Start-Sleep -Seconds $IntervalSeconds
}
Write-Host ("[mem-monitor] done, wrote {0}" -f $out)
Get-Content $out | Select-Object -First 3
Get-Content $out | Select-Object -Last 3
