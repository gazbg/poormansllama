# Relocated from build/run-gplr29-afab.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 af A/B driver: poll GPU, take the mutex once, run both arm bodies
# by file path, then summarize both logs.
$lock = (Join-Path $root 'tools/with-gpu-lock.ps1')
$arms = (Join-Path $root 'build/run-gplr29-afab-arms.ps1')

$deadline = (Get-Date).AddSeconds(1500)
while ((Get-Date) -lt $deadline) {
  $q = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
  if ($q) {
    $parts = $q -split ','
    if ([int]$parts[0].Trim() -lt 15 -and [int]$parts[1].Trim() -lt 3000) { break }
  }
  Start-Sleep -Seconds 30
}

& $lock -Command "powershell -NoProfile -ExecutionPolicy Bypass -File $arms"
"LOCK EXIT: $LASTEXITCODE"

foreach ($name in 'af0','af1') {
  "===== $name ====="
  $log = "$root/build/llama-cuda/gplr29-afab-$name.log"
  if (Test-Path $log) {
    Select-String -Path $log -Pattern 'phase pp|phase tg|PP pass|check-telemetry: OK|error' | ForEach-Object { $_.Line } | Select-Object -First 10
  } else { 'NO LOG' }
}
