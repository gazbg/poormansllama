# Relocated from build/run-gplr29-pinprobe.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 GPU run: the pinned-pool probe under the mutex (tiny GPU footprint,
# ~5 s). Bank-sized region mapped from the real MoE GGUF at offset 1 GiB.
$lock = (Join-Path $root 'tools/with-gpu-lock.ps1')
$exe  = (Join-Path $root 'build/gplr29-pinprobe.exe')
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$log  = (Join-Path $root 'build/llama-cuda/gplr29-pinprobe.log')

$deadline = (Get-Date).AddSeconds(1500)
while ((Get-Date) -lt $deadline) {
  $q = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
  if ($q) {
    $parts = $q -split ','
    if ([int]$parts[0].Trim() -lt 15 -and [int]$parts[1].Trim() -lt 3000) { break }
  }
  Start-Sleep -Seconds 30
}

# the wrapper executes via cmd.exe: invoke powershell by file path, no inline &
$inner = (Join-Path $root 'build/run-gplr29-arm.ps1')
& $lock -Command "powershell -NoProfile -ExecutionPolicy Bypass -File $inner"
"LOCK EXIT: $LASTEXITCODE"

if (Test-Path $log) {
  Get-Content $log
} else { 'NO PROBE LOG' }
