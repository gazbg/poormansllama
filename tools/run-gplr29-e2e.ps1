# Relocated from build/run-gplr29-e2e.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 end-to-end: identical shape to the gplr28 3-pass arm (4k anchor,
# ubatch 512, 2 slots, default schedule = layer-major, handoff auto, kv q8,
# af=0) but with --copy-source pinned. Per-pass walls/transfer compare against
# build/llama-cuda/gplr28-repeat-3pass.jsonl (direct).
$lock = (Join-Path $root 'tools/with-gpu-lock.ps1')
$base = (Join-Path $root 'build/llama-cuda')
$arm  = (Join-Path $root 'build/run-gplr29-e2e-arm.ps1')

$deadline = (Get-Date).AddSeconds(1500)
while ((Get-Date) -lt $deadline) {
  $q = nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
  if ($q) {
    $parts = $q -split ','
    if ([int]$parts[0].Trim() -lt 15 -and [int]$parts[1].Trim() -lt 3000) { break }
  }
  Start-Sleep -Seconds 30
}

& $lock -Command "powershell -NoProfile -ExecutionPolicy Bypass -File $arm"
"LOCK EXIT: $LASTEXITCODE"
