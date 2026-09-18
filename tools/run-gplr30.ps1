# Relocated from build/run-gplr30.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-30 outer wrapper: GPU mutex + the five-arm pool-curve run.
$inner = (Join-Path $root 'build/run-gplr30-poolcurve.ps1')
& (Join-Path $root 'tools/with-gpu-lock.ps1') -TimeoutSeconds 900 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
