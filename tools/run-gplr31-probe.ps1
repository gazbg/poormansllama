# Relocated from build/run-gplr31-probe.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-31: locked-RAM ceiling probe under the GPU mutex.
$lock = (Join-Path $root 'tools/with-gpu-lock.ps1')
$exe  = (Join-Path $root 'build/gplr31-pinpool-probe.exe')
$log  = (Join-Path $root 'build/llama-cuda/gplr31-pinpool-probe.log')
& $lock -TimeoutSeconds 420 -Command ('"' + $exe + '" > "' + $log + '" 2>&1')
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
if (Test-Path $log) {
    Get-Content $log
}
