# Relocated from build/run-gplr29-linkstate.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 addendum: read PCIe link gen/width WHILE copies are in flight
# (idle GPUs downtrain; the earlier query ran after the probe finished).
$exe   = (Join-Path $root 'build/gplr29-pinprobe.exe')
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'

$p = Start-Process -FilePath $exe -ArgumentList @($model, '1073741824', '453000000', '80') -NoNewWindow -PassThru
Start-Sleep -Milliseconds 1200
for ($i = 0; $i -lt 12; ++$i) {
  $q = nvidia-smi --query-gpu=utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>$null
  "load-sample $i : $q"
  Start-Sleep -Milliseconds 400
}
Wait-Process -Id $p.Id
"PROBE EXIT: $($p.ExitCode)"
