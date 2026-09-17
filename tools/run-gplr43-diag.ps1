# T-GPLR-43 diagnostic wrapper: mutex + post-run analysis (created vs reused
# set decisions, act_bind count, checker verdict).
$ErrorActionPreference = 'Continue'
$root  = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$inner = Join-Path $root 'tools\run-gplr43-diag-inner.ps1'
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
& $lock -TimeoutSeconds 900 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File " + $inner)
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
$trace = Join-Path $root 'build\llama-cuda\gplr43-diag.jsonl'
$exe   = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
if (Test-Path $trace) {
  $ab = (Select-String -Path $trace -Pattern '"event":"act_bind"').Count
  Write-Host ("act_bind events in trace: {0}" -f $ab)
  & $exe --check-telemetry $trace 2>&1 | Select-String -Pattern 'OK|violation' | Select-Object -First 3 | ForEach-Object { Write-Host "CHECKER: $_" }
} else {
  Write-Host "NO TRACE"
}
Write-Host 'DONE'
