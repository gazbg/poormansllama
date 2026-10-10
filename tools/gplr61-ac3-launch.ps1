# T-GPLR-61 AC3 follow-up: paired session under ONE with-gpu-lock hold.
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe'
$cmd = "python " + (Join-Path $root 'tools/run-gplr61-ac3.py') + " " + $exe
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'with-gpu-lock.ps1') -TimeoutSeconds 1800 -Command $cmd -VerdictTag 'gplr61-ac3'
exit $LASTEXITCODE
