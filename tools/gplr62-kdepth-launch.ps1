# T-GPLR-62 follow-up: the K=2 / K=4 row-cache depth arms under ONE with-gpu-lock hold.
#
# -AllowBusy is required because ChatGPT.exe (pid 8092) holds a persistent but
# IDLE CUDA context (15 MiB used, 0 % util) and the wrapper's probe refuses
# while any compute process is visible. The wrapper's own verdict records
# gpu_pre_run / gpu_post_run so the contamination caveat travels with the run,
# and the session's K0b arm brackets drift against the K0 reference.
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe'
$cmd = "python " + (Join-Path $root 'tools/run-gplr62-kdepth.py') + " " + $exe
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'with-gpu-lock.ps1') `
    -TimeoutSeconds 3600 -AllowBusy -Command $cmd -VerdictTag 'gplr62-kdepth'
exit $LASTEXITCODE
