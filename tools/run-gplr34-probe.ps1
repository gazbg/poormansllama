# T-GPLR-34: build and run the standalone mid-kernel H2D chaining probe under
# the shared-GPU mutex (~1 min of GPU time). Reproduces the arm [1]/[5]/[6]
# numbers recorded in docs/currentstate.md (2026-09-16 T-GPLR-34 entry):
# [1] pure bank copy ~31.4 ms, [5] one-stream kernel+copy additive ~62 ms,
# [6] two-stream copy window pure ~31.4 ms with the kernel unaffected.
#
# -arch=native is REQUIRED on this box (RTX 5090 Laptop, sm_120): the default
# nvcc target's PTX is JIT-rejected at kernel launch ("provided PTX was
# compiled with an unsupported toolchain").
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe'
$src  = Join-Path $root 'tools\gplr34-midkernel-probe.cu'
$exe  = Join-Path $root 'build\gplr34-midkernel-probe.exe'
$log  = Join-Path $root 'build\llama-cuda\gplr34-midkernel-probe.log'
$lock = Join-Path $root 'tools\with-gpu-lock.ps1'

& $nvcc -O2 -arch=native -o $exe $src
if ($LASTEXITCODE -ne 0) { Write-Host ("NVCC FAILED: {0}" -f $LASTEXITCODE); exit 1 }

& $lock -TimeoutSeconds 420 -Command ("`"" + $exe + "`" > `"" + $log + "`" 2>&1")
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
if (Test-Path $log) { Get-Content $log }
