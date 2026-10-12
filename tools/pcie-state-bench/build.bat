@echo off
rem Builds tools\pcie-state-bench\pcie-state-bench.exe (CUDA runtime only; does not touch the tree).
rem Run it under the GPU mutex:   powershell -NoProfile -File tools\with-gpu-lock.ps1 -VerdictTag pcie -Command "tools\pcie-state-bench\pcie-state-bench.exe 0 40"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set HERE=%~dp0
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 ^
  -Xcompiler "/EHsc /Zc:__cplusplus" -o "%HERE%pcie-state-bench.exe" "%HERE%pcie-state-bench.cu" > "%HERE%build.log" 2>&1
echo nvcc exit %ERRORLEVEL%
