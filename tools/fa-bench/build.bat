@echo off
rem Builds tools\fa-bench\fa-bench.exe against the vendored ggml-cuda headers (does not touch the tree).
rem Run it under the GPU mutex:   powershell -NoProfile -File tools\with-gpu-lock.ps1 -VerdictTag fa-bench -Command "tools\fa-bench\fa-bench.exe"
rem Needs the MSVC build tools (vcvars64.bat) and CUDA 13.x; sm_120 is the RTX 5090 Laptop GPU of this machine.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set HERE=%~dp0
set TREE=%HERE%..\..\third_party\llama.cpp
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\nvcc.exe" -O2 -std=c++17 -gencode arch=compute_120,code=sm_120 --expt-relaxed-constexpr -lineinfo -Xptxas -v ^
  -I "%TREE%\ggml\src\ggml-cuda" -I "%TREE%\ggml\src" -I "%TREE%\ggml\include" -DNDEBUG -DGGML_CUDA_FA_ALL_QUANTS=0 ^
  -Xcompiler "/EHsc /Zc:__cplusplus" -o "%HERE%fa-bench.exe" "%HERE%fa-bench.cu" > "%HERE%build.log" 2>&1
echo nvcc exit %ERRORLEVEL%
