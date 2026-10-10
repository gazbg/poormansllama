@echo off
REM Build the HOST-ONLY expert CPU microbenchmark.
REM
REM Deliberately OUTSIDE the vendored CMake: this links ggml + ggml-cpu only,
REM never ggml-cuda, so the probe cannot initialize a CUDA context. It reuses
REM the import libraries from the existing build/llama-cuda tree, so no
REM reconfigure and no nested-repo change is required.
REM
REM Usage (from the repo root, any shell):
REM   tools\build-expert-cpu-bench.bat

setlocal

set VSWHERE=%ProgramFiles(x86)%
if not exist "%VSWHERE%\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    echo [build] vcvars64.bat not found under "%VSWHERE%\Microsoft Visual Studio\18\BuildTools"
    exit /b 2
)
call "%VSWHERE%\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

if not exist build\expert-cpu-bench mkdir build\expert-cpu-bench

cl -nologo -O2 -std:c++17 -EHsc -DNDEBUG -W3 ^
   -I third_party\llama.cpp\ggml/include ^
   tools\expert-cpu-bench.cpp ^
   -Fe:build\expert-cpu-bench\expert-cpu-bench.exe ^
   -Fo:build\expert-cpu-bench\ ^
   -link -LIBPATH:build\llama-cuda\ggml\src\Release ggml.lib ggml-cpu.lib ggml-base.lib

if errorlevel 1 (
    echo [build] FAILED
    exit /b 1
)

REM The DLLs must sit beside the exe or on PATH; copy only the CPU-side ones.
copy /y build\llama-cuda\bin\Release\ggml.dll      build\expert-cpu-bench\ >nul
copy /y build\llama-cuda\bin\Release\ggml-cpu.dll  build\expert-cpu-bench\ >nul
copy /y build\llama-cuda\bin\Release\ggml-base.dll build\expert-cpu-bench\ >nul

echo [build] OK -^> build\expert-cpu-bench\expert-cpu-bench.exe
exit /b 0
