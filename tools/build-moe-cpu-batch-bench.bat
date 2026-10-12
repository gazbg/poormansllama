@echo off
rem Builds tools\moe-cpu-batch-bench.exe next to the ggml DLLs of the Release build (run from anywhere). CPU only.
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set HERE=%~dp0
set ROOT=%HERE%..
cl /nologo /O2 /EHsc /std:c++17 /arch:AVX2 "%HERE%moe-cpu-batch-bench.cpp" /I "%ROOT%\third_party\llama.cpp\ggml\include" /Fe:"%ROOT%\build\llama-cuda\bin\Release\moe-cpu-batch-bench.exe" /Fo"%ROOT%\build\llama-cuda\moe-cpu-batch-bench.obj" /link /LIBPATH:"%ROOT%\build\llama-cuda\ggml\src\Release" ggml-base.lib ggml-cpu.lib ggml.lib
echo cl exit %ERRORLEVEL%
