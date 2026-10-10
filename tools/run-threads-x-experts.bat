@echo off
REM Cross-product sweep: thread count x expert count, host-only, no GPU, no mutex.
REM
REM Gap this closes: sec 12.1 swept threads at k=8 (14.2 MB working set) and
REM sec 12.1a swept expert count at 8 threads.  The overlapped split runs the
REM host at (1-h)*8 = 2-3 experts, i.e. a 4 MB working set, where the 8-thread
REM saturation point was never verified.  The host leg is the binding leg below
REM h* and sets h* itself, so its thread optimum in the OPERATING regime is a
REM real input, not a detail.
REM
REM iters raised 300 -> 1000: audit-C's 1-expert cell had min 20.4 / median
REM 33.1 / p90 47.7, a +/-44%% spread, too noisy to read a thread optimum from.
setlocal
set BENCH=%~dp0..\build\expert-cpu-bench\expert-cpu-bench.exe
set MODEL=C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf
set OUT=%~dp0..\build\expert-cpu-bench\threads-x-experts

"%BENCH%" --model "%MODEL%" --layer 0 --pool 256 --id-stride 8 ^
  --iters 1000 --threads 1,2,4,6,8,12,16,32 --experts 1,2,3,4,8 ^
  --json "%OUT%.json" > "%OUT%.log" 2> "%OUT%.err"

echo exit=%ERRORLEVEL%
endlocal
