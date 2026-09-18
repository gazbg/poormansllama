# tools/run-memprobe-dense.ps1
#
# T-MEMPROBE-01: live RAM/VRAM validation of one dense 1024-token layer-major
# run on Qwen3.8-27B-UD-Q6_K_XL (the ctest layer-major-1024 shape, ubatch 256).
#
# Measurements:
#   A. runner telemetry run_start/run_end (peak_vram_bytes = before/after
#      free-VRAM delta, tools/layer-streaming/main.cpp:3641)
#   B. tools/gpu-mem-poll.ps1 at 0.5 s: dedicated vs shared adapter memory
#      (Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory) and
#      host commit — no NVML dependency, same segment split as Task Manager.
#
# Sequence: baseline sample -> poller starts (10 s pre-roll, fail-fast if it
# produces no CSV) -> runner under tools/with-gpu-lock.ps1 (synchronous, the
# run-tests.ps1 invocation pattern) -> 30 s teardown capture -> poller
# stopped. Artifacts under build/llama-cuda/memprobe/ (ignored, regenerable;
# rerun this script to recreate them).

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)   # workspace root (script lives in tools/)
$build = Join-Path $root 'build'
$dir = Join-Path $build 'llama-cuda\memprobe'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$runner = Join-Path $build 'llama-cuda\bin\Release\llama-layer-streaming.exe'
if (-not (Test-Path -LiteralPath $runner)) { throw "runner not found: $runner" }
$lock = Join-Path $root 'tools\with-gpu-lock.ps1'
$poll = Join-Path $root 'tools\gpu-mem-poll.ps1'
$model = 'C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf'
if (-not (Test-Path -LiteralPath $model)) { throw "model not found: $model" }

$tel = Join-Path $dir 'memprobe-dense-1024.jsonl'
$csv = Join-Path $dir 'memprobe-dense-1024.csv'
Remove-Item -LiteralPath $tel, $csv -ErrorAction SilentlyContinue

# ---- baseline samples before any CUDA work (recorded, not assumed)
nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader
$base = Get-CimInstance Win32_OperatingSystem
"baseline: avail_phys={0:N0} KB" -f $base.FreePhysicalMemory

# ---- launch the poller (10 s pre-roll before the runner initializes CUDA)
$poller = Start-Process powershell -PassThru -WindowStyle Hidden -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $poll,
    '-Out', $csv, '-IntervalMs', '500', '-DurationSeconds', '240'
)
Write-Host "poller pid=$($poller.Id)"
Start-Sleep -Seconds 10
if (-not (Test-Path -LiteralPath $csv) -or (Get-Item -LiteralPath $csv).Length -eq 0) {
    throw "poller produced no output after 10 s: $csv"
}
Write-Host "poller alive, csv bytes: $((Get-Item -LiteralPath $csv).Length)"

# ---- runner under the GPU mutex (synchronous, proven run-tests.ps1 pattern)
$DirectCmd = "`"$runner`" --model `"$model`" --pp-schedule layer-major --layer-slots 4 " +
    "--pp-tokens 1024 --ubatch 256 --tg-tokens 3 --greedy --no-mtp --pp-no-logits " +
    "--no-handoff --telemetry `"$tel`""
Write-Host "Running direct experiment under GPU mutex: $DirectCmd"
& powershell -NoProfile -ExecutionPolicy Bypass -File $lock `
    -TimeoutSeconds 600 -Command $DirectCmd
$runnerExit = $LASTEXITCODE
Write-Host "runner exit=$runnerExit"

# ---- let the poller capture teardown
Start-Sleep -Seconds 30
if (-not $poller.HasExited) { Stop-Process -Id $poller.Id -Force }
if (-not (Test-Path -LiteralPath $csv) -or (Get-Item -LiteralPath $csv).Length -eq 0) {
    throw "poller CSV missing/empty after run: $csv"
}

Write-Host "artifacts: $tel / $csv"
exit $runnerExit
