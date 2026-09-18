# T-GPLR-13: end-to-end row-set invocability probe.
# One small MoE run with --residency-tg under the shared-GPU mutex:
# the row-set path must run PP, activate staging, stage rows per TG window,
# and emit a checker-clean trace.
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe  = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$lock = Join-Path $root 'tools\with-gpu-lock.ps1'
$model = $env:Q38LS_MODEL_PATH
if (-not $model) { $model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }

$outDir = Join-Path $root 'build\llama-cuda\gplr13'
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$tel = Join-Path $outDir 'rowset-pp64-tg3.jsonl'
if (Test-Path $tel) { Remove-Item $tel -Force }

$cmd = "`"$exe`" --model `"$model`" --residency moe-expert-bank --residency-tg " +
       "--pp-tokens 64 --ubatch 64 --tg-tokens 3 --layer-slots 4 " +
       "--allow-full-residency --telemetry `"$tel`""
Write-Host "Command: $cmd"
& powershell -NoProfile -ExecutionPolicy Bypass -File $lock -Command $cmd
$runExit = $LASTEXITCODE
Write-Host "run exit: $runExit"

if ($runExit -eq 0) {
    Write-Host "--- checker on the trace ---"
    $checkCmd = "`"$exe`" --check-telemetry `"$tel`""
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock -Command $checkCmd
    Write-Host "check exit: $LASTEXITCODE"
}
