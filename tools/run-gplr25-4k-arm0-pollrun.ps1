# Relocated from build/run-gplr25-4k-arm0-pollrun.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Stop'
# T-GPLR-25 4k af=0 arm with availability polling: nvidia-smi is advisory only
# (the mutex remains authoritative); when the GPU looks free we invoke
# with-gpu-lock.ps1 with a short acquire timeout. No workload starts while
# another holder owns Global\AI-GPU-EXCLUSIVE-v1.
$deadline = (Get-Date).AddSeconds(480)
$free = $false
while ((Get-Date) -lt $deadline) {
    $line = & nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
    $parts = $line -split ','
    $util = [int]($parts[0].Trim())
    $mem  = [int]($parts[1].Trim())
    if ($util -lt 15 -and $mem -lt 1500) { $free = $true; break }
    Start-Sleep -Seconds 20
}
if (-not $free) { Write-Host 'POLL: GPU still busy after 480 s; not starting workload.'; exit 2 }
Write-Host 'POLL: GPU looks free; acquiring mutex...'
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools/with-gpu-lock.ps1') -TimeoutSeconds 120 -Command '"$root/build/llama-cuda/bin/Release/llama-layer-streaming.exe" --model "C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf" --prompt-file "$root/build/llama-cuda/gplr07-qa-prompts/anchor-4096.txt" --ubatch 512 --tg-tokens 96 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --no-handoff --layer-slots 2 --pp-async-feed 0 --allow-full-residency --telemetry "$root/build/llama-cuda/gplr25-ab-4096-af0.jsonl" --tg-text "$root/build/llama-cuda/gplr25-ab-4096-af0.tgtext"'
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: af0 runner exit $LASTEXITCODE" -ForegroundColor Red; exit 1 }
& (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe') --check-telemetry (Join-Path $root 'build/llama-cuda/gplr25-ab-4096-af0.jsonl')
if ($LASTEXITCODE -ne 0) { Write-Host 'FAILED: checker rejected af0 trace' -ForegroundColor Red; exit 1 }
Write-Host 'AF0 ARM DONE' -ForegroundColor Green
