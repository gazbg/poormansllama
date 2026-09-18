# Relocated from build/run-gplr25-ab.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Stop'
$exe   = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$lock  = (Join-Path $root 'tools/with-gpu-lock.ps1')
$chk   = (Join-Path $root 'tools/semantic-qa/check-answer.py')
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$base  = (Join-Path $root 'build/llama-cuda')

# T-GPLR-25 paired A/B: af=0 vs af=1 at FIXED --layer-slots 2. Within each
# length the two arms run back-to-back under one mutex hold so they share the
# file-cache state as closely as the machine allows; af is the only flag delta.
$arms = @(
    @{ len = 512;  af = 0; tag = 'af0' },
    @{ len = 512;  af = 1; tag = 'af1' },
    @{ len = 4096; af = 0; tag = 'af0' },
    @{ len = 4096; af = 1; tag = 'af1' }
)

foreach ($arm in $arms) {
    $len = $arm.len; $af = $arm.af; $tag = $arm.tag
    $prompt = Join-Path $base "gplr07-qa-prompts\anchor-$len.txt"
    $trace  = Join-Path $base "gplr25-ab-$len-$tag.jsonl"
    $tgtext = Join-Path $base "gplr25-ab-$len-$tag.tgtext"
    $run = "`"$exe`" --model `"$model`" --prompt-file `"$prompt`" --ubatch 512 --tg-tokens 96 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --no-handoff --layer-slots 2 --pp-async-feed $af --allow-full-residency --telemetry `"$trace`" --tg-text `"$tgtext`""
    Write-Host "[AB] len=$len af=$af : runner under GPU mutex..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock -TimeoutSeconds 1200 -Command $run
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: len=$len af=$af runner exit $LASTEXITCODE" -ForegroundColor Red; exit 1 }
    & $exe --check-telemetry $trace
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: checker rejected $trace" -ForegroundColor Red; exit 1 }
    & python $chk check --tg-text $tgtext --items N1,N2,N3,N4,K1 | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: answer gate len=$len af=$af" -ForegroundColor Red; exit 1 }
    Write-Host "[AB] len=$len af=$af : run + checker + answer gate PASS" -ForegroundColor Green
}
Write-Host "SUCCESS: all four A/B arms completed."
