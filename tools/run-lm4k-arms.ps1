# Relocated from build/run-lm4k-arms.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# The three 4k arms, executed directly (no string composition). Invoked by
# file path from run-lm4k-trio.ps1 under one GPU-mutex hold.
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

foreach ($arm in @(
  @{ name = 'bm1'; sched = 'bm' },
  @{ name = 'lm';  sched = 'lm' },
  @{ name = 'bm2'; sched = 'bm' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr25-lm4k-$name.jsonl"
  $log   = Join-Path $base "gplr25-lm4k-$name.log"
  $tg    = Join-Path $base "gplr25-lm4k-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--telemetry', $trace, '--tg-text', $tg)
  if ($arm.sched -eq 'bm') { $a += @('--pp-schedule', 'batch-major', '--no-handoff') }
  & $exe @a > $log 2>&1
  "ARM $name EXIT: $LASTEXITCODE"
}
