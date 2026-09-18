# Relocated from build/run-gplr29-afab-arms.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-29 addendum: af A/B with pinned staging. Two single-invocation arms
# under ONE mutex hold, identical gplr28/gplr29 shape (4k anchor, ubatch 512,
# 2 slots, default schedule = layer-major, handoff auto, kv q8, --pp-repeat 3,
# --copy-source pinned); only --pp-async-feed differs (0 vs 1).
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

foreach ($arm in @(
  @{ name = 'af0'; af = '0' },
  @{ name = 'af1'; af = '1' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr29-afab-$name.jsonl"
  $log   = Join-Path $base "gplr29-afab-$name.log"
  $tg    = Join-Path $base "gplr29-afab-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--pp-repeat', '3', '--copy-source', 'pinned',
         '--pp-async-feed', $arm.af,
         '--telemetry', $trace, '--tg-text', $tg)
  & $exe @a > $log 2>&1
  "ARM $name EXIT: $LASTEXITCODE"
}
