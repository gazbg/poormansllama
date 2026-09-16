# T-GPLR-32 A/B arms: serialize the worker's H2D DMAs to one in flight.
# Two single-invocation arms (ser0 baseline vs ser1 gated), identical
# gplr29-afab shape (4k anchor, ubatch 512, 2 slots, default schedule =
# layer-major, handoff auto, kv q8, --pp-repeat 3, --copy-source pinned,
# --pp-async-feed 1); only --pp-af-serialize differs (0 vs 1). Runs strict
# --check-telemetry per arm. Invoke under the GPU mutex via run-gplr32.ps1.
# Pre-registered predictions (plan T-GPLR-32): ser1 windows ~31-36 ms (vs
# 69.6 stretched), wait/pass -> ~0, warm wall 2.80-2.87 -> ~1.9-2.3 s.
# RESULT (2026-09-16): falsifier fired — walls identical, stretch intrinsic
# to the per-copy window; see docs/currentstate.md and the plan task.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'ser0'; ser = '0' },
  @{ name = 'ser1'; ser = '1' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr32-$name.jsonl"
  $log   = Join-Path $base "gplr32-$name.log"
  $tg    = Join-Path $base "gplr32-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--pp-repeat', '3', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-af-serialize', $arm.ser,
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr32] arm $name starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 15 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr32] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
}
