# T-GPLR-33 A/B arms: member-batched staging (one H2D per bank when the
# slot-build contiguity check tiled the members exactly) vs the legacy
# per-member submissions. Two arms (batch0 baseline vs batch1), identical
# gplr32 shape (4k anchor, ubatch 512, 2 slots, layer-major default,
# handoff auto, kv q8, --pp-repeat 3, --copy-source pinned, --pp-async-feed
# 1); only --pp-submit-batch differs (0 vs 1). Strict --check-telemetry per
# arm. Invoke under the GPU mutex via run-gplr33.ps1.
# Pre-registered predictions (plan T-GPLR-33): mean effective window
# 91 -> ~44-58 ms, transfer/warm pass 3.65 -> ~2.4 s, warm wall 2.77 ->
# ~1.8-2.0 s. FALSIFIER: window >= ~80 ms (toll is per-operation-bytes).
# Acceptance: submit_batch field = intended path on >= 118/120 banks.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'batch0'; sb = '0' },
  @{ name = 'batch1'; sb = '1' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr33-$name.jsonl"
  $log   = Join-Path $base "gplr33-$name.log"
  $tg    = Join-Path $base "gplr33-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--pp-repeat', '3', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-submit-batch', $arm.sb,
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr33] arm $name starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 15 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr33] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
}
