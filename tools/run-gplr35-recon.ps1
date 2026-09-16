# T-GPLR-35 RECON step: attribute the per-bank coupling before any ordering
# change is made. Same shape as run-gplr35-arms.ps1 (4k anchor, ubatch 512,
# 2 slots, layer-major default, handoff auto, kv q8, --copy-source pinned,
# --pp-async-feed 1) but with --pp-recon 1 and --pp-repeat 2 (pass 1 cold,
# pass 2 warm) so the attribution costs ~1/3 of the full A/B.
#
# What the recon decides (pre-registered in the plan, T-GPLR-35 design contract):
#   recon_identity  -> which streams/contexts the run actually staged on
#   recon_attr.lag_ms  = copy_start - compute_ref. NEGATIVE means the DMA began
#     while the compute stream's backlog was still draining (overlap);
#     >= 0 means the copy began at/after that backlog drained (ordered behind
#     compute -- the chaining signature).
#   recon_attr.copy_gpu_ms = the copy's own GPU window (pure ~31.5 ms at 453 MB
#     pinned; a stretched ~65-75 ms window means engine/rate contention, not
#     ordering).
#   ov0 (baseline, copy on the dedicated copy backend) vs ov1 (copy on the
#   compute context's aux stream) separates an ordering cause from a
#   bandwidth cause.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'recon-ov0'; ov = '0' },
  @{ name = 'recon-ov1'; ov = '1' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr35-$name.jsonl"
  $log   = Join-Path $base "gplr35-$name.log"
  $tg    = Join-Path $base "gplr35-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-recon', '1', '--pp-copy-overlap', $arm.ov,
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr35-recon] arm $name starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 20 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr35-recon] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'PP|tok/s|refus|error' | Select-Object -Last 4 | ForEach-Object { Write-Host $_ }
}
