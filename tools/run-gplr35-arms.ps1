# T-GPLR-35 A/B arms: staging DMAs on a second stream INSIDE the compute
# backend's CUDA context (--pp-copy-overlap 1) vs the dedicated copy backend's
# own stream (0, the recorded baseline). Two arms, identical gplr32/33 shape
# (4k anchor, ubatch 512, 2 slots, default schedule, handoff auto, kv q8,
# --pp-repeat 3, --copy-source pinned, --pp-async-feed 1); only
# --pp-copy-overlap differs. Strict --check-telemetry per arm. Invoke under the
# GPU mutex via run-gplr35.ps1.
#
# Pre-registered predictions (plan T-GPLR-35): warm 4k per-pass wall
# 2.75-2.8 -> ~2.0-2.1 s (ceiling ~1.5-1.6 s if the overlap is perfect),
# effective copy window median ~75 -> ~35-40 ms, per-pass wait collapses,
# PP tok/s ~1,480 -> ~1,900-2,050.
# FALSIFIER: warm wall >= ~2.6 s with the copy window unchanged => the
# coupling is not stream/context-level; close the family and escalate.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'overlap0'; ov = '0' },
  @{ name = 'overlap1'; ov = '1' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr35-$name.jsonl"
  $log   = Join-Path $base "gplr35-$name.log"
  $tg    = Join-Path $base "gplr35-$name.tgtext"
  Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', '512', '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--layer-slots', '2', '--allow-full-residency',
         '--pp-repeat', '3', '--copy-source', 'pinned',
         '--pp-async-feed', '1', '--pp-copy-overlap', $arm.ov,
         '--telemetry', $trace, '--tg-text', $tg)
  Write-Host "[gplr35] arm $name starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 20 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr35] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'PP|tok/s|refus|error' | Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
}
