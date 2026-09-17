# T-GPLR-42: q4-KV discriminator at the 144k pass-2 shape, plus an adjacent
# q8 confirm arm in the SAME mutex hold (a q4 survival must not be attributable
# to environment drift). B1 shape: 143360 = 140x1024, u1024, 3 slots,
# mmap-registered, af=1, --pp-repeat 2, no mctx recon. Failure isolation only.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'q4-disc';   pp = '143360'; ctx = '144128'; kv = 'q4' },
  @{ name = 'q8-confirm'; pp = '143360'; ctx = '144128'; kv = 'q8' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr42-$name.jsonl"
  $log   = Join-Path $base "gplr42-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', $arm.pp, '--ctx-size', $arm.ctx,
         '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', $arm.kv,
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1',
         '--telemetry', $trace)
  Write-Host "[gplr42] arm $name (kv=$($arm.kv)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log -Encoding Unicode | Select-String -Pattern 'PP:|PP pass|CUDA error|illegal' |
    Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
  if (Test-Path $trace) {
    python (Join-Path $root 'tools/analyze-gplr40.py') $trace
  }
}
Write-Host 'DONE'
