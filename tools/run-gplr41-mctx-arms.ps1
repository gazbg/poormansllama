# T-GPLR-41: run the crashing shape (B1 q8) and the f16 control (D1) with the
# pass-boundary memory-lifecycle recon enabled. Failure isolation only -- no
# performance claims. One mutex hold, two arms, ~15 min.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'q8-mctx';  pp = '143360'; ctx = '144128'; kv = 'q8' },
  @{ name = 'f16-mctx'; pp = '143360'; ctx = '144128'; kv = 'f16' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr41-$name.jsonl"
  $log   = Join-Path $base "gplr41-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', $arm.pp, '--ctx-size', $arm.ctx,
         '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', $arm.kv,
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1', '--pp-mctx-recon', '1',
         '--telemetry', $trace)
  Write-Host "[gplr41] arm $name (kv=$($arm.kv)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log -Encoding Unicode | Select-String -Pattern 'PP:|PP pass|CUDA error|illegal' |
    Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
  if (Test-Path $trace) {
    python (Join-Path $root 'tools/analyze-gplr41.py') $trace
  }
}
Write-Host 'DONE'
