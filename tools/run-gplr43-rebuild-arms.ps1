# T-GPLR-43: graph-rebuild-between-passes probe at the 144k quantized-KV
# pass-2 shape. Arms: q8 with --pp-prebuild-per-pass (pass 2 runs FRESH
# graphs) + adjacent rebuilt-off q8 control in the SAME mutex hold.
# B1 shape: 143360 = 140x1024, u1024, 3 slots, mmap-registered, af=1,
# --pp-repeat 2. Failure isolation only.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'q8-rebuild'; rb = '1' },
  @{ name = 'q8-norebuild'; rb = '0' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr43-$name.jsonl"
  $log   = Join-Path $base "gplr43-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', '143360', '--ctx-size', '144128',
         '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1',
         '--pp-prebuild-per-pass', $arm.rb,
         '--telemetry', $trace)
  Write-Host "[gplr43] arm $name (rb=$($arm.rb)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log -Encoding Unicode | Select-String -Pattern 'PP:|PP pass|graphs rebuilt|CUDA error|illegal|prebuild failed' |
    Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
  if (Test-Path $trace) {
    python (Join-Path $root 'tools/analyze-gplr40.py') $trace
  }
}
Write-Host 'DONE'
