# T-GPLR-43 fix regression: B1 (no-tail q8) / C1 (tail q8) / D1 (no-tail f16)
# with the DEFAULT per-pass rebuild active. Pre-registered acceptance: B1 and
# C1 SURVIVE both passes (they crashed before the fix) with --check-telemetry
# OK; D1 unchanged (survived before, must survive now). Failure isolation +
# fix evidence; no performance claims.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'fix-B1-q8';   pp = '143360'; ctx = '144128'; kv = 'q8' },
  @{ name = 'fix-C1-q8tail'; pp = '144000'; ctx = '144256'; kv = 'q8' },
  @{ name = 'fix-D1-f16';  pp = '143360'; ctx = '144128'; kv = 'f16' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr43-$name.jsonl"
  $log   = Join-Path $base "gplr43-$name.log"
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
  Write-Host "[gplr43fix] arm $name (kv=$($arm.kv) pp=$($arm.pp)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log -Encoding Unicode | Select-String -Pattern 'PP:|PP pass|graphs rebuilt|CUDA error|illegal' |
    Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
  if (Test-Path $trace) {
    python (Join-Path $root 'tools/analyze-gplr40.py') $trace
    # fix acceptance: surviving traces must pass the strict checker
    & $exe --check-telemetry $trace 2>&1 | Select-String -Pattern 'OK|violation|missing|error' |
      Select-Object -First 4 | ForEach-Object { Write-Host "CHECKER: $_" }
  }
}
Write-Host 'DONE'
