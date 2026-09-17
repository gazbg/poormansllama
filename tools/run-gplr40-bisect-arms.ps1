# T-GPLR-40 bisect: isolate the pass-2 --pp-repeat crash at long context.
#   B1 'nobreak': --pp-tokens 143360 (140x1024 EXACTLY, no tail chunk)
#   C1 'tail':    --pp-tokens 144000 (141 chunks, 640-token tail) -- adjacency re-run
# Both u1024, 3 slots, ron mode, --pp-repeat 2. Failure isolation only: no
# performance claims. Outcome table is pre-registered in the plan block.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

# Bisect outcome so far: tail EXONERATED (B1 no-tail crashed identically). Refined
# hypothesis from the invariant (pass 2, layer 3, slot 0, gen 16): the surviving 200k
# repeat run used f16 KV, the crashing 144k arms use q8 KV -> probe D1: same 143360
# shape with --kv-cache f16.
foreach ($arm in @(
  @{ name = 'f16-143360'; pp = '143360'; ctx = '144128'; kv = 'f16' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr40-$name.jsonl"
  $log   = Join-Path $base "gplr40-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', $arm.pp, '--ctx-size', $arm.ctx,
         '--ubatch', '1024', '--tg-tokens', '0', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', $arm.kv,
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace)
  Write-Host "[gplr40] arm $name (pp=$($arm.pp) kv=$($arm.kv)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'PP:|PP pass|CUDA error|phase pp|illegal' |
    Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
  if (Test-Path $trace) {
    # where did this arm get to? count passes completed and the last chunk seen
    python (Join-Path $root 'tools/analyze-gplr40.py') $trace
  }
}
Write-Host 'DONE'
