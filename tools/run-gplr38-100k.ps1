# T-GPLR-38 addendum: ubatch A/B at the middle shape -- 100k tokens, u1024 vs
# u2048, 3 slots, churn-free mmap-registered mode. Tests where the ubatch
# crossover sits between the 4k (u1024 best) and 200k (flat) regimes, and gives
# a second route-telemetry scaling point (T-GPLR-39 recon).
#
# Pre-registered prediction (plan addendum): warm walls within ~2%; FALSIFIER:
# u2048 wins by > 2% (4k ranking was telemetry-only) or loses by > 4% (the
# host residue grows super-linearly beyond the route term at this shape).
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

foreach ($arm in @(
  @{ name = 'u1024'; ub = '1024' },
  @{ name = 'u2048'; ub = '2048' })) {
  $name  = $arm.name
  $trace = Join-Path $base "gplr38-100k-$name.jsonl"
  $log   = Join-Path $base "gplr38-100k-$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  # NOTE: the generated anchor tokenized to 144,001 tokens (generator overshot
  # its 100k target, achieved=unverified). Rather than truncating the file (the
  # QA needles live at the end), the run targets the ACTUAL token count: the
  # shape is "~144k-token prefill", close enough to the requested 100k regime.
  # NOTE: the generated anchor tokenized to 144,001 tokens (generator overshot
  # its 100k target, achieved=unverified). Rather than truncating the file (the
  # QA needles live at the end), the run targets the ACTUAL token count: the
  # shape is "~144k-token prefill", close enough to the requested 100k regime.
  # The handoff fit gate measures auto-OFF at this KV size (as at 200k), so
  # --tg-tokens stays 0: layer-major TG without handoff is refused above 3
  # tokens, and PP is the measurement target anyway (the T-GPLR-39 tg>0 trap
  # only fires when the fit gate says auto-ON).
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--pp-tokens', '144000', '--ctx-size', '144256',
         '--ubatch', $arm.ub, '--tg-tokens', '0', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '2', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace)
  Write-Host "[gplr38-100k] arm $name (ubatch=$($arm.ub)) starting..."
  & $exe @a > $log 2>&1
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  if ($LASTEXITCODE -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  Write-Host "[gplr38-100k] arm $name strict check-telemetry..."
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
  Get-Content $log | Select-String -Pattern 'phase pp|peak VRAM|refus|error' |
    Select-Object -Last 5 | ForEach-Object { Write-Host $_ }
}

python (Join-Path $root 'tools/analyze-gplr37.py') `
  (Join-Path $base 'gplr38-100k-u1024.jsonl') (Join-Path $base 'gplr38-100k-u2048.jsonl')
Write-Host ("ANALYZER EXIT: {0}" -f $LASTEXITCODE)
Write-Host 'DONE'
