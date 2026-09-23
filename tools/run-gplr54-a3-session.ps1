# T-GPLR-54 A3: the pre-registered discriminator session (owner-invoked).
#
# Arms (one mutex hold, adjacent, in this order):
#   1. u512-off     : the neutrality-pair control   (flag absent)
#   2. u512-on      : the neutrality-pair treatment (--tg-logit-margin 96)
#   3. u512r-on     : C4 repeat of the u512 reference arm
#   4. u2048-on     : intermediate shape (expected IDENTICAL, K = 0)
#   5. u4096-on     : the single-chunk arm under test
#   6. u4096r-on    : C4 repeat of the single-chunk arm
#
# Pre-registered decision rule (plan block T-GPLR-54, unchanged):
#   M_band = max margin over the agreeing steps (0..40) of the multi-chunk arms.
#   benign (H1)  iff ID-CHECK passes AND margin_u4096(step 40) <= M_band + 1.0.
#   escalate(H2) iff ID-CHECK fails OR margin_u4096(step 40) >  M_band + 1.0.
#   Neutrality: ids(u512-on) == ids(u512-off) byte-identical, or the session is
#   confounded (re-run, do not interpret).
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

$arms = @(
  @{ name = 'gplr54-a3-u512-off';  ub = '512';  margin = $false },
  @{ name = 'gplr54-a3-u512-on';   ub = '512';  margin = $true  },
  @{ name = 'gplr54-a3-u512r-on';  ub = '512';  margin = $true  },
  @{ name = 'gplr54-a3-u2048-on';  ub = '2048'; margin = $true  },
  @{ name = 'gplr54-a3-u4096-on';  ub = '4096'; margin = $true  },
  @{ name = 'gplr54-a3-u4096r-on'; ub = '4096'; margin = $true  })

foreach ($arm in $arms) {
  $name  = $arm.name
  $trace = Join-Path $base "$name.jsonl"
  $log   = Join-Path $base "$name.log"
  Remove-Item $trace, $log -ErrorAction SilentlyContinue
  $a = @('--model', $model, '--prompt-file', $prompt,
         '--ubatch', $arm.ub, '--tg-tokens', '96', '--pp-no-logits',
         '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
         '--pp-schedule', 'layer-major',
         '--layer-slots', '3', '--allow-full-residency',
         '--pp-repeat', '1', '--copy-source', 'mmap-registered',
         '--pp-async-feed', '1', '--pp-recon', '1',
         '--telemetry', $trace, '--tg-text', (Join-Path $base "$name.tgtext"))
  if ($arm.margin) { $a += @('--tg-logit-margin', '96') }
  Write-Host "[gplr54-a3] arm $name (ubatch=$($arm.ub), margin=$($arm.margin)) starting..."
  & $exe @a > $log 2>&1
  $code = $LASTEXITCODE
  Write-Host ("ARM {0} EXIT: {1}" -f $name, $code)
  if ($code -ne 0) {
    Get-Content $log -Tail 25 | ForEach-Object { Write-Host $_ }
    continue
  }
  & $exe --check-telemetry $trace
  Write-Host ("CHECK {0} EXIT: {1}" -f $name, $LASTEXITCODE)
}

Write-Host "[gplr54-a3] all arms done; run the analyzer next:"
Write-Host "  python tools/analyze-gplr54-a3-verdict.py"
