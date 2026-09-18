# Relocated from build/run-gplr30-poolcurve.ps1 (2026-09-16 provenance sweep; the T-GPLR-34
# convention: committed runners/probes live under tools/, with paths made
# relative to the repository root so a clean checkout reproduces the run).
# Recorded verdict: docs/currentstate.md + docs/plans/generic-prefill-layer-residency-plan.md.
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ErrorActionPreference = 'Continue'
# T-GPLR-30: the partial persistent pinned pool — savings-curve arms.
# All five arms under ONE mutex hold for cache adjacency. Shape = the proven
# gplr28/29 3-pass 4k config (2 slots, af=1, pinned, T-GPLR-10-legal TG=3),
# plus one new flag per arm: --pp-pool N[,policy].
$exe    = (Join-Path $root 'build/llama-cuda/bin/Release/llama-layer-streaming.exe')
$base   = (Join-Path $root 'build/llama-cuda')
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-4096.txt'

$arms = @(
    @{ tag = 'gplr30-lru4';    pool = '4,lru'   },
    @{ tag = 'gplr30-fixed4';  pool = '4'       },
    @{ tag = 'gplr30-fixed10'; pool = '10'      },
    @{ tag = 'gplr30-fixed20'; pool = '20'      },
    @{ tag = 'gplr30-fixed40'; pool = '40'      }
)

foreach ($arm in $arms) {
    $trace = Join-Path $base ($arm.tag + '.jsonl')
    $log   = Join-Path $base ($arm.tag + '.log')
    $tg    = Join-Path $base ($arm.tag + '.tgtext')
    Remove-Item $trace, $log, $tg -ErrorAction SilentlyContinue
    Write-Host ("=== ARM {0} (pool {1}) ===" -f $arm.tag, $arm.pool)
    $a = @('--model', $model, '--prompt-file', $prompt,
           '--ubatch', '512', '--tg-tokens', '3', '--pp-no-logits',
           '--residency', 'moe-expert-bank', '--kv-cache', 'q8',
           '--layer-slots', '2', '--allow-full-residency',
           '--pp-repeat', '3', '--copy-source', 'pinned',
           '--pp-async-feed', '1',
           '--pp-pool', $arm.pool,
           '--telemetry', $trace, '--tg-text', $tg)
    & $exe @a > $log 2>&1
    Write-Host ("ARM EXIT: {0}" -f $LASTEXITCODE)
    if (Test-Path $log) {
        Select-String -Path $log -Pattern 'phase pp|phase tg|check-telemetry: OK|error|refus' |
            ForEach-Object { $_.Line } | Select-Object -First 8
    }
}
Write-Host '=== ALL ARMS DONE ==='
