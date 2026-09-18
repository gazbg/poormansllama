# T-GPLR-44: run the four A/B arms back-to-back so the pairs sit in one session,
# which is the only way the WALL comparison is worth anything given the recorded
# ~2.9% session-to-session spread.
#
#   4k  unscoped  (pre-patch binary)  -- tg 96, the long-id identity check
#   4k  scoped    (patched binary)    -- tg 96
#   50k unscoped  (pre-patch binary)  -- tg 3, the performance pair
#   50k scoped    (patched binary)    -- tg 3
#
# Each arm acquires and releases the GPU mutex itself, so each waits on the shared
# GPU independently: a long wait by another project costs a run, never correctness.
# Run this detached; completion is signalled by the four `.done` sentinels, not by
# watching this output.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr44-ab-50k.ps1'
$pre  = Join-Path $root 'build\llama-cuda\bin\unscoped-bin\llama-layer-streaming.exe'
$post = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'

foreach ($p in @($arm, $pre, $post)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "MISSING: $p"; exit 1 }
}
Write-Host ("[pair] pre-patch  exe sha256[:16]={0}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $pre).Hash.Substring(0,16))
Write-Host ("[pair] post-patch exe sha256[:16]={0}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $post).Hash.Substring(0,16))

$runs = @(
    @{ Label = 'unscoped'; Exe = $pre;  Tokens = 4096;  Tg = 96 },
    @{ Label = 'scoped';   Exe = $post; Tokens = 4096;  Tg = 96 },
    @{ Label = 'unscoped'; Exe = $pre;  Tokens = 50000; Tg = 3  },
    @{ Label = 'scoped';   Exe = $post; Tokens = 50000; Tg = 3  }
)

foreach ($r in $runs) {
    Write-Host ("[pair] ===== {0} tokens {1} tg {2} =====" -f $r.Label, $r.Tokens, $r.Tg)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Label $r.Label -Exe $r.Exe -PromptTokens $r.Tokens -TgTokens $r.Tg
    Write-Host ("[pair] arm exit: {0}" -f $LASTEXITCODE)
}
Write-Host '[pair] all four arms finished'
