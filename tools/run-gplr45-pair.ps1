# T-GPLR-45: run the four identity/performance arms back-to-back so the pairs sit
# in one session (the recorded ~2.9% session-spread lesson makes cross-session
# patch_us comparisons worthless).
#
#   4k control (pre-T-GPLR-45 binary) -- tg 96, the long-id identity check
#   4k changed (T-GPLR-45 binary)     -- tg 96
#   50k control                       -- tg 3, the performance pair
#   50k changed                       -- tg 3
#
# Control = build/llama-cuda/bin/gplr45-control (side-copied BEFORE the rebuild,
# exe + ALL DLLs: the streaming code lives in llama.dll). Changed = the current
# Release build. Each arm acquires and releases the GPU mutex itself.
# Run detached; completion is signalled by the four gplr45-*-*.done sentinels.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr45-arm.ps1'
$pre  = Join-Path $root 'build\llama-cuda\bin\gplr45-control\llama-layer-streaming.exe'
$post = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'

foreach ($p in @($arm, $pre, $post)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "MISSING: $p"; exit 1 }
}
Write-Host ("[pair] control  exe sha256[:16]={0}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $pre).Hash.Substring(0,16))
Write-Host ("[pair] changed  exe sha256[:16]={0}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $post).Hash.Substring(0,16))

$runs = @(
    @{ Label = 'control'; Exe = $pre;  Tokens = 4096;  Tg = 96 },
    @{ Label = 'changed'; Exe = $post; Tokens = 4096;  Tg = 96 },
    @{ Label = 'control'; Exe = $pre;  Tokens = 50000; Tg = 3  },
    @{ Label = 'changed'; Exe = $post; Tokens = 50000; Tg = 3  }
)

foreach ($r in $runs) {
    Write-Host ("[pair] ===== {0} tokens {1} tg {2} =====" -f $r.Label, $r.Tokens, $r.Tg)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Label $r.Label -Exe $r.Exe -PromptTokens $r.Tokens -TgTokens $r.Tg
    Write-Host ("[pair] arm exit: {0}" -f $LASTEXITCODE)
}
Write-Host '[pair] all four arms finished'
