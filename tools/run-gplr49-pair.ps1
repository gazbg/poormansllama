# T-GPLR-49: the eight arms, back-to-back in ONE session so the pairs share
# machine state (the recorded ~2.9% cross-session spread makes cross-session
# comparisons worthless).
#
#   pw   = the DEFAULT layer-major path (pw_probe on). Must not move at all:
#          T-GPLR-49 repairs an unexercised mode and must never cost the default.
#   nopw = --no-pw-probe, the ONLY reachable form of the pre-fix offset formula.
#
#   pw   4k/96  control | changed     identity: greedy ids + act_bind stream
#   pw   50k/3  control | changed     identity, larger shape
#   nopw 4k/96  control (EXPECTED to refuse loudly at chunk 4) | changed (must run clean)
#   nopw 50k/3  control (EXPECTED to refuse at chunk 49)       | changed (must run clean)
#
# Control = build/llama-cuda/bin/gplr49-control (built from nested HEAD, i.e.
# the PRE-fix source, exe + ALL DLLs: the streaming code lives in llama.dll).
# Changed = build/llama-cuda/bin/gplr49-changed (this task's binary).
# Each arm acquires and releases the GPU mutex itself. Run detached; completion
# is signalled by the eight gplr49-*.done sentinels.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr49-arm.ps1'
$pre  = Join-Path $root 'build\llama-cuda\bin\gplr49-control\llama-layer-streaming.exe'
$post = Join-Path $root 'build\llama-cuda\bin\gplr49-changed\llama-layer-streaming.exe'

foreach ($p in @($arm, $pre, $post)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "MISSING: $p"; exit 1 }
}
$preDll  = Join-Path (Split-Path $pre  -Parent) 'llama.dll'
$postDll = Join-Path (Split-Path $post -Parent) 'llama.dll'
foreach ($p in @($preDll, $postDll)) {
    if (-not (Test-Path -LiteralPath $p)) { Write-Host "MISSING DLL: $p"; exit 1 }
}
Write-Host ("[pair] control  exe {0} llama.dll {1}" -f `
    (Get-FileHash -Algorithm SHA256 -LiteralPath $pre).Hash.Substring(0,16), `
    (Get-FileHash -Algorithm SHA256 -LiteralPath $preDll).Hash.Substring(0,16))
Write-Host ("[pair] changed  exe {0} llama.dll {1}" -f `
    (Get-FileHash -Algorithm SHA256 -LiteralPath $post).Hash.Substring(0,16), `
    (Get-FileHash -Algorithm SHA256 -LiteralPath $postDll).Hash.Substring(0,16))

$runs = @(
    @{ Label = 'control'; Exe = $pre;  Tokens = 4096;  Tg = 96; Mode = 'pw';   ExpectRefusal = $false },
    @{ Label = 'changed'; Exe = $post; Tokens = 4096;  Tg = 96; Mode = 'pw';   ExpectRefusal = $false },
    @{ Label = 'control'; Exe = $pre;  Tokens = 50000; Tg = 3;  Mode = 'pw';   ExpectRefusal = $false },
    @{ Label = 'changed'; Exe = $post; Tokens = 50000; Tg = 3;  Mode = 'pw';   ExpectRefusal = $false },
    @{ Label = 'control'; Exe = $pre;  Tokens = 4096;  Tg = 96; Mode = 'nopw'; ExpectRefusal = $true  },
    @{ Label = 'changed'; Exe = $post; Tokens = 4096;  Tg = 96; Mode = 'nopw'; ExpectRefusal = $false },
    @{ Label = 'control'; Exe = $pre;  Tokens = 50000; Tg = 3;  Mode = 'nopw'; ExpectRefusal = $true  },
    @{ Label = 'changed'; Exe = $post; Tokens = 50000; Tg = 3;  Mode = 'nopw'; ExpectRefusal = $false }
)

foreach ($r in $runs) {
    Write-Host ("[pair] ===== {0} tokens {1} tg {2} mode {3} expect_refusal {4} =====" -f `
        $r.Label, $r.Tokens, $r.Tg, $r.Mode, $r.ExpectRefusal)
    $armArgs = @('-Label', $r.Label, '-Exe', $r.Exe, '-PromptTokens', $r.Tokens,
                 '-TgTokens', $r.Tg, '-Mode', $r.Mode)
    if ($r.ExpectRefusal) { $armArgs += '-ExpectRefusal' }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm @armArgs
    Write-Host ("[pair] arm exit: {0}" -f $LASTEXITCODE)
}
Write-Host '[pair] all eight arms finished'
