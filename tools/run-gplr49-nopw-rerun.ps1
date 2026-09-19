# T-GPLR-49 follow-up runs, in one session and one mutex queue:
#
#  1. the two --no-pw-probe CHANGED arms again. The first attempt exited 0 with
#     check-telemetry OK and no refusal (the repair worked), but the arm wrapper
#     demanded the T-GPLR-45 composition dump, which prepare_chunk never builds
#     when pw_probe is off -- a harness expectation error, not a code failure.
#     The wrapper now treats the dump as pw-only, so these verdicts match reality.
#  2. the semantic-QA smoke rung (512-token anchor, must-fail negative control),
#     owed because T-GPLR-49 touches the streaming path.
#
# Each GPU step acquires and releases Global\AI-GPU-EXCLUSIVE-v1 itself.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr49-arm.ps1'
$post = Join-Path $root 'build\llama-cuda\bin\gplr49-changed\llama-layer-streaming.exe'
if (-not (Test-Path -LiteralPath $post)) { Write-Host "MISSING: $post"; exit 1 }

foreach ($r in @(@{ Tokens = 4096; Tg = 96 }, @{ Tokens = 50000; Tg = 3 })) {
    Write-Host ("[rerun] ===== changed tokens {0} tg {1} mode nopw =====" -f $r.Tokens, $r.Tg)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Label changed -Exe $post `
        -PromptTokens $r.Tokens -TgTokens $r.Tg -Mode nopw
    Write-Host ("[rerun] arm exit: {0}" -f $LASTEXITCODE)
}

Write-Host '[rerun] ===== semantic-QA smoke rung ====='
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root 'tools\run-tests.ps1') -Suite qa-smoke
Write-Host ("[rerun] qa-smoke exit: {0}" -f $LASTEXITCODE)
Write-Host '[rerun] all steps finished'
