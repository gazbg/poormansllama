# Detached full-suite runner for the T-GPLR-54 A3 close-out: sets the model-path
# env var the CTest run-targets require (the interactive shell lost it on the
# 2026-09-22 Freebuff restart) and tees run-tests.ps1 to a log file. run-tests.ps1
# performs its own GPU-mutex handling for the QA-smoke portion.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
Set-Location -LiteralPath $root
$env:Q38LS_MODEL_PATH = 'C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf'
& powershell -NoProfile -ExecutionPolicy Bypass -File tools/run-tests.ps1 *>&1 |
    Out-File build/llama-cuda/gplr54-suite.out -Encoding utf8
"SUITE EXIT: {0}" -f $LASTEXITCODE |
    Out-File build/llama-cuda/gplr54-suite.out -Append -Encoding utf8
