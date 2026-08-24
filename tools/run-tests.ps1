# tools/run-tests.ps1
#
# Convenience runner for the Qwen3.8 layer-streaming CTest suite.
# Wraps the correct ctest pattern in the shared-GPU mutex
# (Global\AI-GPU-EXCLUSIVE-v1) for any suite that touches CUDA, and runs
# host-only suites without the lock.
#
# Usage (from any directory):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1            # full suite (~8 min, GPU)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite quick     # everything except the 350s transition run
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite boundary  # boundary + slot lifecycle (~35s, GPU)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite telemetry # host-only JSONL validation (no GPU, seconds)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite host      # host-only chunk-layout + telemetry fixture tests (no GPU, seconds)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite pp20k   # 20K-token PP + 3-token TG experiment (~10-20 min, GPU)
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite pp20k-layer-major # 20K layer-major PP (T-Q38LM-03, ~10-20 min, GPU)
#
# Suites:
#   all               - all layer-streaming / lifecycle / boundary / telemetry targets
#                       (includes the T-Q38CTX-01 tail runs: tail513 lm/bm, lm-tiny1, lm511)
#   quick             - scheduler + pp512 + boundary + lifecycle (no transition/telemetry targets)
#   boundary          - qwen38-layer-boundary + layer-slot-lifecycle only
#   telemetry         - host-side telemetry validator targets only (no GPU, no lock)
#   host              - host-only chunk-layout unit test + lm40/tail513/tail1025
#                       telemetry fixtures (T-Q38CTX-01/02, no GPU, no lock)
#   pp20k             - one experimental run: PP 20480 tokens (ubatch 512) + TG 3 tokens,
#                       --pp-no-logits, ~10-20 min, GPU (BLOCKED/superseded by
#                       layer-major; kept for the owner to remove explicitly)
#   pp20k-layer-major - T-Q38LM-03: PP 20480 tokens (ubatch 512, 40 chunks) in
#                       layer-major order, PP-only, --pp-no-logits, ~10-20 min, GPU
#               (model path: -ModelPath, else $env:Q38LS_MODEL_PATH, else the
#               docs/AGENT_ENVIRONMENT.md path)
#
# Exits with ctest's exit code (0 = all passed).

[CmdletBinding()]
param(
    [ValidateSet('all', 'quick', 'boundary', 'telemetry', 'host', 'pp20k', 'pp20k-layer-major')]
    [string] $Suite = 'all',

    [string] $BuildDir = '',
    [string] $Config = 'Release',
    [string] $ModelPath = ''
)

$ErrorActionPreference = 'Stop'

if (-not $BuildDir) {
    $BuildDir = Join-Path $PSScriptRoot '..\build\llama-cuda'
}

# --verbose: print every test's command and full output (the runners print
# per-phase tok/s + telemetry summaries on stderr) even when the suite passes
$cTest = "ctest --test-dir $BuildDir -C $Config --verbose"
$lock = Join-Path (Split-Path $PSCommandPath -Parent) 'with-gpu-lock.ps1'
$buildDirResolved = (Resolve-Path -LiteralPath $BuildDir).Path

$suites = @{
    'all' = @{
        Gpu = $true
        Include = 'layer-streaming|layer-slot-lifecycle|layer-boundary'
        Exclude = ''
        TimeoutSeconds = 1800
    }
    'quick' = @{
        # Excludes the expensive transition run AND the telemetry CTest targets
        # (their fixture would re-pull in the transition run); the same checks
        # are available host-only via -Suite telemetry.
        Gpu = $true
        Include = 'layer-streaming-(scheduler|pp512)|layer-slot-lifecycle|layer-boundary'
        Exclude = ''
        TimeoutSeconds = 900
    }
    'boundary' = @{
        Gpu = $true
        Include = 'qwen38-layer-boundary|layer-slot-lifecycle'
        Exclude = ''
        TimeoutSeconds = 300
    }
    'telemetry' = @{
        # Host-only: runs the same checks as the layer-streaming-telemetry and
        # layer-streaming-telemetry-negative CTest targets, but by invoking the
        # validator exe directly (no GPU, no ctest fixture pull-in).
        Gpu = $false
        Host = $true
        TimeoutSeconds = 120
    }
    'host' = @{
        # T-Q38CTX-01/02: host-only unit + fixture tests (no GPU, no lock):
        # the chunk-layout unit test and the lm40/tail513/tail1025 telemetry
        # fixtures (positive + WILL_FAIL negatives).
        Gpu = $false
        # Host-only: the layout unit test + the fixture JSONL checks (lm40,
        # tail513, tail1025). Exclude the tail513 GPU run targets AND the
        # tail513-lm / tail513-bm host checks that FIXTURES-require those GPU
        # runs (the fixture files under fixtures/ are what the host suite uses).
        Include = 'layer-streaming-(lm40|tail513|tail1025)-|test-layer-streaming-layout'
        Exclude = 'tail513-(lm|bm)(-run)?$'
        TimeoutSeconds = 300
    }
    'pp20k' = @{
        # Experimental: large-prompt PP (20480 tokens, ubatch 512 = 40 whole
        # ubatches) + 3 greedy TG tokens, single direct runner invocation
        # (not a CTest target). --pp-no-logits avoids unused PP logits.
        # BLOCKED/SUPERSEDED by the layer-major design (T-Q38LS-07); kept for
        # the owner to remove explicitly. Do not extend it.
        Gpu = $true
        Direct = $true
        DirectArgs = '--layer-slots 4 --pp-tokens 20480 --ubatch 512 --tg-tokens 3 --greedy --no-mtp --pp-no-logits'
        DirectTrace = 'q38ls-pp20k.jsonl'
        TimeoutSeconds = 3600
    }
    'pp20k-layer-major' = @{
        # T-Q38LM-03 + T-Q38LM-04: 20480-token / 512-ubatch layer-major PP pass
        # (40 chunks) through all 64 layers, then 3 greedy TG tokens, single
        # direct runner invocation (not a CTest target). One run satisfies both
        # the 20K PP telemetry (T-Q38LM-03) and the PP->TG handoff (T-Q38LM-04).
        # --pp-no-logits avoids a 20480-row PP logits buffer.
        Gpu = $true
        Direct = $true
        DirectArgs = '--pp-schedule layer-major --layer-slots 4 --pp-tokens 20480 --ubatch 512 --tg-tokens 3 --greedy --no-mtp --pp-no-logits'
        DirectTrace = 'q38ls-pp20k-layer-major.jsonl'
        TimeoutSeconds = 3600
    }
}

$spec = $suites[$Suite]

if (-not (Test-Path -LiteralPath (Join-Path $buildDirResolved 'CTestTestfile.cmake'))) {
    Write-Host "ERROR: No CMake build found at $buildDirResolved. Build first with:" -ForegroundColor Red
    Write-Host '  cmake -S third_party\llama.cpp -B build\llama-cuda -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON'
    Write-Host '  cmake --build build\llama-cuda --config Release --target llama-layer-streaming --parallel 16'
    exit 1
}

if ($spec.Direct) {
    if (-not $ModelPath) { $ModelPath = $env:Q38LS_MODEL_PATH }
    if (-not $ModelPath) { $ModelPath = 'C:/Models/ALL/other/qwen-3.8/Qwen3.8-27B-UD-Q6_K_XL.gguf' }
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        Write-Host "ERROR: model not found at $ModelPath. Pass -ModelPath." -ForegroundColor Red
        exit 1
    }
}

$ctestCommand = $cTest
if (-not $spec.Host) {
    $ctestCommand = "$cTest -R `"$($spec.Include)`""
    if ($spec.Exclude) {
        $ctestCommand += " -E `"$($spec.Exclude)`""
    }
    # with-gpu-lock.ps1 executes via cmd.exe, where '|' is a pipe even inside
    # double quotes. Escape '|' as '^|' so the regex passes through literally.
    if ($spec.Gpu) {
        $ctestCommand = $ctestCommand -replace '\|', '^|'
    }
}

Write-Host "Suite: $Suite"
Write-Host "Command: $ctestCommand"

if ($spec.Host) {
    # Replicates the CTest telemetry targets without ctest fixtures:
    #  - positive: validate the transition trace (must exit 0, write summary)
    #  - negative: validate a missing trace (must exit non-zero)
    $exe = Join-Path $buildDirResolved "bin\$Config\llama-layer-streaming.exe"
    $trace = Join-Path $buildDirResolved 'q38ls-pp-tg.jsonl'
    if (-not (Test-Path -LiteralPath $trace)) {
        Write-Host "ERROR: Missing trace $trace. Run the GPU suites first (e.g. -Suite all)." -ForegroundColor Red
        exit 1
    }

    Write-Host "[1/2] Positive: --validate-telemetry $trace"
    & $exe --validate-telemetry $trace
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: positive validation exited $LASTEXITCODE." -ForegroundColor Red
        exit $LASTEXITCODE
    }

    $missing = Join-Path $buildDirResolved 'q38ls-does-not-exist.jsonl'
    Write-Host "[2/2] Negative: --validate-telemetry $missing (expect non-zero exit)"
    & $exe --validate-telemetry $missing
    $negExit = $LASTEXITCODE
    if ($negExit -eq 0) {
        Write-Host "FAILED: negative validation unexpectedly exited 0." -ForegroundColor Red
        exit 1
    }
    Write-Host "Negative correctly rejected (exit $negExit)." -ForegroundColor Green

    Write-Host "SUCCESS: suite '$Suite' passed." -ForegroundColor Green
    exit 0
}
elseif ($spec.Direct) {
    $exe = Join-Path $buildDirResolved "bin\$Config\llama-layer-streaming.exe"
    $trace = Join-Path $buildDirResolved $spec.DirectTrace
    $DirectCmd = "`"$exe`" --model `"$ModelPath`" $($spec.DirectArgs) --telemetry `"$trace`""
    Write-Host "Running direct experiment under GPU mutex: $DirectCmd"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock `
        -TimeoutSeconds $spec.TimeoutSeconds -Command $DirectCmd
    $exitCode = $LASTEXITCODE
}
elseif ($spec.Gpu) {
    Write-Host "Running under GPU mutex (waits if another process holds it)..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $lock `
        -TimeoutSeconds $spec.TimeoutSeconds -Command $ctestCommand
    $exitCode = $LASTEXITCODE
} else {
    Write-Host "Running without GPU mutex (host-only tests)..."
    Invoke-Expression $ctestCommand
    $exitCode = $LASTEXITCODE
}

if ($exitCode -eq 0) {
    Write-Host "SUCCESS: suite '$Suite' passed." -ForegroundColor Green
} else {
    Write-Host "FAILED: suite '$Suite' exited with code $exitCode." -ForegroundColor Red
}
exit $exitCode
