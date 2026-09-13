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
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\run-tests.ps1 -Suite qa-smoke # T-GPLR-07 semantic-QA smoke gate: default 512-token run on the 512-token anchor
#                       (calibrated prompt -> runner + dedicated --telemetry JSONL -> strict
#                       --check-telemetry -> checker on the runner's --tg-text file + negative
#                       control; GPU, ~5 min; 50-token rung opt-in via -SmokeLengths 50;
#                       4k+ ladder lengths are manual owner-invoked commands)
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
#                       layer-major; kept for the owner to remove explicitly)#   pp20k-layer-major - T-Q38LM-03: PP 20480 tokens (ubatch 512, 40 chunks) in
#               layer-major order, PP-only, --pp-no-logits, ~10-20 min, GPU
#               (model path: -ModelPath, else $env:Q38LS_MODEL_PATH, else the
#               docs/AGENT_ENVIRONMENT.md path)
#   qa-smoke          - T-GPLR-07 semantic-QA smoke gate. Default run: 512-token
#               rung on the 512-token anchor (full 5-item document anchor); the
#               50-token rung is opt-in via -SmokeLengths 50 (comma list; the
#               quoted-string form silently binds as one int and must not be
#               used). 4k+ ladder lengths are manual owner-invoked commands, never
#               part of this suite. The gate a developer runs after
#               every runner change that touches the experimental path: calibrated
#               needle prompt (gen-needle-prompt.py, exact token count) -> runner with
#               --prompt-file + dedicated --telemetry JSONL under the GPU mutex ->
#               strict --check-telemetry on the JSONL -> check-answer.py grading the
#               runner's --tg-text generated-text file directly (no trace re-parsing)
#               + negative control. (model path: -ModelPath, else
#               $env:QWEN36_MODEL, else the docs/AGENT_ENVIRONMENT.md path)
#
# Exits with ctest's exit code (0 = all passed).

[CmdletBinding()]
param(
    [ValidateSet('all', 'quick', 'boundary', 'telemetry', 'host', 'qa-smoke', 'pp20k', 'pp20k-layer-major')]
    [string] $Suite = 'all',

    [string] $BuildDir = '',
    [string] $Config = 'Release',
    [string] $ModelPath = '',

    # qa-smoke rung selection: default = 512 only; the 50-token rung is opt-in
    # via -SmokeLengths 50 (comma list). 4k+ ladder lengths are manual
    # owner-invoked commands, never part of this suite.
    [int[]] $SmokeLengths = @(512)
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
        # T-GPLR-05: host-only capability probe + policy-refusal tests are
        # included here (no GPU, no lock).
        # Host-only: the layout unit test + the fixture JSONL checks (lm40,
        # tail513, tail1025) + the T-GPLR-10 gate fixtures (mtp+handoff+moe
        # refusal, layer-major tg cap and its handoff lift, dense legal
        # combos). Exclude the tail513 GPU run targets AND the tail513-lm /
        # tail513-bm host checks that FIXTURES-require those GPU runs (the
        # fixture files under fixtures/ are what the host suite uses).
        Include = 'layer-streaming-(lm40|tail513|tail1025|make-unsupported-fixture|probe-unsupported|refuse-unsupported|check-off-audit|refuse-mtp-handoff-moe|lm-tg-cap|lm-tg-cap-lift|dense-mtp-handoff-legal|dense-attn-only-legal|dense-mtp-auto-refusal)(-|$)|test-layer-streaming-(layout|model-inspect)'
        Exclude = 'tail513-(lm|bm)(-run)?$'
        TimeoutSeconds = 300
    }
    'qa-smoke' = @{
        # T-GPLR-07 semantic-QA smoke gate, end-to-end on the real surface under
        # the GPU mutex. Default rung: 512 tokens on the 512-token anchor (full
        # 5-item document anchor); the 50-token rung is opt-in via -SmokeLengths
        # 50. 4k+ ladder lengths are manual owner-invoked commands, never part
        # of this suite. The dedicated --telemetry JSONL +
        # strict --check-telemetry pair is the point: it prevents exactly the
        # mixed-log-only 'check-telemetry OK' claims the ladder recorded for the
        # 512/4k rungs. Every step must pass or the suite exits 1.
        Gpu = $true
        QaSmoke = $true
        TimeoutSeconds = 1800
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
elseif ($spec.QaSmoke) {
    # T-GPLR-07: semantic-QA smoke gate. Smoke tier = 50/512-token runs on the
    # 512-token anchor (default: 512 only). Runs the real surface end-to-end:
    #   calibrated needle prompt (gen-needle-prompt.py, exact token count;
    #   50 = N1 scaffold, 512 = full document mode)
    #   -> runner with --prompt-file + dedicated --telemetry JSONL under the
    #      GPU mutex (single-ubatch PP: --ubatch = the REAL achieved token
    #      count, the documented workaround for the tail-ubatch gap)
    #   -> strict --check-telemetry on the JSONL (the dedicated-JSONL gate)
    #   -> check-answer.py per-needle grading (--items from the generator's
    #      per-length items= list) + negative control on the same tg-text
    #      (must FAIL as designed; a grader that cannot fail is not one)
    $exe = Join-Path $buildDirResolved "bin\$Config\llama-layer-streaming.exe"
    $tok = Join-Path $buildDirResolved "bin\$Config\llama-tokenize.exe"
    if (-not (Test-Path -LiteralPath $tok)) {
        Write-Host "ERROR: llama-tokenize.exe not found at $tok (needed for prompt calibration)." -ForegroundColor Red
        exit 1
    }
    if (-not $ModelPath) { $ModelPath = $env:QWEN36_MODEL }
    if (-not $ModelPath) { $ModelPath = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
    if (-not (Test-Path -LiteralPath $ModelPath)) {
        Write-Host "ERROR: model not found at $ModelPath. Pass -ModelPath or set QWEN36_MODEL." -ForegroundColor Red
        exit 1
    }

    $py = 'python'
    $gen = Join-Path $PSScriptRoot 'semantic-qa\gen-needle-prompt.py'
    $chk = Join-Path $PSScriptRoot 'semantic-qa\check-answer.py'
    $promptDir = Join-Path $buildDirResolved 'gplr07-qa-prompts'
    New-Item -ItemType Directory -Force -Path $promptDir | Out-Null

    $failed = $false
    foreach ($len in $SmokeLengths) {
        if ($len -lt 15) {
            Write-Host "FAILED: smoke rung length $len is below the generator floor (15 tokens)." -ForegroundColor Red
            $failed = $true
            break
        }
        $prompt = Join-Path $promptDir "anchor-$len.txt"
        Write-Host "[qa-smoke] $len tokens: calibrating anchor prompt..."
        $genOut = & $py $gen --target-tokens $len --calibrate $tok --model $ModelPath --out $prompt 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: prompt calibration exited $LASTEXITCODE" -ForegroundColor Red
            Write-Host $genOut -ForegroundColor Red
            $failed = $true
            break
        }
        Write-Host $genOut
        $achieved = ($genOut | Select-String 'achieved=(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
        $items = ($genOut | Select-String 'items=(\S+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -First 1)
        if (-not $achieved -or -not $items) {
            Write-Host "FAILED: generator output missing achieved=/items= (got: $genOut)" -ForegroundColor Red
            $failed = $true
            break
        }

        $trace = Join-Path $buildDirResolved "gplr07-qa-$len.jsonl"
        $tgtext = Join-Path $buildDirResolved "gplr07-qa-$len.txt"
        $log = Join-Path $buildDirResolved "gplr07-qa-$len.log"
        # T-GPLR-12: --no-handoff pins this gate to the recorded streaming-TG
        # shape (the default-profile QA rungs run separately with auto).
        $runCmd = "`"$exe`" --model `"$ModelPath`" --prompt-file `"$prompt`" --ubatch $achieved --tg-tokens 96 --pp-no-logits --pp-schedule batch-major --residency moe-expert-bank --kv-cache q8 --no-handoff --allow-full-residency --telemetry `"$trace`" --tg-text `"$tgtext`" > `"$log`" 2>&1"
        Write-Host "[qa-smoke] $len tokens: runner under GPU mutex (log $log)..."
        & powershell -NoProfile -ExecutionPolicy Bypass -File $lock `
            -TimeoutSeconds $spec.TimeoutSeconds -Command $runCmd
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            Write-Host "FAILED: $len-token run exited $exitCode" -ForegroundColor Red
            if (Test-Path -LiteralPath $log) { Get-Content -LiteralPath $log -Tail 25 | ForEach-Object { Write-Host $_ -ForegroundColor Red } }
            $failed = $true
            continue
        }

        Write-Host "[qa-smoke] $len tokens: strict --check-telemetry on $trace"
        & $exe --check-telemetry $trace
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: check-telemetry rejected $trace" -ForegroundColor Red
            $failed = $true
            continue
        }

        Write-Host "[qa-smoke] $len tokens: grading generated text $tgtext (items=$items)"
        & $py $chk check --tg-text $tgtext --items $items
        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: checker rejected the $len-token answer" -ForegroundColor Red
            $failed = $true
            continue
        }

        Write-Host "[qa-smoke] $len tokens: negative control on the same tg-text (must FAIL as designed)"
        $negOut = & $py $chk check --tg-text $tgtext --items $items --negative 2>&1
        $negExit = $LASTEXITCODE
        $negText = $negOut -join "`n"
        Write-Host $negOut
        if ($negExit -eq 0 -or $negText -notmatch 'aggregate=FAIL') {
            Write-Host "FAILED: negative control did not fail as designed (exit $negExit)" -ForegroundColor Red
            $failed = $true
            continue
        }
        Write-Host "[qa-smoke] $len-token rung PASSED (check-telemetry OK, answer PASS, negative control FAILs)" -ForegroundColor Green
    }

    if ($failed) {
        Write-Host "FAILED: suite 'qa-smoke' failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "SUCCESS: suite 'qa-smoke' passed." -ForegroundColor Green
    exit 0
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
