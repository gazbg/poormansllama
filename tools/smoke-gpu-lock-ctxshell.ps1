# GPU SMOKE TEST for the ctx_shell arm-launch path (one mutex hold, seconds of GPU).
#
# WHY THIS EXISTS. GPU arms were launched from the app's terminal because
# ctx_shell refused `powershell` (lean-ctx shell allowlist) and, separately,
# because ctx_shell inherits no CUDA_PATH -- which broke MSBuild CUDA builds
# there (docs/AGENT_ENVIRONMENT.md addenda F and G; `ctest` and `powershell` are
# now allowlisted globally). Whether a mutex-held ARM can run to completion from
# ctx_shell without CUDA_PATH was still untested. This wrapper is that test.
#
# It is deliberately NOT one of the real arms. Those cost 6-10 min of GPU
# (200k ledger, 144k stock probe) and this repo's rule is not to run them merely
# for confidence. Instead this runs the SMALLEST real GPU workload that still
# exercises the whole chain end to end: acquire the shared mutex, load the MoE
# vehicle through the residency path, run a few prefill tokens and a few TG
# rounds on CUDA, emit telemetry, and have the strict checker read it back.
#
# What it proves when it exits 0:
#   - `powershell` can be launched at all from ctx_shell (the allowlist gate);
#   - the named mutex Global\AI-GPU-EXCLUSIVE-v1 is acquired and released;
#   - llama-layer-streaming.exe initialises CUDA and executes WITHOUT CUDA_PATH;
#   - telemetry is written and passes `--check-telemetry`.
# What it does NOT prove: any performance number, any arm's behaviour at a real
# shape, or that the memory poller / long runs behave. It is a connectivity
# check, not evidence about the mechanism.
#
# Exit codes: 0 = chain works. 1 = a step failed (the log says which).
# Log: build/llama-cuda/gpu-smoke-ctxshell.log (gitignored build/ root).
param(
    # Prefill tokens for the smoke arm. Small on purpose: model load dominates.
    [int] $PpTokens = 64,

    # TG rounds for the smoke arm. Enough to emit rowset_round / rowset_fetch
    # records for the checker, not enough to be a measurement.
    [int] $TgTokens = 8,

    # Mutex wait. Short: if the shared GPU is busy, this test can wait for a
    # later window rather than hold a queue.
    [int] $LockTimeoutSeconds = 300
)
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$exe    = Join-Path $root 'build\llama-cuda\bin\Release\llama-layer-streaming.exe'
$lock   = Join-Path $root 'tools\with-gpu-lock.ps1'
$outDir = Join-Path $root 'build\llama-cuda'
$log    = Join-Path $outDir 'gpu-smoke-ctxshell.log'
$moe    = $env:QWEN36_MODEL
if (-not $moe) { $moe = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
# The recorded 512-token anchor is the cheapest prompt on disk for this vehicle.
$prompt = Join-Path $root 'build\llama-cuda\gplr13-ladder\512\anchor-512.txt'
$trace  = Join-Path $outDir 'gpu-smoke-ctxshell.jsonl'
$txt    = Join-Path $outDir 'gpu-smoke-ctxshell.txt'

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
if (Test-Path -LiteralPath $log) { Remove-Item $log -Force }
foreach ($p in @($trace, $txt)) { if (Test-Path -LiteralPath $p) { Remove-Item $p -Force } }

function Log([string]$m) {
    $line = ('[{0}] {1}' -f (Get-Date -Format 'HH:mm:ss.fff'), $m)
    Write-Host $line
    Add-Content -LiteralPath $log -Value $line
}

Log '=== GPU smoke test: ctx_shell -> mutex -> CUDA ==='
# The environment this shell actually has. CUDA_PATH is EXPECTED to be empty
# here and the arm must not need it; recording it makes the run self-describing
# rather than dependent on someone remembering addendum F.
Log ('pid={0} parent-pid={1}' -f $PID, (Get-CimInstance Win32_Process -Filter "ProcessId = $PID").ParentProcessId)
Log ('cwd={0}' -f (Get-Location).Path)
Log ('CUDA_PATH=[{0}]' -f $env:CUDA_PATH)
Log ('CUDA_PATH_V13_2=[{0}]' -f $env:CUDA_PATH_V13_2)
Log ('COMSPEC=[{0}] PATHEXT present={1}' -f $env:COMSPEC, [bool]$env:PATHEXT)

foreach ($f in @($exe, $lock, $prompt)) {
    if (-not (Test-Path -LiteralPath $f)) { Log ('FATAL missing: ' + $f); exit 1 }
}
if (-not (Test-Path -LiteralPath $moe)) { Log ('FATAL missing model: ' + $moe); exit 1 }

# The exe links llama.dll/ggml.dll/cudart from its own bin/Release directory;
# put that directory on PATH for the child, exactly as the T-GPLR-40 wrapper does.
$env:PATH = (Join-Path $root 'build\llama-cuda\bin\Release') + ';' + $env:PATH

$a0 = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1
Log ('GPU before: ' + ($a0 -join '; '))

$argStr = ('--model "{0}" --prompt-file "{1}" --residency moe-expert-bank --residency-tg ' +
           '--pp-schedule batch-major --pp-tokens {2} --ubatch 512 --tg-tokens {3} ' +
           '--layer-slots 4 --kv-cache q8 --greedy --no-mtp --pp-no-logits --allow-full-residency ' +
           '--telemetry "{4}" --tg-text "{5}"') -f $moe, $prompt, $PpTokens, $TgTokens, $trace, $txt
$cmd = ('"{0}" {1}' -f $exe, $argStr)
Log ('ARM: ' + $cmd)

$sw = [System.Diagnostics.Stopwatch]::StartNew()
# The mutex hold for this smoke arm, via the repo's shared wrapper. Captured
# first and piped second on purpose: piping the invocation in the same statement
# risks reading $LASTEXITCODE after the pipeline, not after the script.
$lockOut = & $lock -TimeoutSeconds $LockTimeoutSeconds -Command $cmd 2>&1
$armExit = $LASTEXITCODE
$lockOut | ForEach-Object { Log ('  | ' + $_) }
$sw.Stop()
Log ('arm exit {0} after {1:N1} s' -f $armExit, $sw.Elapsed.TotalSeconds)

$a1 = & nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>&1
Log ('GPU after: ' + ($a1 -join '; '))

$ok = $true
if ($armExit -ne 0) { Log ('FAIL: arm exit ' + $armExit); $ok = $false }

if (Test-Path -LiteralPath $trace) {
    $raw = [System.IO.File]::ReadAllText($trace)
    foreach ($ev in @('rowset_init', 'rowset_fetch', 'rowset_round', 'rowset_refused', 'peak_vram_bytes')) {
        Log ('{0}: {1}' -f $ev, ([regex]::Matches($raw, [regex]::Escape($ev))).Count)
    }
    Log '--- strict check-telemetry ---'
    $c = & $exe --check-telemetry $trace 2>&1
    $c | ForEach-Object { Log ('  # ' + $_) }
    if ($LASTEXITCODE -ne 0) { Log ('FAIL: check-telemetry exit ' + $LASTEXITCODE); $ok = $false }
} else {
    Log 'FAIL: no telemetry trace was produced'
    $ok = $false
}

if (Test-Path -LiteralPath $txt) {
    $t = [System.IO.File]::ReadAllText($txt)
    Log ('tg text: {0} bytes' -f $t.Length)
    if ($t.Length -eq 0) { Log 'FAIL: empty TG text'; $ok = $false }
} else {
    Log 'FAIL: no TG text was produced'
    $ok = $false
}

Log ('RESULT: {0}' -f $(if ($ok) { 'CHAIN OK (ctx_shell -> mutex -> CUDA, no CUDA_PATH)' } else { 'FAILED' }))
exit $(if ($ok) { 0 } else { 1 })
