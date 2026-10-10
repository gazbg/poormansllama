<#
    T-GPLR-82 N1 arm: run the FROZEN step-2 binaries with the host expert leg
    switched on, under the shared-GPU mutex, then check the demand ledger.

    WHY THE FROZEN DIRECTORY, NOT bin\Release. The experiment's control is the
    build that introduced --experts-host. bin\Release is overwritten by the next
    build, so an arm pointed there is an arm against whatever happens to be
    there. This launcher REFUSES rather than falling back: a silent fallback
    would run a different binary than the one the results would be filed under.
    Every DLL travels with the exe for the same reason -- copying the exe alone
    leaves it loading a newer llama.dll.

    WHY THE WRAPPER IS STARTED, NOT WAITED ON. tools/with-gpu-lock.ps1 owns the
    mutex, the pre/post GPU probe and the verdict file. The launcher clears the
    verdict files for this tag BEFORE starting it, because an immediate read
    would otherwise pick up the PREVIOUS run's marker and verdict and report a
    stale answer. Nothing here is detached through setsid.

    WHAT IT DOES NOT DO. It takes no measurement of quality and claims none. The
    post-arm checker verifies BOOKKEEPING (the routed-expert ledger balances),
    not arithmetic -- a trace can balance while the CPU half computes the wrong
    numbers, and that stays with the existing quality benchmark.

    Usage:
      pwsh -File tools/run-gplr82-hostarm.ps1 -DryRun          # print, launch nothing
      pwsh -File tools/run-gplr82-hostarm.ps1 -TgTokens 16 -ExpertsHost 2
      pwsh -File tools/run-gplr82-hostarm.ps1 -ExpertsHost 0 -AllowHostOff   # host-leg-off BASELINE arm
#>
[CmdletBinding()]
param(
    [int]    $TgTokens   = 16,
    [int]    $ExpertsHost = 2,
    [string] $PromptFile = '',
    [string] $ModelPath  = '',
    [string] $Tag        = '',
    [int]    $TimeoutSeconds = 1800,
    [switch] $DryRun,
    [switch] $StrictProvenance,
    [switch] $SkipChecker,
    # T-GPLR-82 N1 ATTRIBUTION: run the SAME frozen binary with the host leg OFF
    # (--experts-host 0). This baseline arm is what separates "the enabled host
    # leg broke the row-set graph build" from "the row-set graph build was already
    # broken before T-GPLR-82 S2". It changes no source and no binary: --experts-host
    # is a RUNTIME gate (main.cpp: 'Default 0 = off, which builds no host leg at
    # all'), so one binary can serve as both arms.
    [switch] $AllowHostOff
)

$ErrorActionPreference = 'Stop'

$root     = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$frozen   = Join-Path $root 'build\llama-cuda\bin\unscoped-bin\s2-hostleg'
$exe      = Join-Path $frozen 'llama-layer-streaming.exe'
$lock     = Join-Path $root 'tools\with-gpu-lock.ps1'
$verdictLib = Join-Path $root 'tools\arm-verdict.ps1'
$checker  = Join-Path $root 'tools\check-gplr82-ledger.py'

if (-not $ModelPath) {
    $ModelPath = if ($env:Q38LS_MODEL_PATH) { $env:Q38LS_MODEL_PATH }
                 else { 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf' }
}
if (-not $PromptFile) {
    $PromptFile = Join-Path $root 'build\llama-cuda\gplr07-qa-prompts\anchor-512.txt'
}
if (-not $Tag) {
    $Tag = ('gplr82-n1-host{0}-tg{1}' -f $ExpertsHost, $TgTokens)
}

$outDir = Join-Path $root ('build\llama-cuda\gplr82\{0}' -f $Tag)
$tel    = Join-Path $outDir 'trace.jsonl'
$log    = Join-Path $outDir 'runner.log'
$verdictDir = Join-Path $root 'build\llama-cuda\verdicts'

# ---- refusals, before anything is created -----------------------------------
if ($ExpertsHost -lt 0) { throw "-ExpertsHost must be >= 0" }
if ($ExpertsHost -lt 1 -and -not $AllowHostOff) {
    throw "-ExpertsHost must be >= 1 for the N1 arm, or pass -AllowHostOff for the host-leg-off baseline (N1 itself does not test the off case)"
}
if ($TgTokens -lt 1)    { throw "-TgTokens must be >= 1" }
foreach ($p in @($exe, $lock, $verdictLib, $checker)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "required file missing: $p" }
}
if (-not (Test-Path -LiteralPath (Join-Path $frozen 'llama.dll'))) {
    throw "frozen directory has no llama.dll: $frozen -- an exe without its DLLs loads a different build"
}
if (-not (Test-Path -LiteralPath $ModelPath))  { throw "model not found: $ModelPath" }

# Guard against exactly the class of mistake above: every long flag this launcher
# emits must be a flag the runner's parser actually recognises. Cheap, and it
# turns a silent argument-parsing death into a loud refusal before the mutex.
$runnerSrc = Join-Path $root 'third_party\llama.cpp\tools\layer-streaming\main.cpp'
if (Test-Path -LiteralPath $runnerSrc) {
    $srcText = Get-Content -Raw -LiteralPath $runnerSrc
    $missing = @()
    foreach ($f in @('--model', '--prompt-file', '--telemetry', '--residency',
                     '--pp-schedule', '--residency-tg', '--experts-host',
                     '--residency-tg-retain', '--layer-slots', '--kv-graph-pad',
                     '--kv-cache', '--ctx-size', '--ubatch', '--tg-tokens',
                     '--copy-source')) {
        if ($srcText -notmatch [regex]::Escape('arg == "' + $f + '"')) {
            $missing += $f
        }
    }
    if ($missing.Count -gt 0) {
        throw ("these flags are not in the runner's parser: {0}" -f ($missing -join ', '))
    }
}
if (-not (Test-Path -LiteralPath $PromptFile)) { throw "prompt not found: $PromptFile" }

# ---- the runner command -----------------------------------------------------
# Flag-identical to the recorded tg82 row-set shape, with ONE variable changed
# (--experts-host) and the prompt/TG length shortened so N1 is a look, not a
# benchmark. Changing anything else would make the result unattributable.
$runnerArgs = @(
    # `--model`, NOT `-m`: this runner has its own parser with no
    # common_params_parse and no -m alias (verified against tools/layer-streaming/main.cpp),
    # so the short form would be an unrecognised argument and the arm would die
    # in argument parsing while looking like a mechanism failure.
    '--model', $ModelPath,
    '--prompt-file', $PromptFile,
    '--telemetry', $tel,
    '--residency', 'moe-expert-bank',
    '--pp-schedule', 'batch-major',
    '--residency-tg',
    '--experts-host', $ExpertsHost,
    '--residency-tg-retain', '2',
    '--layer-slots', '2',
    '--kv-graph-pad', '256',
    '--kv-cache', 'q8',
    '--ctx-size', '262144',
    '--ubatch', '1024',
    '--tg-tokens', $TgTokens,
    '--copy-source', 'mmap-registered'
)
# with-gpu-lock.ps1 executes this through `cmd.exe /d /s /c`, so this string is
# CMD syntax, NOT PowerShell. A leading '& ' is the PowerShell call operator and
# cmd.exe rejects it verbatim: "& was unexpected at this time." (measured 2026-10-05)
$runnerCmd = '"{0}" {1}' -f $exe, (($runnerArgs | ForEach-Object { '"' + $_ + '"' }) -join ' ')

$lockArgs = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $lock,
    # Start-Process joins -ArgumentList elements with spaces and does NOT re-quote
    # them, so a bare multi-token $runnerCmd arrives at the child as $Command='&'
    # with every runner argument spilled into unbound positionals -- the wrapper
    # then dies on a parse error before it ever takes the mutex. Quote-wrap the
    # value and backslash-escape the inner quotes so it survives as ONE token.
    '-Command', ('"' + ($runnerCmd -replace '"','\"') + '"'),
    '-VerdictDir', $verdictDir,
    '-VerdictTag', $Tag,
    '-TimeoutSeconds', $TimeoutSeconds
)
if (-not $StrictProvenance) {
    # the recorded tg82 arm needed this too: the provenance gate refuses when the
    # arm supplies no binary-state attestation. It is WAIVED, not satisfied, and
    # the banner below says so out loud.
    $lockArgs += '-AllowUnprovenanced'
}

Write-Host '=== T-GPLR-82 N1 host-leg arm ==='
Write-Host ("  frozen binaries : {0}" -f $frozen)
Write-Host ("  model           : {0}" -f $ModelPath)
Write-Host ("  prompt          : {0}" -f $PromptFile)
Write-Host ("  experts-host    : {0}   tg-tokens: {1}" -f $ExpertsHost, $TgTokens)
Write-Host ("  telemetry       : {0}" -f $tel)
Write-Host ("  verdict tag     : {0}" -f $Tag)
Write-Host ("  runner          : {0}" -f $runnerCmd)
if (-not $StrictProvenance) {
    Write-Host '  PROVENANCE GATE : WAIVED via -AllowUnprovenanced (recorded, not satisfied)'
}
Write-Host ''

function Get-GpuComputeApps {
    try {
        $out = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1
        if ($LASTEXITCODE -ne 0) { return "probe-unavailable (exit $LASTEXITCODE)" }
        $txt = ($out | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($txt)) { return 'no compute processes' }
        return ($txt -replace "`r?`n", ' | ')
    } catch {
        return "probe-error: $($_.Exception.Message)"
    }
}

if ($DryRun) {
    Write-Host '--- DRY RUN: nothing launched, no mutex taken, no verdict cleared ---'
    Write-Host ("  lock args      : {0}" -f ($lockArgs -join ' '))
    Write-Host ("  pre-run GPU    : {0}" -f (Get-GpuComputeApps))
    Write-Host '--- would then Clear-ArmVerdict, Start-Process powershell -File with-gpu-lock.ps1, Wait-ArmVerdict ---'
    exit 0
}

New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# ---- GPU state BEFORE, recorded by this launcher as well as the wrapper -----
$gpuPre = @(Get-GpuComputeApps | ForEach-Object { [string]$_ }) -join ' | '
Write-Host "[pre ] GPU compute apps: $gpuPre"

. $verdictLib
# BEFORE Start-Process: an immediate read would otherwise see the previous run's
# marker and verdict and report it as this run's.
Clear-ArmVerdict -Dir $verdictDir -Tag $Tag

Write-Host "[run ] starting mutex wrapper..."
# The runner's stderr is the ONLY place an abort reason (e.g. 0xC0000409) shows
# up. -WindowStyle Hidden discarded it into an unreadable window and left the
# verdict's `log` field empty, so an arm that died could not be diagnosed.
# -NoNewWindow + explicit redirects keep it on disk. (measured 2026-10-05)
$errLog = Join-Path $outDir 'runner.err.log'
$proc = Start-Process -FilePath 'powershell' `
                      -NoNewWindow -PassThru `
                      -RedirectStandardOutput $log `
                      -RedirectStandardError  $errLog `
                      -ArgumentList $lockArgs
Write-Host ("[run ] wrapper pid {0}; waiting up to {1}s" -f $proc.Id, $TimeoutSeconds)

$verdict = Wait-ArmVerdict -Dir $verdictDir -Tag $Tag -TimeoutSeconds $TimeoutSeconds
$exit = if ($null -ne $verdict) { $verdict.exit_code } else { -1 }

$gpuPost = @(Get-GpuComputeApps | ForEach-Object { [string]$_ }) -join ' | '
Write-Host "[post] GPU compute apps: $gpuPost"
if ($null -ne $verdict) {
    # Read-ArmVerdict returns the RAW parsed verdict JSON: the verdict word is
    # `status`, and the GPU probes live under `fields`. Reading .verdict /
    # .gpu_pre_run / .gpu_post_run printed EMPTY in every arm (measured 2026-10-05
    # on both the treatment and the control) -- a post-run report that looks like it
    # reported something and did not. Read the real paths.
    $vf = $verdict.fields
    Write-Host ("[post] mutex verdict : {0}" -f $verdict.status)
    Write-Host ("[post] gpu_pre_run   : {0}" -f $vf.gpu_pre_run)
    Write-Host ("[post] gpu_post_run  : {0}" -f $vf.gpu_post_run)
    Write-Host ("[post] runner exit   : {0}" -f $exit)
} else {
    Write-Host '[post] NO VERDICT -- the wrapper did not finish. Treat this arm as invalid, not as a clean run.'
}

if (-not (Test-Path -LiteralPath $tel)) {
    Write-Host ("[post] telemetry missing: {0}" -f $tel)
    exit 1
}

$checkExit = 0
if ($SkipChecker) {
    Write-Host '[check] skipped by -SkipChecker'
} else {
    Write-Host '[check] demand ledger:'
    & python $checker $tel
    $checkExit = $LASTEXITCODE
}

Write-Host ''
if ($exit -ne 0) {
    Write-Host ("ARM FAILED (runner exit {0}); the ledger result above is reported but not a pass." -f $exit)
    exit $exit
}
Write-Host ("ARM RAN (runner exit 0). Ledger checker exit {0} -- 0 closed, 1 skew, 2 nothing to evaluate." -f $checkExit)
exit $checkExit
