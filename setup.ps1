# setup.ps1 - poormansllama bootstrap + setup wizard (Windows)
#
# One-liner (no local clone needed):
#   iex (irm 'https://raw.githubusercontent.com/gazbg/poormansllama/claude/flint-support/setup.ps1')
#
# What it does: checks the environment (git, CMake, CUDA, MSVC, NVIDIA GPU),
# offers to install missing tools via winget, clones the repository if not
# already inside one, then asks ALL settings up front (model + quant, local
# path, VRAM target, context; SDD D20) - the unmeasured-combo refusal (exit 3)
# happens BEFORE any build or download. The build and the model download then
# run IN PARALLEL as two legs with one combined status line, fail-fast. It
# emits a ready-to-run run-server.ps1 built from a MEASURED recipe (no
# unmeasured flag combos).
#
# Mirror for Linux: setup.sh (keep behavior in sync; see docs/SETUP.md).
#
# Side effects (each behind confirmation or an explicit flag): winget/apt
# installs, git clone into $CloneDir, cmake configure+build in the clone,
# model download into $ModelDir, run-server.ps1 written to the repo root.
# Exit codes: 0 ok, 1 usage, 2 environment, 3 no measured recipe,
# 4 build failure, 5 download failure, 6 user abort.
#
# PowerShell parameter names cannot contain hyphens, so the flags are single-
# dash (-DryRun, -NonInteractive; the double leading dash --DryRun also works).
# The Linux mirror (setup.sh) uses conventional dashed names.
param(
    [switch]$DryRun,
    [switch]$NonInteractive,
    [string]$Model = '',            # qwen38 | qwen36 | flint | apex35 | 1-7
    [string]$Quant = '',            # q4k_m | q5k_xl | q8k_xl | q5k_m (SDD D19, D23); default = the model's row quant
    [ValidateSet('full', '16g', '8g')][string]$Vram = '',
    [ValidateSet('262k', '128k')][string]$Context = '',
    [string]$Port = '',            # '' = ask (default 8090); an explicit value suppresses the question (D22)
    [string]$Bind = '',             # default: per-recipe (127.0.0.1 or 0.0.0.0)
    [int]$Device = -1,           # CUDA index for device-pinned rows (R5/R7); -1 = auto (SDD D17)
    [string]$ModelDir = '',
    [string]$CloneDir = '',
    [string]$Branch = 'claude/flint-support',
    [string]$RepoUrl = 'https://github.com/gazbg/poormansllama.git',
    [switch]$FullClone,
    [switch]$FullBuild,             # force rebuild even if the binary is fresh
    [switch]$SkipBuild,
    [switch]$SkipModel,             # stop after clone+build (no model/recipe)
    [switch]$SkipDownload,
    [switch]$YesInstall,            # allow auto-install in non-interactive mode
    [switch]$Start,
    [switch]$NoStart,
    [string]$HfToken = $env:HF_TOKEN,
    [string]$ModelFile = '',        # full path to a local gguf (overrides the table file)
    [string]$MmprojFile = '',       # full path to a local multimodal projector gguf (VOD-6; implies vision)
    [switch]$NoVision,              # VOD-6: no vision support (no projector download, no --mmproj flags)
    [int]$MaxDownloadMib = 0        # 0 = full file; test hook, see docs/SETUP.md
)

# --- dashed-arg normalization (SDD D19) ----------------------------------
# PS parameter names cannot contain hyphens, so Linux-style dashed flags
# (--dry-run, --non-interactive, --skip-model, ...) arrive unbound in $args.
# PowerShell DROPS them silently - a '--dry-run' invocation then runs for
# real (measured 2026-11-05: it started downloading the model). Normalize
# the known dashed names here; anything unrecognized fails loudly.
if ($args.Count -gt 0) {
    $rest = @()
    for ($i = 0; $i -lt $args.Count; $i++) {
        $a = $args[$i]
        switch -Regex ($a) {
            '^--?dry-run$'          { $DryRun = $true }
            '^--?non-interactive$'  { $NonInteractive = $true }
            '^--?skip-model$'       { $SkipModel = $true }
            '^--?skip-build$'       { $SkipBuild = $true }
            '^--?skip-download$'    { $SkipDownload = $true }
            '^--?full-clone$'       { $FullClone = $true }
            '^--?full-build$'       { $FullBuild = $true }
            '^--?yes-install$'      { $YesInstall = $true }
            '^--?no-vision$'        { $NoVision = $true }
            '^--?no-start$'         { $NoStart = $true }
            '^--?start$'            { $Start = $true }
            default {
                $wantVal = $a -match '^--?(quant|model|vram|context|port|host|device|model-file|mmproj-file|model-dir|clone-dir|branch|repo-url|hf-token|max-download-mib)$'
                if (-not $wantVal) { $rest += $a; break }
                if ($i + 1 -ge $args.Count) { Write-Host "  [FAIL] missing value for $a"; exit 1 }
                $v = $args[++$i]
                switch -Regex ($a) {
                    '^--?quant$'            { $Quant = $v }
                    '^--?model$'            { $Model = $v }
                    '^--?vram$'             { $Vram = $v }
                    '^--?context$'          { $Context = $v }
                    '^--?port$'             { $Port = $v }
                    '^--?host$'             { $Bind = $v }
                    '^--?device$'           { $Device = [int]$v }
                    '^--?model-file$'       { $ModelFile = $v }
                    '^--?mmproj-file$'      { $MmprojFile = $v }
                    '^--?model-dir$'        { $ModelDir = $v }
                    '^--?clone-dir$'        { $CloneDir = $v }
                    '^--?branch$'           { $Branch = $v }
                    '^--?repo-url$'         { $RepoUrl = $v }
                    '^--?hf-token$'         { $HfToken = $v }
                    '^--?max-download-mib$' { $MaxDownloadMib = [int]$v }
                }
            }
        }
    }
    if ($rest.Count -gt 0) { Write-Host "  [FAIL] unrecognized argument(s): $($rest -join ' ') (PS parameters are single-dash, e.g. -DryRun)"; exit 1 }
}

# --- state -------------------------------------------------------------
$ErrorActionPreference = 'Stop'
$NI = $NonInteractive
$DR = $DryRun
$script:AbortExit = 6

trap {
    # D20: a Ctrl-C in the parallel phase must not leave the build .bat running
    if ($script:BUILD_LEG_ACTIVE) {
        try { Stop-BuildLeg -leg $script:BUILD_LEG_ACTIVE } catch { }
    }
    $msg = if ($_.Exception) { $_.Exception.Message } else { "$_" }
    Write-Host ''
    if ($msg -match 'cancel|break') {
        Write-Host '  [aborted by user]' -ForegroundColor Red
        exit 6
    }
    Write-Host "  [error] $msg" -ForegroundColor Red
    exit 1
}

$script:DryNote = ''
function Say([string]$m)      { Write-Host "  $m" }

function Invoke-Native {
    # Run a native command with BOTH streams captured to files and return its
    # exit code. Measured on PS 5.1 (2026-11-03/04, AC-5 attempts):
    #  - '2>&1' piped under $ErrorActionPreference='Stop' terminates the host
    #    on the first stderr line (killed the wizard mid-clone three times);
    #  - a 'SilentlyContinue' guard survives but silently drops EVERY stderr
    #    line, leaving incomplete build logs;
    #  - 'Start-Process' WITHOUT -Wait never populates ExitCode (null even
    #    after WaitForExit/Refresh); the -Wait form does, with both streams
    #    redirected to files (exit 9 captured in the measurement).
    # Cost accepted: no live console output; the log paths are printed and the
    # last lines are shown on failure.
    param([string]$Exe, [string[]]$ArgList, [string]$OutLog, [string]$ErrLog)
    Say ("$Exe " + ($ArgList -join ' ') + "  (logs: $OutLog / $ErrLog)")
    $p = Start-Process -FilePath $Exe -ArgumentList $ArgList -Wait -NoNewWindow -PassThru `
         -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog
    return $p.ExitCode
}

function Invoke-NativeCmd {
    # Same as Invoke-Native, but for commands whose descendants outlive the
    # command itself (cmake -> MSBuild nodes, vctip.exe). Measured 2026-11-03:
    #  - 'Start-Process -Wait' + redirected stdout does NOT return until the
    #    LAST process holding that redirected handle exits: the wizard sat
    #    ~15 min after a finished build (31 MSBuild nodes) and again after a
    #    finished configure (vctip.exe, a long-lived VS daemon; killing it
    #    unblocked the wait instantly);
    #  - a PowerShell-level '> $file' redirect on a native call has the SAME
    #    block (cap-test17: waited 9.1 s for a 10 s grandchild);
    #  - passing the command string via 'cmd /s /c <string>' is mangled by
    #    PowerShell's native-argument re-quoting (inner '"' arrive as '\"',
    #    cmd ran the child INTERACTively; cap-test19), so the command line is
    #    written to a temp .bat file instead - no argument layer at all;
    #  - the redirections live INSIDE the .bat, so no redirected handle is
    #    ever created by PowerShell and long-lived grandchildren cannot block
    #    the wait (cap-test15/16: 0.0 s return with a 12 s grandchild alive);
    #  - the .bat has TWO lines (not 'line & echo %ERRORLEVEL%'): %ERRORLEVEL%
    #    is expanded when the PHYSICAL LINE is parsed, so on one line it would
    #    capture the value from before the target ran (measured: rc=0 for a
    #    child that exited 9); as a second line it parses after the target
    #    exits and carries its exit code.
    # The target's streams go to files inside the .bat, so nothing reaches
    # the EAP=Stop host as an ErrorRecord.
    #
    # Two more measured traps (2026-11-03):
    #  - the batch file MUST carry a .bat name: '& <file>' on a .tmp name is a
    #    SILENT NO-OP (no output, no error, no exit code) - the body never runs
    #    and no rc file appears, which reads like a missing exit code;
    #  - '@echo off' first line: without it, cmd inherits the console's
    #    PROMPT and echoes '<prompt><command>' into the redirected logs.
    param([string]$Exe, [string[]]$ArgList, [string]$OutLog, [string]$ErrLog)
    Say ("$Exe " + ($ArgList -join ' ') + "  (logs: $OutLog / $ErrLog)")
    $rcFile  = Join-Path (Split-Path $OutLog -Parent) 'setup-wizard-rc.tmp'
    $batFile = Join-Path (Split-Path $OutLog -Parent) 'setup-wizard-cmd.bat'
    Remove-Item $rcFile, $batFile, $OutLog, $ErrLog -Force -ErrorAction SilentlyContinue
    # cmd's quoting convention: wrap in double quotes, double any inner
    # quotes (backslash-escaping does not exist in cmd).
    $qArg = { param($s) if ($s -match '\s' -or ($s -match '["&<>|,]')) { '"' + ($s -replace '"','""') + '"' } else { $s } }
    $argStr = ($ArgList | ForEach-Object { & $qArg ([string]$_) }) -join ' '
    if ($Exe -match '\s' -or ($Exe -match '"')) { $exeQ = '"' + ($Exe -replace '"','""') + '"' } else { $exeQ = $Exe }
    # 'call' when the target is a batch file: a bat launched WITHOUT call
    # does not return - an 'exit /b N' inside it terminates the whole cmd
    # (measured: direct form dies, 'call' form continues with ERRORLEVEL=N),
    # which would swallow the rc-capturing line below.
    $callPrefix = ''
    if ($Exe -match '\.(bat|cmd)$') { $callPrefix = 'call ' }
    $lines = @(
        '@echo off',
        "$callPrefix$exeQ $argStr > `"$OutLog`" 2> `"$ErrLog`"",
        "echo %ERRORLEVEL% > `"$rcFile`""
    )
    [System.IO.File]::WriteAllLines($batFile, $lines, [System.Text.Encoding]::Default)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $batFile } finally { $ErrorActionPreference = $prev }
    Remove-Item $batFile -Force -ErrorAction SilentlyContinue
    if (-not (Test-Path $rcFile)) { ExitCode 4 "target did not report an exit code (logs: $OutLog / $ErrLog)" }
    $rc = [int](Get-Content $rcFile | Select-Object -First 1)
    Remove-Item $rcFile -Force -ErrorAction SilentlyContinue
    return $rc
}
function Step([string]$m)     { Write-Host ''; Write-Host "== $m" -ForegroundColor Cyan }
function Ok([string]$m)       { Write-Host "  [ok]   $m" -ForegroundColor Green }
function Warn([string]$m)     { Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Fail([string]$m)     { Write-Host "  [FAIL] $m" -ForegroundColor Red }
function ExitCode([int]$c, [string]$m) { Fail $m; exit $c }

function Confirm([string]$q, [bool]$defaultYes) {
    if ($NI) { return $defaultYes }
    $s = 'n'
    if ($defaultYes) { $s = 'Y' }
    $a = Read-Host "  $q [$s]"
    if (-not $a) { return $defaultYes }
    return $a -match '^(y|Y)'
}
function Ask([string]$q, [string]$def) {
    if ($NI) { return $def }
    $a = Read-Host "  $q"
    if (-not $a) { return $def }
    return $a
}
function AskChoice([string]$q, [object[]]$opts, [string]$defKey) {
    for ($i = 0; $i -lt $opts.Count; $i++) {
        $o = $opts[$i]
        $mark = ''
        if ($o.Key -eq $defKey) { $mark = '  (default)' }
        $lbl = $o.Label
        Write-Host "    $($i + 1)) $lbl$mark"
    }
    $sel = ''
    while (-not $sel) {
        $a = if ($NI) { $defKey } else { (Read-Host "  $q [1-$($opts.Count)]").Trim() }
        if (-not $a) { $sel = $defKey; break }
        if ($a -match '^\d+$' -and [int]$a -ge 1 -and [int]$a -le $opts.Count) { $sel = $opts[[int]$a - 1].Key; break }
        if ($a -match '^[a-z0-9_-]+$' -and @($opts | ForEach-Object { $_.Key }) -contains $a) { $sel = $a; break }
        if (-not $NI) { Say "  pick a number 1-$($opts.Count)" } else { $sel = $defKey; break }
    }
    return $sel
}

# --- data tables -------------------------------------------------------
# Verified against the Hugging Face API on 2026-11-04 (Content-Length of the
# resolve endpoint). Sizes are GiB to one decimal.
# SDD D19: the table is flat over (model, quant) entries; the entry marked
# Default is the row's own file (the one the recipe rows were measured on).
$script:MODELS = @(
    [pscustomobject]@{ Key='qwen38'; Quant='q4k_m'; Label='Qwen3.8-27B dense hybrid (unsloth UD Q4_K_M)'
        Repo='unsloth/Qwen3.8-27B-GGUF'; File='Qwen3.8-27B-UD-Q4_K_M.gguf'; SizeGib=15.4
        Mmproj='mmproj-BF16.gguf'; MmprojBytes=931146432
        DefaultDir='C:/Models/ALL/other/qwen-3.8'; Kind='dense'; Default=$true },
    [pscustomobject]@{ Key='qwen36'; Quant='q4k_m'; Label='Qwen3.6-35B-A3B MoE (unsloth UD Q4_K_M)'
        Repo='unsloth/Qwen3.6-35B-A3B-GGUF'; File='Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'; SizeGib=20.6
        Mmproj='mmproj-BF16.gguf'; MmprojBytes=902822624
        DefaultDir='C:/Models/ALL/other/qwen-3.6'; Kind='moe'; Default=$true },
    [pscustomobject]@{ Key='flint'; Quant='q5k_xl'; Label='Flint SWA MoE (unsloth UD Q5_K_XL)'
        Repo='unsloth/flint-GGUF'; File='flint-UD-Q5_K_XL.gguf'; SizeGib=27.9
        DefaultDir='C:/Models/ALL/other/flint'; Kind='moe-swa'; Default=$true },
    [pscustomobject]@{ Key='qwen38'; Quant='q5k_xl'; Label='Qwen3.8-27B dense hybrid (unsloth UD Q5_K_XL)'
        Repo='unsloth/Qwen3.8-27B-GGUF'; File='Qwen3.8-27B-UD-Q5_K_XL.gguf'; SizeGib=19.4
        Mmproj='mmproj-BF16.gguf'; MmprojBytes=931146432
        DefaultDir='C:/Models/ALL/other/qwen-3.8'; Kind='dense'; Default=$false },
    [pscustomobject]@{ Key='qwen36'; Quant='q5k_xl'; Label='Qwen3.6-35B-A3B MoE (unsloth UD Q5_K_XL)'
        Repo='unsloth/Qwen3.6-35B-A3B-GGUF'; File='Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf'; SizeGib=24.7
        Mmproj='mmproj-BF16.gguf'; MmprojBytes=902822624
        DefaultDir='C:/Models/ALL/other/qwen-3.6'; Kind='moe'; Default=$false },
    [pscustomobject]@{ Key='qwen36'; Quant='q8k_xl'; Label='Qwen3.6-35B-A3B MoE + MTP (unsloth UD Q8_K_XL, MTP repo)'
        Repo='unsloth/Qwen3.6-35B-A3B-MTP-GGUF'; File='Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf'; SizeGib=36.4
        Mmproj='mmproj-BF16.gguf'; MmprojBytes=902822528
        DefaultDir='C:/Models/ALL/other/qwen-3.6'; Kind='moe-mtp'; Default=$false },
    # SDD D23 (2026-11-09): Apex-35B-V1.0 (qwen35moe, 41 blocks incl. 1 MTP layer). Sizes and the
    # resolve-endpoint Content-Length checked against the Hugging Face API on 2026-11-09
    # (25,347,532,640 bytes for Q5_K_M). The Q6_K_L build was offered as entry 8 and removed on 2026-11-09.
    [pscustomobject]@{ Key='apex35'; Quant='q5k_m'; Label='Apex-35B-V1.0 MoE + MTP (Apexintelligence-AI Q5_K_M)'
        Repo='Apexintelligence-AI/Apex-35B-V1.0-GGUF'; File='Apex-35B-V1.0-Q5_K_M.gguf'; SizeGib=23.6
        MmprojRepo='bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF'; Mmproj='mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf'; MmprojBytes=902822560
        DefaultDir='C:/Models/ALL/other/apex'; Kind='moe-mtp'; Default=$true }
)

# Recipe rows: MEASURED, verbatim (SDD section 4.6, decision D4/D11).
# SDD D21: a row is keyed on (model, VRAM, context, quant). A quant with no row
# of its own falls back to the model's default-quant row (Check-RecipeQuant warns).
# Flags = the exact command from the source run, minus the exe path, -m, and
# the host/port the wizard re-asks (defaults keep the row's own values).
$script:RECIPES = @(
    [pscustomobject]@{ Id='R1'; Model='qwen38'; Vram='full'; Ctx='262k'; Quant='q4k_m'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-37 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        Bind='127.0.0.1'; Measured='PP 1093.25 t/s, TG 14.95 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        Env=@{ 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1' } },
    [pscustomobject]@{ Id='R2'; Model='qwen38'; Vram='16g'; Ctx='262k'; Quant='q4k_m'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        Bind='127.0.0.1'; Measured='PP 1007.87 t/s, TG 9.96 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        Env=@{ 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1' } },
    [pscustomobject]@{ Id='R3'; Model='qwen38'; Vram='8g'; Ctx='128k'; Quant='q4k_m'
        Flags='-c 128000 -ctk q4_0 -ctv q4_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --lm-prefill-prearm always -lv 4'
        Bind='127.0.0.1'; Measured='PP 36.95 t/s, TG 9.92 t/s (owner 2026-10-29, RTX 5090 24 GiB)'
        Env=@{ 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1' } },
    [pscustomobject]@{ Id='R4'; Model='flint'; Vram='full'; Ctx='262k'; Quant='q5k_xl'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --lm-prefill-copy-piece 64'
        Bind='0.0.0.0'; Measured='PP 2422.94 t/s, TG 53.80 t/s (owner 2026-11-02, RTX 5090 24 GiB)'
        Env=@{ 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1' } },
    [pscustomobject]@{ Id='R5'; Model='qwen36'; Vram='full'; Ctx='262k'; Quant='q8k_xl'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 65536 --lm-prefill-ubatch 2048 --lm-prefill-segment 65536 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse --no-warmup'
        Device='auto'; Bind='0.0.0.0'; Measured='PP 3600 t/s @50k, 2350 t/s @200k; TG 77 t/s @50k, 66 t/s @200k (owner, GTX 5090 Laptop, Windows)'
        Env=@{ 'LLAMA_MOE_HYBRID_LEAN_STALL_US' = '10000'; 'LLAMA_FATTN_GQA8_MAXROWS' = '4'; 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1'; 'LLAMA_FATTN_TILE_Q8QK' = '1' }
        Note='needs the MTP build of this quant (unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Measured on Windows with one MoE model only; Linux and other models are not measured with these flags' },
    [pscustomobject]@{ Id='R6'; Model='qwen36'; Vram='16g'; Ctx='262k'; Quant='q8k_xl'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 8 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse'
        Device='auto'; Bind='0.0.0.0'; Measured='PP 2000 t/s, TG 51 t/s @200k (owner, GTX 5090 Laptop, Windows)'
        Env=@{ 'LLAMA_MOE_HYBRID_LEAN_STALL_US' = '10000'; 'LLAMA_FATTN_GQA8_MAXROWS' = '4'; 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1'; 'LLAMA_FATTN_TILE_Q8QK' = '1' }
        Note='needs the MTP build of this quant (unsloth/Qwen3.6-35B-A3B-MTP-GGUF). Measured on Windows with one MoE model only; Linux and other models are not measured with these flags' },
    [pscustomobject]@{ Id='R7'; Model='qwen36'; Vram='8g'; Ctx='262k'; Quant='q4k_m'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 32768 -ub 512 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 16384'
        Device='auto'; Bind='0.0.0.0'; Measured='PP 590 t/s, TG 33 t/s @20k (owner, GTX 1070 8 GiB, Windows)'
        Env=@{ 'LLAMA_MOE_HYBRID_LEAN_STALL_US' = '10000'; 'LLAMA_FATTN_GQA8_MAXROWS' = '4'; 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1'; 'LLAMA_FATTN_TILE_Q8QK' = '1' }
        Note='measured at 20k context on an 8 GiB card, Windows, one MoE model only; the row is 262k context as the owner ran the command' },
    [pscustomobject]@{ Id='R8'; Model='qwen36'; Vram='16g'; Ctx='262k'; Quant='q4k_m'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32'
        Bind='127.0.0.1'; Measured='G42 canonical: TG 3.0-3.7x stock, PP 3.65-4.11x stock (docs/guide/USAGE.md, build s53)'
        Env=@{ 'LLAMA_FATTN_GQA8' = '1'; 'LLAMA_CUDA_INPUT_BATCH' = '1' }
        Note='measured on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S; the wizard offers the closest HF file, Q4_K_M (SDD D13). Kept as R8 (the former R6)' },
    # SDD D23: the owner's running command for Apex-35B (the R5 shape, MTP head used). Measured by this
    # repo on 2026-11-09 with Apex-35B-V1.0-Q5_K_S.gguf - a file the wizard does not offer. No row env:
    # the measured runs set none.
    [pscustomobject]@{ Id='R9'; Model='apex35'; Vram='full'; Ctx='262k'; Quant='q5k_m'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 65536 --lm-prefill-ubatch 2048 --lm-prefill-segment 65536 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all --spec-type draft-mtp --spec-draft-p-min 0.1 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 --mtp-draft-vocab 32768 --lm-prefill-fuse --no-warmup'
        Device='auto'; Bind='0.0.0.0'; Measured='TG ~100-110 tok/s (600-token greedy reply), PP ~3,900 t/s on a 20.7k-token prompt (2026-11-09, RTX 5090 Laptop 24 GiB, Windows, on Apex-35B-V1.0-Q5_K_S.gguf)'
        Note='measured on Apex-35B-V1.0-Q5_K_S.gguf, NOT on the offered Q5_K_M (same architecture qwen35moe, 41 blocks with 1 MTP layer; the offered file carries the blk.40.nextn tensors, GGUF header read 2026-11-09). Windows only, one MoE family' }
)

function DefaultQuant([string]$m) {
    $d = $script:MODELS | Where-Object { $_.Key -eq $m -and $_.Default } | Select-Object -First 1
    if ($d) { return $d.Quant }
    return ''
}

function IsMoeModel([string]$m) {
    return [bool]($script:MODELS | Where-Object { $_.Key -eq $m -and $_.Kind -like 'moe*' } | Select-Object -First 1)
}

function MoeClassRow([string]$v, [string]$c) {
    # SDD D24: the 8g profile is shared by EVERY MoE model, whatever its name and whether or not it
    # was benchmarked: it is the owner's 8 GiB GTX 1070 row (R7, qwen36 q4k_m). Other VRAM targets
    # keep their per-model rows.
    if ($v -ne '8g') { return $null }
    return ($script:RECIPES | Where-Object { $_.Model -eq 'qwen36' -and $_.Vram -eq '8g' -and $_.Ctx -eq $c -and $_.Quant -eq 'q4k_m' } | Select-Object -First 1)
}

function RecipeFor([string]$m, [string]$v, [string]$c, [string]$q) {
    # SDD D21: an exact (model, VRAM, ctx, quant) row first; otherwise the model's
    # default-quant row (the caller warns that it was measured on another file)
    $hit = $script:RECIPES | Where-Object { $_.Model -eq $m -and $_.Vram -eq $v -and $_.Ctx -eq $c -and $_.Quant -eq $q } | Select-Object -First 1
    if ($hit) { return $hit }
    $dq = DefaultQuant $m
    if ($dq -and $dq -ne $q) {
        $hit = $script:RECIPES | Where-Object { $_.Model -eq $m -and $_.Vram -eq $v -and $_.Ctx -eq $c -and $_.Quant -eq $dq } | Select-Object -First 1
        if ($hit) { return $hit }
    }
    # SDD D24: a MoE model with no row of its own at 8g takes the shared 8g MoE profile
    if (IsMoeModel $m) { return (MoeClassRow $v $c) }
    return $null
}

function Get-NvidiaSmiPath {
    $p = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
    if (-not $p -and (Test-Path 'C:\Windows\System32\nvidia-smi.exe')) { $p = 'C:\Windows\System32\nvidia-smi.exe' }
    return $p
}

function Select-GpuIndex {
    param([string[]]$Lines, [string]$VramTarget)
    # SDD D17 (pure): pick the CUDA index from parsed 'index, <MiB>' rows.
    # 'full' -> the largest card; '16g'/'8g' -> the smallest card at or above
    # the row minimum (10240/8192 MiB), else the smallest card (the D12
    # warning then fires). Returns [pscustomobject]@{ Index; Why; Count }.
    $needMib = 0
    if ($VramTarget -eq '16g') { $needMib = 10240 }
    elseif ($VramTarget -eq '8g') { $needMib = 8192 }
    $largest = -1; $largestMib = -1
    $smallest = -1; $smallestMib = [long]::MaxValue
    $fit = -1; $fitMib = [long]::MaxValue
    $count = 0
    foreach ($line in $Lines) {
        if ($line -match '^\s*(\d+)\s*,\s*([\d,]+)\s*MiB') {
            $idx = [int]$Matches[1]; $mib = [long]($Matches[2].Replace(',', ''))
            $count++
            if ($largest -lt 0 -or $mib -gt $largestMib -or ($mib -eq $largestMib -and $idx -lt $largest)) { $largest = $idx; $largestMib = $mib }
            if ($smallest -lt 0 -or $mib -lt $smallestMib -or ($mib -eq $smallestMib -and $idx -lt $smallest)) { $smallest = $idx; $smallestMib = $mib }
            if ($needMib -gt 0 -and $mib -ge $needMib -and ($fit -lt 0 -or $mib -lt $fitMib -or ($mib -eq $fitMib -and $idx -lt $fit))) { $fit = $idx; $fitMib = $mib }
        }
    }
    if ($count -eq 0) { return [pscustomobject]@{ Index = 0; Why = 'no rows parsed'; Count = 0 } }
    if ($count -gt 1) {
        if ($VramTarget -eq 'full') { $pick = $largest; $why = "largest card, ${largestMib} MiB" }
        elseif ($fit -ge 0) { $pick = $fit; $why = "smallest card >= $needMib MiB, ${fitMib} MiB" }
        else { $pick = $smallest; $why = "smallest card, ${smallestMib} MiB (below the row minimum)" }
    }
    else { $pick = $largest; $why = "single card, ${largestMib} MiB" }
    return [pscustomobject]@{ Index = $pick; Why = $why; Count = $count }
}

function Resolve-GpuIndex {
    param([string]$VramTarget)
    # SDD D17: CUDA index for device-pinned rows (R5/R7). -Device N wins.
    # Auto via Select-GpuIndex. Result is cached in $script:GpuDev.
    if ($null -ne $script:GpuDev) { return $script:GpuDev }
    if ($Device -ge 0) { $script:GpuDev = $Device; return $Device }
    $ns = Get-NvidiaSmiPath
    $lines = @()
    if ($ns -and (Test-Path $ns)) { $lines = @(& $ns --query-gpu=index,memory.total --format=csv,noheader 2>$null) }
    $sel = Select-GpuIndex -Lines $lines -VramTarget $VramTarget
    if ($sel.Count -eq 0) { Warn 'nvidia-smi failed; pinning --device CUDA0 (override with -Device N)' }
    if ($sel.Count -gt 1) { Say "multi-GPU system: pinning --device CUDA$($sel.Index) ($($sel.Why); override with -Device N)" }
    $script:GpuDev = $sel.Index
    return $sel.Index
}

# --- row-flag completion (D22) --------------------------------------------
# The wizard's one place that changes what a row emits, beyond the D11/D17
# substitutions (model path, bind, port, device index). Pure over the row:
# the same (flags, env, compute cap) always gives the same output, so the
# same-day byte-identical re-run (SDD AC-2) still holds.
function Detect-GpuArch {
    # D22: compute capability of the card the run will use (nvidia-smi's
    # first line, the D12/D17 convention) as 'M.m'. SETUP_WIZARD_GPU_CC
    # overrides for testing (SETUP_WIZARD_FORCE_MISSING is the precedent).
    # '' = unavailable; the caller then emits the row as measured.
    if ($env:SETUP_WIZARD_GPU_CC) { return $env:SETUP_WIZARD_GPU_CC }
    $ns = Get-NvidiaSmiPath
    if (-not $ns -or -not (Test-Path $ns)) { return '' }
    $cap = (& $ns --query-gpu=compute_cap --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($cap -match '^(\d+)\.(\d+)$') { return $Matches[0] }
    return ''
}

function Complete-RowFlags {
    param([string]$Flags, $EnvText, [string]$ComputeCap, [bool]$Vision = $false)
    # VOD-6 (docs/plans/vision-on-demand-plan.md): with $Vision the vision flags are appended,
    # ' --mmproj $mmproj --mmproj-on-demand', where $mmproj is the variable the emitted script
    # defines next to $model (so a path with spaces needs no quoting here). Exactly once; the row's
    # own flags never carry --mmproj (Test-VisionOffered refuses a row that does).
    # D22-B: every profile emits --lm-prefill-prearm always. An existing
    # occurrence (R3's 'always', R8's 'auto') is REPLACED, otherwise the
    # flag is appended - two occurrences would be a parse hazard.
    # D22-A: on Pascal and lower (compute cap <= 6.1) the pass's flash
    # attention takes the int8 Q.K^T tile (the flag is literally
    # setenv(LLAMA_FATTN_TILE_Q8QK,1) in the server, help: 'Same as
    # LLAMA_FATTN_TILE_Q8QK=1', target 'a GPU without fast FP16, such as
    # Pascal'). If the row's env already sets that env, the env is the
    # emitted config - it stays, and nothing is appended; otherwise the
    # flag is appended. AN APPROXIMATION (int8 Q per 32 values): the
    # caller prints a D22 note when the flag was added.
    # Returns [pscustomobject]@{ Flags; AddedInt8 }.
    $f = $Flags
    if ($f -match '(?<!-)--lm-prefill-prearm\s+(?:off|always|auto)') {
        $f = $f -replace '(?<!-)--lm-prefill-prearm\s+(?:off|always|auto)', '--lm-prefill-prearm always'
    } else {
        $f = $f + ' --lm-prefill-prearm always'
    }
    $addedInt8 = $false
    if ($ComputeCap -match '^([0-9]+)\.([0-9]+)$') {
        $maj = [int]$Matches[1]; $min = [int]$Matches[2]
        if ($maj -lt 6 -or ($maj -eq 6 -and $min -le 1)) {
            $envHas = $false
            if ($EnvText) {
                if ($EnvText -is [hashtable] -or $EnvText -is [System.Collections.Specialized.OrderedDictionary]) {
                    $envHas = $EnvText.ContainsKey('LLAMA_FATTN_TILE_Q8QK') -and ([string]$EnvText['LLAMA_FATTN_TILE_Q8QK']) -eq '1'
                } else {
                    $envHas = (([string]$EnvText) -split '\s+') -contains 'LLAMA_FATTN_TILE_Q8QK=1'
                }
            }
            if (-not $envHas -and -not (@($f -split '\s+') -contains '--lm-prefill-attn-int8')) {
                $f = $f + ' --lm-prefill-attn-int8'; $addedInt8 = $true
            }
            else { Say "row env already sets LLAMA_FATTN_TILE_Q8QK=1 (compute capability $ComputeCap); the env is the emitted config, no flag added" }
        }
    }
    if ($Vision -and $f -notmatch '(?<!-)--mmproj-on-demand') {
        $f = $f + ' --mmproj $mmproj --mmproj-on-demand'
    }
    return [pscustomobject]@{ Flags = $f; AddedInt8 = $addedInt8 }
}

# VOD-6 / SDD D27: is vision offered for this model row and recipe? The table row must list a projector
# (Mmproj; from the model's own Hugging Face repo, or from MmprojRepo when that repo publishes none), and the
# recipe must not carry --mmproj itself. The MTP / draft rows (--spec-type) are included since D27: the
# owner's own Q8_K_XL MTP command ran with a projector (5 of 5 graded fixtures). Returns $true / $false (pure).
function Test-VisionOffered {
    param([object]$m, [object]$r)
    if (-not $m.PSObject.Properties['Mmproj'] -or -not $m.Mmproj) { return $false }
    if ($r.Flags -match '(?<!-)--mmproj\b') { return $false }
    return $true
}

# SDD D28: does a flag string already set the per-response token limit (-n / --n-predict / --predict)? Pure.
function Test-HasPredictFlag {
    param([string]$Flags)
    return [bool]($Flags -match '(^|\s)(-n|--n-predict|--predict)(\s|=)')
}

# --- environment check (SDD 4.1) ----------------------------------------
$script:EnvTools = @()

function ToolRow([string]$name, [string]$state, [string]$detail) {
    $script:EnvTools += [pscustomobject]@{ Name=$name; State=$state; Detail=$detail }
}

function Test-ForceMissing([string]$name) {
    if ($env:SETUP_WIZARD_FORCE_MISSING) {
        return ($env:SETUP_WIZARD_FORCE_MISSING -split ',' | ForEach-Object { $_.Trim().ToLowerInvariant() }) -contains $name.ToLowerInvariant()
    }
    return $false
}

function Check-Environment {
    Step 'Environment check'
    $script:EnvTools = @()   # reset: re-checks must not accumulate duplicate rows
    $rows = @()

    # git
    if (Test-ForceMissing 'git') { $rows += 'MISSING' }
    else {
        $g = $null
        try { $g = (& git --version 2>$null) | Select-Object -First 1 } catch { }
        if ($g) { $rows += "ok  ($($g -replace '^git version ', ''))" }
        else { $rows += 'MISSING' }
    }

    # cmake >= 4.2
    if (Test-ForceMissing 'cmake') { $rows += 'MISSING' }
    else {
        $cm = $null
        try { $cm = (& cmake --version 2>$null) | Select-Object -First 1 } catch { }
        if ($cm) {
            if ($cm -match 'cmake version (\d+)\.(\d+)') {
                if ([int]$Matches[1] -gt 4 -or ([int]$Matches[1] -eq 4 -and [int]$Matches[2] -ge 2)) { $rows += "ok  ($($cm -replace '\r', ''))" }
                else { $rows += "TOO-OLD ($($cm -replace '\r', ''), need >= 4.2)" }
            } else { $rows += "ok  ($($cm -replace '\r', ''))" }
        } else { $rows += 'MISSING' }
    }

    # CUDA toolkit (nvcc)
    if (Test-ForceMissing 'cuda') { $rows += 'MISSING' }
    else {
        $nv = $null
        try { $nv = (& nvcc --version 2>$null) | Select-String 'release (\d+\.\d+)' } catch { }
        if (-not $nv) {
            $cand = Get-ChildItem -Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
                Where-Object { Test-Path (Join-Path $_.FullName 'bin\nvcc.exe') } |
                Sort-Object Name -Descending | Select-Object -First 1
            if ($cand) { $nv = "$($cand.Name) (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$($cand.Name))" }
        }
        if ($nv) { $rows += "ok  ($($nv -replace '\r', ''))" }
        else { $rows += 'MISSING' }
    }

    # MSVC v143 + Windows SDK (vswhere; scan ALL instances - 'latest' can be an
    # SDK-less newer BuildTools while the older one is complete, measured 2026-11-04)
    if (Test-ForceMissing 'msvc') { $rows += 'MISSING' }
    else {
        $vsroot = ${env:ProgramFiles(x86)} + '\Microsoft Visual Studio\Installer\vswhere.exe'
        $instances = @()
        if (Test-Path $vsroot) {
            try {
                $instances = & $vsroot -all -products * -property installationPath 2>$null
            } catch { }
        }
        # the Windows SDK may live under the VS instance OR at machine level
        # (C:\Program Files (x86)\Windows Kits\10\bin) - CMake accepts both.
        $machineSdk = $null
        foreach ($kroot in @("${env:ProgramFiles(x86)}\Windows Kits\10\bin", "${env:ProgramFiles}\Windows Kits\10\bin")) {
            $machineSdk = Get-ChildItem -Path $kroot -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } | Sort-Object Name -Descending | Select-Object -First 1
            if ($machineSdk) { break }
        }
        $found = $null
        foreach ($vs in $instances) {
            $cl = Get-ChildItem -Path (Join-Path $vs 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -like '14.*' } | Sort-Object Name -Descending | Select-Object -First 1
            $sdkbin = Get-ChildItem -Path (Join-Path $vs 'Windows Kits\10\bin') -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^\d+\.\d+\.\d+\.\d+$' } | Sort-Object Name -Descending | Select-Object -First 1
            if (-not $sdkbin -and $machineSdk) { $sdkbin = $machineSdk }
            if ($cl -and $sdkbin) {
                $found = "ok  (MSVC $($cl.Name), Windows SDK $($sdkbin.Name), $vs)"
                break
            }
        }
        if ($found) { $rows += $found }
        elseif ($instances -and $instances.Count -gt 0) { $rows += 'INCOMPLETE (VC tools or Windows SDK missing in every Visual Studio instance)' }
        else { $rows += 'MISSING' }
    }

    # nvidia-smi (GPU + driver)
    if (Test-ForceMissing 'nvidia-smi') { $rows += 'MISSING' }
    else {
        $ns = $null
        $nsPath = (Get-Command nvidia-smi -ErrorAction SilentlyContinue).Source
        if (-not $nsPath -and (Test-Path 'C:\Windows\System32\nvidia-smi.exe')) { $nsPath = 'C:\Windows\System32\nvidia-smi.exe' }
        if ($nsPath) {
            $q = & $nsPath --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1
            if ($q) { $rows += "ok  ($($q -replace '\r', ''))"; $script:GpuLine = $q }
            else { $rows += 'PRESENT-BUT-ERROR (nvidia-smi failed)' }
        } else { $rows += 'MISSING' }
    }

    # RAM (warn only)
    $ram = 0
    try { $ram = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 0) } catch { }

        $i = 0
    $names = @('git', 'cmake', 'cuda', 'msvc', 'nvidia-smi')
    foreach ($n in $names) {
        $s = $rows[$i]; $i++
        if ($s -match 'MISSING|TOO-OLD|INCOMPLETE|ERROR') { ToolRow $n $s '' }
        else { ToolRow $n 'OK' $s }
    }
    $ramState = 'LOW'
    if ($ram -ge 48) { $ramState = 'OK' }
    ToolRow 'ram' $ramState "$ram GiB (>= 48 GiB recommended for MoE host experts)"

    # print table
    foreach ($t in $script:EnvTools) {
        $col = 'Green'
        if ($t.State -eq 'OK') { $col = 'Green' } elseif ($t.State -eq 'LOW') { $col = 'Yellow' } else { $col = 'Red' }
        Write-Host ("    {0,-12} {1,-10} {2}" -f $t.Name, $t.State, $t.Detail) -ForegroundColor $col
    }
    if ($ram -lt 48) { Warn 'RAM below 48 GiB: MoE models keep expert banks in host RAM' }
    # @() wrap: a single missing tool would be a scalar without .Count in PS 5.1,
    # which silently skipped the install path (measured 2026-11-04, AC-4)
    $bad = @($script:EnvTools | Where-Object { $_.State -match 'MISSING|TOO-OLD|INCOMPLETE|ERROR' })
    return $bad
}

# --- auto-install (SDD 4.2, winget IDs verified 2026-11-04) --------------
$script:INSTALL_PLAN = @{
    'git'     = @('Git.Git', @())
    'cmake'   = @('Kitware.CMake', @())
    'cuda'    = @('Nvidia.CUDA', @())
    'msvc'    = @('Microsoft.VisualStudio.2022.BuildTools', @('--override', 'quiet add Microsoft.VisualStudio.Workload.VCTools --includeRecommended'))
    'nvidia-smi' = @($null, @())   # never auto-install: driver => manual guidance
}

function Install-Missing {
    param([object[]]$bad)
    $toInstall = $bad | Where-Object { $_.Name -ne 'nvidia-smi' -and $_.State -match 'MISSING|TOO-OLD|INCOMPLETE' }
    if (-not $toInstall) { return }
    Step 'Install missing tools (winget)'
    foreach ($t in $toInstall) {
        $plan = $script:INSTALL_PLAN[$t.Name]
        $pkgArgs = @('--id', $plan[0], '--accept-source-agreements', '--accept-package-agreements')
        $pkgArgs += $plan[1]
        $line = 'winget install ' + (($pkgArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' ')
        Say ("$($t.Name) MISSING -> $line")
    }
    if ($DR) { Say '(dry-run: nothing installed)'; return }
    $go = Confirm 'Install now?' $true
    if (-not $go) {
        Warn 'manual equivalents:'
        Say '  git:    https://git-scm.com/download/win'
        Say '  cmake:  https://cmake.org/download/ (or: winget install --id Kitware.CMake -e)'
        Say '  cuda:   https://developer.nvidia.com/cuda-downloads (toolkit, not driver)'
        Say '  msvc:   Visual Studio 2022 Build Tools, workload "Desktop development with C++"'
        Say '  then re-run this wizard.'
        ExitCode 2 'user declined install; missing tools remain'
    }
    if ($NI -and -not $YesInstall) { ExitCode 2 'non-interactive install requires -YesInstall' }
    foreach ($t in $toInstall) {
        $plan = $script:INSTALL_PLAN[$t.Name]
        Say "installing $($t.Name) ... (this can take several minutes)"
        $winArgs = @('install', '--id', $plan[0], '--accept-source-agreements', '--accept-package-agreements') + $plan[1]
        $wOut = Join-Path $env:TEMP ('swz-winget-' + $t.Name + '.log')
        $wErr = Join-Path $env:TEMP ('swz-winget-' + $t.Name + '.err.log')
        $wrc = Invoke-Native -Exe 'winget' -ArgList $winArgs -OutLog $wOut -ErrLog $wErr
        if ($wrc -ne 0) {
            Warn "winget reported exit $wrc for $($t.Name) (log: $wOut / $wErr)"
        }
    }
    # refresh PATH from the registry (winget installs land on the machine/user PATH)
    $mach = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$mach;$user"
    $bad2 = Check-Environment
    $still = $bad2 | Where-Object { $_.State -match 'MISSING|TOO-OLD|INCOMPLETE' -and $_.Name -ne 'nvidia-smi' }
    if ($still) {
        foreach ($s in $still) { Warn "after install: $($s.Name) still $($s.State)" }
        ExitCode 2 'a required tool is still missing after the install attempt; see the table above and the manual steps in docs/SETUP.md'
    }
    Ok 'install complete; environment re-checked'
}

# --- clone (SDD 4.3, decision D8) ---------------------------------------
function Get-RepoRoot {
    # the directory holding third_party/llama.cpp and setup.ps1
    # ($PSScriptRoot is empty when piped via iex (irm ...); then walk up from cwd)
    $here = $PSScriptRoot
    if ($here -and (Test-Path (Join-Path $here 'third_party\llama.cpp'))) {
        return (Resolve-Path $here).Path
    }
    $d = if ($here) { $here } else { (Get-Location).Path }
    while ($d) {
        if (Test-Path (Join-Path $d 'third_party\llama.cpp')) { return $d }
        $d = Split-Path -Parent $d
    }
    return $null
}

function Invoke-Clone {
    param([string]$destRoot)
    Step 'Repository'
    if (-not $destRoot) {
        # no explicit target: running inside the repo means nothing to clone
        $inRepo = Get-RepoRoot
        if ($inRepo) {
            Ok "already inside a checkout: $inRepo (clone skipped)"
            return $inRepo
        }
        $destRoot = "$env:LOCALAPPDATA\llama-setup"
        if (-not $destRoot) { $destRoot = 'C:\llama-setup' }
        Say "no -CloneDir given; using default $destRoot"
    }
    $dest = Join-Path $destRoot 'poormansllama'
    if (Test-Path $dest) {
        $items = Get-ChildItem $dest -Force -ErrorAction SilentlyContinue
        if ($items -and $items.Count -gt 0) {
            # reuse a valid existing checkout so a re-run (different model,
            # same day or later) works; anything else is still refused.
            # valid = a git repo whose origin is this wizard's repo URL.
            # (two separate Test-Path statements: on this machine's PS parser,
            # 'Test-Path (expr) -and (expr)' mis-parses -and as a parameter)
            $hasGit = Test-Path (Join-Path $dest '.git')
            $isOurs = $false
            if ($hasGit) {
                Push-Location $dest
                try {
                    $origin = [string](& git remote get-url origin 2>$null)
                } catch { $origin = [string]'' }
                Pop-Location
                if ($origin) {
                    $oNorm = ($origin -replace '\.git$', '').ToLower().TrimEnd('/')
                    $uNorm = ($RepoUrl -replace '\.git$', '').ToLower().TrimEnd('/')
                    $isOurs = ($oNorm -eq $uNorm)
                }
            }
            if ($isOurs) {
                Ok "reusing existing checkout: $dest"
                return $dest
            }
            ExitCode 1 "clone target $dest exists and is not empty (use -CloneDir to pick another)"
        }
    }
    $cloneArgs = @('clone')
    if (-not $FullClone) { $cloneArgs += '--depth', '1' }
    $cloneArgs += @('--branch', $Branch, $RepoUrl, $dest)
    if ($DR) { Say ('(dry-run: ' + ($cloneArgs -join ' ') + ')'); return $dest }
    $gLog = Join-Path $env:TEMP 'swz-clone.log'
    $gErr = Join-Path $env:TEMP 'swz-clone.err.log'
    $rc = Invoke-Native -Exe 'git' -ArgList $cloneArgs -OutLog $gLog -ErrLog $gErr
    if ($rc -ne 0 -and $Branch -ne 'main') {
        # the requested branch is not published (e.g. in-flight work): fall back
        # to the remote default branch; the flag-capability check below will
        # refuse to emit a row the resulting binary cannot run.
        Warn "clone of branch '$Branch' failed (not on the remote?); falling back to the remote default branch"
        if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
        $cloneArgs = @('clone')
        if (-not $FullClone) { $cloneArgs += '--depth', '1' }
        $cloneArgs += @($RepoUrl, $dest)
        $rc = Invoke-Native -Exe 'git' -ArgList $cloneArgs -OutLog $gLog -ErrLog $gErr
    }
    if ($rc -ne 0) {
        Get-Content $gErr -Tail 10 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        ExitCode 1 "git clone failed (exit $rc; branch: $Branch; log: $gLog / $gErr)"
    }
    # Windows: long paths for the vendored tree (USAGE.md)
    Push-Location $dest
    try { & git config core.longpaths true | Out-Null } catch { }
    Pop-Location
    Ok "cloned -> $dest (branch: $Branch or remote default)"
    return $dest
}

# --- build (SDD 4.4, decisions D9/D10) -----------------------------------
function Test-BinaryFresh {
    param([string]$repo, [string]$bin)
    if (-not (Test-Path $bin)) { return $false }
    $binTime = (Get-Item $bin).LastWriteTime
    $srcRoots = @('src', 'ggml', 'common', 'tools\llama-server')
    $newest = $null
    foreach ($r in $srcRoots) {
        $root = Join-Path $repo ("third_party\llama.cpp\$r")
        if (-not (Test-Path $root)) { continue }
        $n = Get-ChildItem -Path $root -Recurse -File -Include *.cpp,*.cu,*.cuh,*.h,*.hpp -ErrorAction SilentlyContinue |
            Measure-Object -Property LastWriteTime -Maximum | Select-Object -ExpandProperty Maximum
        if ($n -and (-not $newest -or $n -gt $newest)) { $newest = $n }
    }
    if (-not $newest) { return $true }
    return $binTime -ge $newest
}

function Invoke-Build {
    # SEQUENTIAL form (SDD D20): used for the -SkipModel-without-Model run and
    # for single-leg (build-only) phases. The parallel phase (step 5) uses
    # Start-BuildLeg + the download tick instead.
    param([string]$repo)
    Step 'Build llama-server'
    $bin = Join-Path $repo 'build\llama-cuda\bin\Release\llama-server.exe'
    if ($SkipBuild) { Say '(skipped: -SkipBuild)'; return $bin }
    if (-not $FullBuild -and (Test-BinaryFresh $repo $bin)) {
        Ok "build skipped (binary newer than sources): $bin"
        return $bin
    }
    # CUDA_PATH export for this process (USAGE.md; forward slashes in -D values)
    $cudaRoot = $env:CUDA_PATH
    if (-not $cudaRoot) {
        $cand = Get-ChildItem -Path 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($cand) { $cudaRoot = $cand.FullName }
    }
    if (-not $cudaRoot) { ExitCode 2 'CUDA toolkit not found (nvcc missing and no C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\*)' }
    $env:CUDA_PATH = $cudaRoot
    $majmin = Split-Path -Leaf $cudaRoot   # e.g. v13.2
    if ($majmin -match 'v(\d+)[._](\d+)$') {
        [Environment]::SetEnvironmentVariable("CUDA_PATH_V${Matches[1]}_${Matches[2]}", $cudaRoot, 'Process')
    }
    Say "CUDA_PATH = $cudaRoot"
    $nproc = [Environment]::ProcessorCount
    $src = Join-Path $repo 'third_party\llama.cpp'
    $bld = Join-Path $repo 'build\llama-cuda'
    if (-not (Test-Path $bld)) { $null = New-Item -ItemType Directory -Path $bld -Force }
    $cfgOut = Join-Path $bld 'setup-wizard-cmake-configure.log'
    $cfgErr = Join-Path $bld 'setup-wizard-cmake-configure.err.log'
    $bldOut = Join-Path $bld 'setup-wizard-cmake-build.log'
    $bldErr = Join-Path $bld 'setup-wizard-cmake-build.err.log'
    if ($DR) {
        Say ("(dry-run: cmake -S $src -B $bld -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON)")
        Say ("(dry-run: cmake --build $bld --config Release --target llama-server --parallel $nproc)")
        return $bin
    }
    # cmake spawns long-lived descendants (vctip.exe, MSBuild nodes) that hold
    # the redirected stdout and block 'Start-Process -Wait' for up to ~15 min
    # after the command finished (measured 2026-11-03) - so both cmake calls
    # go through Invoke-NativeCmd (exit code via the '%ERRORLEVEL%' rc file).
    # The log paths above are the progress source.
    $rc = Invoke-NativeCmd -Exe 'cmake' -ArgList @('-S', $src, '-B', $bld, '-DGGML_CUDA=ON', '-DLLAMA_LAYER_STREAMING=ON') -OutLog $cfgOut -ErrLog $cfgErr
    if ($rc -ne 0) {
        Get-Content $cfgErr -Tail 50 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        ExitCode 4 "cmake configure failed (exit $rc; logs: $cfgOut / $cfgErr)"
    }
    # /nodeReuse:false (forwarded to MSBuild after '--'): without it, MSBuild
    # keeps ~32 worker-node daemons for ~15 min after the build. Harmless now
    # that Invoke-NativeCmd returns before they die, but it leaves a pile of
    # MSBuild.exe processes behind every wizard run - so it stays off.
    $rc = Invoke-NativeCmd -Exe 'cmake' -ArgList @('--build', $bld, '--config', 'Release', '--target', 'llama-server', '--parallel', ([string]$nproc), '--', '/nodeReuse:false') -OutLog $bldOut -ErrLog $bldErr
    if ($rc -ne 0) {
        Get-Content $bldErr -Tail 50 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        Get-Content $bldOut -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        ExitCode 4 "build failed (exit $rc; logs: $bldOut / $bldErr)"
    }
    if (-not (Test-Path $bin)) { ExitCode 4 "build finished but $bin was not produced" }
    Ok "built $bin"
    return $bin
}

# --- parallel phase (SDD D20): non-blocking build leg ---------------------
# Same .bat + rc-file design as Invoke-NativeCmd (D50/D53: both streams live
# inside the .bat, the rc is written on a SECOND physical line), but launched
# WITHOUT waiting: Start-Process -WindowStyle Hidden -PassThru creates no
# redirected handle in this process, so the long-lived MSBuild/vctip
# descendants (the D49 trap) cannot block us, and the rc file appears when
# each cmake stage exits. The download's progress tick (step 5) polls both rc
# files and prints the combined status line; Wait-BuildLeg is the blocking
# collector (single-leg runs, dry-run, failure tails).
function Start-BuildLeg {
    param([string]$repo)
    $bin = Join-Path $repo 'build\llama-cuda\bin\Release\llama-server.exe'
    $bld = Join-Path $repo 'build\llama-cuda'
    if (-not (Test-Path $bld)) { $null = New-Item -ItemType Directory -Path $bld -Force }
    $src = Join-Path $repo 'third_party\llama.cpp'
    $nproc = [Environment]::ProcessorCount
    $qArg = { param($s) if ($s -match '\s' -or ($s -match '["&<>|,]')) { '"' + ($s -replace '"','""') } else { $s } }
    $mkBat = {
        param([string]$Bat, [string]$Rc, [string]$OutL, [string]$ErrL, [string]$Exe, [string[]]$Argl)
        $argStr = ($Argl | ForEach-Object { & $qArg ([string]$_) }) -join ' '
        # cmd quoting for the exe (same convention as Invoke-NativeCmd). NO
        # leading '&': a command line that STARTS with '&' is a syntax error
        # for cmd ('& was unexpected at this time') and the whole .bat aborts
        # before the rc line runs (measured 2026-11-06: the parallel build
        # legs sat at 'configuring' forever with no logs and no rc files).
        $exeQ = $Exe
        if ($Exe -match '\s' -or ($Exe -match '"')) { $exeQ = '"' + ($Exe -replace '"','""') + '"' }
        $lines = @(
            '@echo off',
            "$exeQ $argStr > `"$OutL`" 2> `"$ErrL`"",
            "echo %ERRORLEVEL% > `"$Rc`""
        )
        [System.IO.File]::WriteAllLines($Bat, $lines, [System.Text.Encoding]::Default)
    }
    $cfgBat = Join-Path $bld 'swz-parallel-cfg.bat'
    $cfgRc  = Join-Path $bld 'swz-parallel-cfg.rc'
    $bldBat = Join-Path $bld 'swz-parallel-build.bat'
    $bldRc  = Join-Path $bld 'swz-parallel-build.rc'
    $cfgOut = Join-Path $bld 'setup-wizard-cmake-configure.log'
    $cfgErr = Join-Path $bld 'setup-wizard-cmake-configure.err.log'
    $bldOut = Join-Path $bld 'setup-wizard-cmake-build.log'
    $bldErr = Join-Path $bld 'setup-wizard-cmake-build.err.log'
    Remove-Item $cfgRc, $bldRc, $cfgOut, $cfgErr, $bldOut, $bldErr -Force -ErrorAction SilentlyContinue
    # named arguments: a positional @() would UNROLL into $Argl's first
    # element (PS scriptblock positional array behavior)
    & $mkBat -Bat $cfgBat -Rc $cfgRc -OutL $cfgOut -ErrL $cfgErr -Exe 'cmake' -Argl @('-S', $src, '-B', $bld, '-DGGML_CUDA=ON', '-DLLAMA_LAYER_STREAMING=ON')
    & $mkBat -Bat $bldBat -Rc $bldRc -OutL $bldOut -ErrL $bldErr -Exe 'cmake' -Argl @('--build', $bld, '--config', 'Release', '--target', 'llama-server', '--parallel', ([string]$nproc), '--', '/nodeReuse:false')
    # configure and build MUST be sequential: started as two independent
    # processes the build raced the configure and died with 'not a CMake build
    # directory (missing CMakeCache.txt)' (measured 2026-11-13). One driver
    # .bat runs the configure .bat, then the build .bat only if it succeeded;
    # a failed configure writes rc 1 into the build rc so the poller stops.
    $seqBat = Join-Path $bld 'swz-parallel-seq.bat'
    $seqLines = @(
        '@echo off',
        "call `"$cfgBat`"",
        "set /p CFGRC=<`"$cfgRc`"",
        'if not "%CFGRC%"=="0 " if not "%CFGRC%"=="0" goto cfgfail',
        "call `"$bldBat`"",
        'goto :eof',
        ':cfgfail',
        "echo 1 > `"$bldRc`""
    )
    [System.IO.File]::WriteAllLines($seqBat, $seqLines, [System.Text.Encoding]::Default)
    $p1 = Start-Process -FilePath $seqBat -WindowStyle Hidden -PassThru
    $p2 = $p1
    return [pscustomobject]@{ Bin=$bin; P1=$p1; P2=$p2; CfgRc=$cfgRc; BldRc=$bldRc; CfgOut=$cfgOut; CfgErr=$cfgErr; BldOut=$bldOut; BldErr=$bldErr }
}

function Test-BuildLegRunning {
    # true while the build leg can still fail (configure bat alive, or the
    # rc missing, or the build stage not yet reported). The rc file is the
    # completion signal, NOT the process handle: the .bat's cmd may linger on
    # MSBuild/vctip descendants holding no handles of ours (D49).
    param([object]$leg)
    $rc1Done = Test-Path $leg.CfgRc
    if (-not $rc1Done) { return $true }
    $rc2Done = Test-Path $leg.BldRc
    return (-not $rc2Done)
}

function Stop-BuildLeg {
    # fail-fast / Ctrl-C: kill the cmake tree. The .bat's cmd is the process
    # group leader for its children; taskkill /T takes the whole tree.
    param([object]$leg)
    foreach ($p in @($leg.P1, $leg.P2)) {
        if ($null -ne $p -and -not $p.HasExited) {
            try { $null = & taskkill /PID $p.Id /T /F 2>$null } catch { }
        }
    }
}

function Wait-BuildLeg {
    param([object]$leg)
    $leg.P1.WaitForExit() | Out-Null
    if (-not (Test-Path $leg.CfgRc)) { ExitCode 4 'build configure did not report an exit code (logs: ' + $leg.CfgOut + ' / ' + $leg.CfgErr + ')' }
    $rc1 = [int](Get-Content $leg.CfgRc | Select-Object -First 1)
    if ($rc1 -ne 0) {
        Get-Content $leg.CfgErr -Tail 50 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        ExitCode 4 "cmake configure failed (exit $rc1; logs: $($leg.CfgOut) / $($leg.CfgErr))"
    }
    $leg.P2.WaitForExit() | Out-Null
    if (-not (Test-Path $leg.BldRc)) { ExitCode 4 'build did not report an exit code (logs: ' + $leg.BldOut + ' / ' + $leg.BldErr + ')' }
    $rc2 = [int](Get-Content $leg.BldRc | Select-Object -First 1)
    if ($rc2 -ne 0) {
        Get-Content $leg.BldErr -Tail 50 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        Get-Content $leg.BldOut -Tail 20 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
        ExitCode 4 "build failed (exit $rc2; logs: $($leg.BldOut) / $($leg.BldErr))"
    }
    if (-not (Test-Path $leg.Bin)) { ExitCode 4 "build finished but $($leg.Bin) was not produced" }
    Ok ("built " + $leg.Bin)
    return $leg.Bin
}

# --- model (SDD 4.5) ------------------------------------------------------
function Test-DiskSpace {
    # SDD 4.1: disk >= model size + 15 GiB, confirm to proceed (default no).
    # The PS twin of the sh check_disk (D20: moved into the settings phase,
    # where the model size is known, BEFORE the download starts).
    param([string]$dir, [double]$sizeGib)
    $drive = [System.IO.Path]::GetPathRoot($dir)
    $needGib = [math]::Ceiling($sizeGib) + 15
    try {
        $freeGib = [math]::Round((Get-PSDrive -Name $drive.TrimEnd('\')).Free / 1GB, 0)
    } catch { return }
    if ($freeGib -lt $needGib) {
        Warn "disk has $freeGib GiB free, the model plus build wants ~$needGib GiB"
        if (-not (Confirm 'proceed anyway?' $false)) {
            ExitCode 6 'aborted: not enough disk space'
        }
    }
}

function Pick-Model {
    param([string]$key)
    # SDD D19: flat (model, quant) entries. -Model picks the model; -Quant
    # picks the quant (default = the model's Default entry); numeric keys 1-7
    # address the flat list directly.
    if ($key) {
        $pool = $script:MODELS | Where-Object { $_.Key -eq $key }
        if (-not $pool) {
            if ($key -match '^[1-7]$') { return $script:MODELS[[int]$key - 1] }
            ExitCode 1 "unknown model '$key' (choose qwen38, qwen36, flint, apex35, or 1-7)"
        }
        if ($Quant) {
            $m = $pool | Where-Object { $_.Quant -eq $Quant } | Select-Object -First 1
            if (-not $m) {
                ExitCode 1 "no '$Quant' build for $key (available: $(@($pool | ForEach-Object { $_.Quant }) -join ', '))"
            }
            return $m
        }
        return ($pool | Where-Object { $_.Default } | Select-Object -First 1)
    }
    $opts = @()
    for ($i = 0; $i -lt $script:MODELS.Count; $i++) {
        $m = $script:MODELS[$i]
        $q = if ($m.Default) { $m.Quant } else { "$($m.Quant) (alt)" }
        $opts += [pscustomobject]@{ Key=($m.Key + '-' + $m.Quant); Label="$($m.Key): $($m.Label), $($m.SizeGib) GiB [Q: $q]" }
    }
    $sel = AskChoice 'Which model (and quant)?' $opts 'qwen36-q4k_m'
    return $script:MODELS | Where-Object { ($_.Key + '-' + $_.Quant) -eq $sel } | Select-Object -First 1
}

function Resolve-ModelFile {
    # D20: settings phase - decides LOCAL_FILE (an existing local file) or
    # DOWNLOAD_TARGET (the table file, to be fetched by Download-Model in the
    # parallel phase). Nothing is downloaded here.
    param([object]$m, [string]$dir)
    if (-not $dir) { $dir = $m.DefaultDir }
    if (-not (Test-Path $dir)) { $null = New-Item -ItemType Directory -Path $dir -Force }
    $local = Join-Path $dir $m.File
    if (Test-Path $local) { $script:LOCAL_FILE = $local; $script:DOWNLOAD_TARGET = $null; return }
    # offer other ggufs in the dir
    $others = Get-ChildItem -Path $dir -Filter *.gguf -File -ErrorAction SilentlyContinue
    if ($others -and $others.Count -gt 0) {
        Say "no $($m.File) in $dir, but found:"
        for ($i = 0; $i -lt $others.Count; $i++) { Say "    $($i + 1)) $($others[$i].Name)" }
        $a = Ask "use one of these (number), 'y' to download $($m.File), or type a full path" 'y'
        if ($a -match '^[0-9]+$' -and [int]$a -ge 1 -and [int]$a -le $others.Count) {
            $script:LOCAL_FILE = $others[[int]$a - 1].FullName; $script:DOWNLOAD_TARGET = $null; return
        }
        if ($a -match '^y' -or $a -eq '') {
            $script:LOCAL_FILE = $null; $script:DOWNLOAD_TARGET = $local; return
        }
        if ($a -match '\.gguf$' -and (Test-Path $a)) { $script:LOCAL_FILE = $a; $script:DOWNLOAD_TARGET = $null; return }
        $script:LOCAL_FILE = $null; $script:DOWNLOAD_TARGET = $local
        return
    }
    $script:LOCAL_FILE = $null
    $script:DOWNLOAD_TARGET = $local
}

function Download-Model {
    param([object]$m, [string]$dir, [scriptblock]$Status = $null)
    $file = Join-Path $dir $m.File
    $part = "$file.part"
    $url = "https://huggingface.co/$($m.Repo)/resolve/main/$($m.File)"
    # D20: called from the parallel phase (step 5) - the phase header already
    # announced this leg; no Step() here. When a -Status callback is supplied
    # (the parallel phase), the combined build|model status line is rendered
    # by it; otherwise this function prints its own progress line.
    Say ("model: downloading " + $m.File + " (" + $m.SizeGib + " GiB) from Hugging Face")
    Say $url
    if ($SkipDownload) { ExitCode 5 "-SkipDownload set but no local model found (expected $file)" }
    $script:DL_DONE = $false
    $script:DL_FAILMSG = $null
    $maxBytes = 0
    if ($MaxDownloadMib -gt 0) { $maxBytes = [int64]$MaxDownloadMib * 1MB; Say "(test hook: -MaxDownloadMib $MaxDownloadMib, stopping at $MaxDownloadMib MiB)" }
    if ($DR) { Say '(dry-run: not downloading)'; return $file }
    $hdr = @{}
    if ($HfToken) { $hdr['Authorization'] = "Bearer $HfToken"; Say '(using HF token from --hf-token/HF_TOKEN; never printed)' }

    $resume = $false
    $have = 0
    if (Test-Path $part) { $have = (Get-Item $part).Length; $resume = $true; Say ("resuming: {0:N1} MiB already in {1}" -f ($have / 1MB), $part) }

    if (-not $resume -and (Test-Path $file)) {
        # a complete file exists (e.g. user placed it): verify size only
        return $file
    }

    Say 'downloading ... (Ctrl-C aborts; partial data is kept for resume)'
    # curl.exe (ships with Windows 10 and later) does the transfer. It resumes
    # with -C - from the .part size, runs as its own process (this thread only
    # polls the .part size and redraws the status line), and has no per-chunk
    # sleep. The HF token goes in a header file, not on the command line.
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curl) { ExitCode 5 'curl.exe not found (it ships with Windows 10 and later)' }
    $hdrFile = $null
    if ($HfToken) {
        $hdrFile = Join-Path ([System.IO.Path]::GetTempPath()) ('hf-hdr-' + [guid]::NewGuid().ToString('N') + '.txt')
        [System.IO.File]::WriteAllText($hdrFile, "Authorization: Bearer $HfToken`r`n")
    }
    try {
        for ($attempt = 1; $attempt -le 2; $attempt++) {
            $cargs = @('-L', '--fail', '-sS', '--retry', '3', '--retry-delay', '5', '-C', '-', '-o', "`"$part`"")
            if ($hdrFile) { $cargs += @('-H', "`"@$hdrFile`"") }
            $cargs += "`"$url`""
            $p = Start-Process -FilePath $curl.Source -ArgumentList $cargs -NoNewWindow -PassThru
            while (-not $p.HasExited) {
                $sz = if (Test-Path $part) { (Get-Item $part).Length } else { 0 }
                if ($maxBytes -gt 0 -and $sz -ge $maxBytes) { $p.Kill(); break }
                if ($Status) { & $Status }
                else { Write-Host ("`r    {0,8} MiB" -f [math]::Floor($sz / 1MB)) -NoNewline }
                Start-Sleep -Milliseconds 500
                $p.Refresh()
            }
            $p.WaitForExit()
            if ($maxBytes -gt 0 -and (Get-Item $part).Length -ge $maxBytes) {
                # test hook: cut the .part at exactly the requested size
                $fsT = [System.IO.File]::Open($part, 'Open', 'Write')
                $fsT.SetLength($maxBytes)
                $fsT.Close()
                break
            }
            if ($p.ExitCode -eq 0) { break }
            if ($p.ExitCode -eq 33 -and $attempt -eq 1) {
                # 33 = server refused byte ranges: start over once
                Say 'server ignored Range; restarting from scratch'
                Remove-Item $part -Force
                continue
            }
            throw "curl exited with code $($p.ExitCode)"
        }
        Write-Host ''
        $script:DL_DONE = $true
    } catch {
        $script:DL_FAILMSG = "download failed: $($_.Exception.Message) (manual: $url)"
        ExitCode 5 $script:DL_FAILMSG
    } finally {
        if ($hdrFile) { Remove-Item $hdrFile -Force -ErrorAction SilentlyContinue }
    }

    # integrity: complete transfer expected
    $got = (Get-Item $part).Length
    if ($maxBytes -gt 0) {
        Warn "test hook stopped transfer at $([math]::Floor($got / 1MB)) MiB (expected full $([math]::Round($m.SizeGib, 1)) GiB)"
        Say 'leaving .part in place to verify resume; NOT renaming to the final name.'
        return $part
    }
    if ($got -le 0) { ExitCode 5 'download produced an empty file' }
    # rename with sha-256 guard if a file of the same size already exists
    if (Test-Path $file) {
        $a = (Get-FileHash $file -Algorithm SHA256).Hash
        $b = (Get-FileHash $part -Algorithm SHA256).Hash
        if ($a -eq $b) { Remove-Item $file -Force }
        else { ExitCode 5 'existing file has the same size but a different sha-256; manual check needed' }
    }
    Move-Item $part $file -Force
    Ok "model ready: $file"
    return $file
}

# VOD-6: fetch the vision projector with the model (same Hugging Face repo, same resume / Range /
# token logic as the model). Nothing to do without vision or when the file is already there.
function Download-Mmproj {
    param([object]$m)
    if (-not $script:Vision -or -not $script:MmprojPath) { return }
    if (Test-Path $script:MmprojPath) { Ok "projector: $($script:MmprojPath)"; return }
    $mmRepo = if ($m.PSObject.Properties['MmprojRepo'] -and $m.MmprojRepo) { $m.MmprojRepo } else { $m.Repo }
    $mm = [pscustomobject]@{ Repo = $mmRepo; File = $m.Mmproj; SizeGib = [math]::Round($m.MmprojBytes / 1GB, 2) }
    $null = Download-Model -m $mm -dir (Split-Path $script:MmprojPath -Parent)
}

# --- recipe + emission (SDD 4.6/4.7) --------------------------------------
function Pick-Recipe {
    param([object]$m)
    # D20: recipe selection is part of the settings phase (no step number).
    Say 'recipe (measured rows only):'
    # promote the -Vram/-Context parameters into script scope so main can see them
    if ($Vram) { $script:Vram = $Vram }
    if ($Context) { $script:Context = $Context }
    $avail = $script:RECIPES | Where-Object { $_.Model -eq $m.Key }
    if ($script:Vram -and $script:Context) {
        $r = RecipeFor $m.Key $script:Vram $script:Context $m.Quant
        if ($r) { return (Check-RecipeQuant $r $m) }
    }
    if (-not $script:Vram) {
        $opts = @(
            [pscustomobject]@{ Key='full'; Label='full - use all card VRAM (~22+ GiB class)' },
            [pscustomobject]@{ Key='16g';  Label='16g  - ~16 GiB class target (keep ~8 GB free)' },
            [pscustomobject]@{ Key='8g';   Label='8g   - 8 GiB class target' }
        )
        $script:Vram = AskChoice 'VRAM target?' $opts 'full'
    }
    if (-not $script:Context) {
        $opts = @(
            [pscustomobject]@{ Key='262k'; Label='262k - 262,144 tokens' },
            [pscustomobject]@{ Key='128k'; Label='128k - 128,000 tokens' }
        )
        $script:Context = AskChoice 'Context size?' $opts '262k'
    }
    $r = RecipeFor $m.Key $script:Vram $script:Context $m.Quant
    if (-not $r) {
        Warn "no measured recipe for $($m.Key) / $($m.Quant) / $($script:Vram) / $($script:Context) - available rows for $($m.Key):"
        foreach ($a in $avail) {
            Write-Host ("    {0}  {1,-5} {2,-6} {3,-7} {4}" -f $a.Id, $a.Vram, $a.Ctx, $a.Quant, $a.Measured)
        }
        if (IsMoeModel $m.Key) {
            $cls = MoeClassRow '8g' '262k'
            if ($cls) { Write-Host ("    {0}  {1,-5} {2,-6} {3,-7} {4}" -f $cls.Id, $cls.Vram, $cls.Ctx, 'any MoE', ('(shared 8g profile of every MoE model) ' + $cls.Measured)) }
        }
        ExitCode 3 'no measured recipe for the chosen model/quant/VRAM/context (rows above are the available combinations; the wizard only emits measured rows)'
    }
    return (Check-RecipeQuant $r $m)
}

function Check-RecipeQuant {
    param([object]$r, [object]$m)
    # SDD D21/D19: a fallback row (the model's default quant) was measured on a different file
    if ($r.Model -ne $m.Key) {
        # SDD D24: the shared 8g MoE profile, measured on another model
        Warn "row $($r.Id) is the shared 8g profile of every MoE model (measured on $($r.Model) / $($r.Quant)); applied to $($m.Key) / $($m.Quant) ($($m.File)) - its speed/VRAM numbers are for the measured model"
        # SDD D25 (2026-11-09): the owner's own run of this model through the shared profile
        if ($m.Key -eq 'apex35' -and $m.Quant -eq 'q5k_m' -and $r.Vram -eq '8g') {
            Say "measured on $($m.File) with this row (owner, 8 GiB card, Linux): prompt 51,885 tokens at 537.42 t/s (96.5 s), 1,400 generated tokens at 30.88 t/s (32.39 ms per token)"
        }
    }
    elseif ($r.Quant -ne $m.Quant) {
        Warn "row $($r.Id) was measured on quant $($r.Quant); you chose $($m.Quant) ($($m.File)) - its speed/VRAM numbers are for $($r.Quant)"
    }
    return $r
}

function Get-GitShort {
    param([string]$repo)
    $h = 'unknown'
    try {
        $h = (& git -C $repo rev-parse --short HEAD 2>$null) | Select-Object -First 1
    } catch { }
    if (-not $h) { $h = 'unknown' }
    return $h
}

function Test-RowFlags {
    param([object]$r, [string]$exe)
    # the emitted command must be runnable by the binary it points at: every
    # --flag of the row must appear in 'llama-server --help' (host-only check)
    if (-not (Test-Path $exe)) { Say "(flag check skipped: $exe not built yet)"; return }
    $hOut = Join-Path $env:TEMP 'swz-help.log'
    $hErr = Join-Path $env:TEMP 'swz-help.err.log'
    $null = Invoke-Native -Exe $exe -ArgList @('--help') -OutLog $hOut -ErrLog $hErr
    $help = [string](Get-Content $hOut -Raw -ErrorAction SilentlyContinue) + ' ' + [string](Get-Content $hErr -Raw -ErrorAction SilentlyContinue)
    $missing = @()
    foreach ($tok in ($r.Flags -split '\s+')) {
        if ($tok -notmatch '^--[a-z]') { continue }
        if ($help -notmatch [regex]::Escape($tok)) { $missing += $tok }
    }
    # VOD-6: the vision flags the wizard adds must exist in this binary too
    if ($script:Vision) {
        foreach ($tok in @('--mmproj', '--mmproj-on-demand')) {
            if ($help -notmatch [regex]::Escape($tok)) { $missing += $tok }
        }
    }
    if ($missing.Count -gt 0) {
        ExitCode 1 ("binary does not support row $($r.Id) flags: " + ($missing -join ', ') +
            ' — rebuild from a branch that contains them, or pick another row')
    }
    Ok "binary supports all $($r.Id) row flags"
}

function Emit-RunScript {
    param([object]$r, [string]$modelFile, [string]$repo, [string]$exe)
    Step 'Emit run-server.ps1'
    $bind = $Bind
    if (-not $bind) { $bind = $r.Bind }
    if (-not $port) { $port = '8090' }   # D22: the settings phase normally fills $Port; this keeps the emitter self-contained
    $out = Join-Path $repo 'run-server.ps1'
    $last = Join-Path $repo 'run-server.last.ps1'
    if (Test-Path $out) { Move-Item $out $last -Force; Say "previous run-server.ps1 saved as run-server.last.ps1" }

    $logName = Join-Path $repo ("build\llama-cuda\server-$($r.Id).log")
    $secNote = ''
    if ($bind -eq '0.0.0.0') {
        # single line: two PS string assignments concatenate without a newline (measured 2026-11-04)
        $secNote = "# NOTE: this recipe binds 0.0.0.0 (measured that way). If you do not need the server on the network, change --host 0.0.0.0 to --host 127.0.0.1 (see docs/guide/USAGE.md section 5)."
    }
    $note = ''
    if ($r.Note) { $note = "# NOTE: $($r.Note)" }

    $lines = @()
    # date (day) + git short hash only: re-running the wizard on the same day
    # must produce a byte-identical file (plan AC-2 idempotence)
    $lines += '# run-server.ps1 - generated by setup.ps1 on ' + (Get-Date -Format 'yyyy-MM-dd')
    $lines += "# Recipe $($r.Id) (measured: $($r.Measured))"
    $lines += "# Model: $modelFile"
    $lines += "# Wizard: $script:GitShort  |  regenerate: .\\setup.ps1"
    if ($note)   { $lines += $note }
    if ($secNote) { $lines += $secNote }
    $lines += ''
    $lines += '$exe = "' + $exe + '"'
    $lines += '$model = "' + $modelFile + '"'
    if ($script:Vision) {
        $lines += '$mmproj = "' + $script:MmprojPath + '"'
        $lines += '# vision (not part of the measured row): --mmproj-on-demand keeps the projector (~0.84 GiB) in host RAM and on the GPU only while a request''s images are encoded (docs/plans/vision-on-demand-results.md). For a text-only server remove "--mmproj $mmproj --mmproj-on-demand" below.'
        if ($m.Key -notin @('qwen36', 'apex35')) {
            $lines += '# vision on this model is NOT measured: --mmproj-on-demand was measured on Qwen3.6-35B-A3B only (apex35 shares its architecture). The first image request on this model is the test; to go back to text only, remove the two vision flags below.'
        }
    }
    $lines += '$port = ' + $port
    $lines += '$log = "' + $logName + '"'
    # SDD D28: a default cap on the length of one response. -n is the limit for a request that names none of its own
    # (a request that sends max_tokens overrides it); a recipe that already sets -n keeps its own.
    $hasPredict = Test-HasPredictFlag $script:RowFlags.Flags
    if (-not $hasPredict) {
        $lines += '# max new tokens per response (-n) for a request that sets no max_tokens of its own: 16000. Raise it here if you need longer answers.'
        $lines += '$maxTokens = 16000'
    }
    if ($r.Env) {
        foreach ($k in $r.Env.Keys) {
            $lines += "# env (row $($r.Id), verbatim): $k=$($r.Env[$k])"
            $lines += '$env:' + $k + ' = "' + $r.Env[$k] + '"'
        }
    }
    if ($script:RowFlagsAddedInt8) {
        $lines += '# D22: --lm-prefill-attn-int8 was ADDED for this card (compute capability <= 6.1, Pascal or lower): the pass flash attention takes the int8 Q.K^T tile, an approximation of the exact kernel (see the help text of --lm-prefill-attn-int8). Remove it for exact arithmetic.'
    }
    $lines += ''
    $lines += 'if (-not (Split-Path $log -Parent | Test-Path)) { New-Item -ItemType Directory -Path (Split-Path $log -Parent) -Force | Out-Null }'
    if (-not $hasPredict) {
        $lines += 'Write-Host "max new tokens per response: $maxTokens (-n). Edit `$maxTokens at the top of this script if you need more (a request that sets max_tokens itself overrides it)."'
    }
    $flags = $script:RowFlags.Flags
    if ($r.Device) { $flags = $flags + ' --device CUDA' + (Resolve-GpuIndex -VramTarget $r.Vram) }
    $nArg = if ($hasPredict) { '' } else { '-n $maxTokens ' }
    $lines += '& $exe -m $model --host ' + $bind + ' --port $port ' + $nArg + $flags + ' 2>&1 | Tee-Object -FilePath $log'
    $lines += ''
    $lines += '# stop: Ctrl-C in this console.  Health: curl.exe http://127.0.0.1:' + $port + '/health'
    $lines += '# Pong test (USAGE.md section 5): see docs/SETUP.md FAQ'
    if ($DR) {
        Say '(dry-run: would write '$out')'
        $lines | ForEach-Object { Say "    $_" }
        return
    }
    Set-Content -Path $out -Value $lines -Encoding ASCII
    Ok "wrote $out"
}

function Start-ServerNow {
    param([string]$script, [string]$repo)
    if ($NoStart) { Say '(not starting: -NoStart)'; return }
    if ($DR) { Say '(dry-run: not starting)'; return }
    $go = $Start
    if (-not $Start) { $go = Confirm 'Start the server now?' $false }
    if (-not $go) { Ok 'done. Run it any time with: powershell -ExecutionPolicy Bypass -File .\\run-server.ps1'; return }
    Say 'starting server (Ctrl-C stops it here and in the server)...'
    & powershell -NoProfile -ExecutionPolicy Bypass -File $script
    if ($LASTEXITCODE -ne 0) { Warn "server exited with code $LASTEXITCODE (log tail below)" }
    $log = Join-Path $repo ("build\llama-cuda\server-$($script:RecipeId).log")
    if (Test-Path $log) { Get-Content $log -Tail 10 | ForEach-Object { Say "    $_" } }
}

# --- main ------------------------------------------------------------------
# D20 flow: env -> install -> clone -> SETTINGS (all questions up front,
# refusal exit 3 BEFORE any build or download) -> PARALLEL PHASE (build leg
# and model download in background, one combined status line, fail-fast)
# -> flag gate -> emit -> start. Invoke-Build (the legacy sequential form)
# stays as the documented command shape for fresh-clone/-SkipModel runs.
Write-Host ''
Write-Host '  poormansllama setup wizard (Windows)' -ForegroundColor Cyan
Write-Host ''

$repo = Get-RepoRoot
$destRoot = $CloneDir   # empty = in-repo detection / default path (see Invoke-Clone)

# @() at every call site: 'return' unrolls a single-element array to a scalar
# (no .Count in PS 5.1), which silently skipped the install path (AC-4)
$bad = @(Check-Environment)
if ($bad.Count -gt 0) { Install-Missing -bad $bad; $bad = @(Check-Environment) }
if ($bad.Count -gt 0) {
    foreach ($b in $bad) {
        if ($b.Name -eq 'nvidia-smi') {
            Warn 'no NVIDIA GPU/driver found. Install the driver first: https://www.nvidia.com/Download/index.aspx'
            ExitCode 2 'NVIDIA driver/GPU required (not auto-installed)'
        }
    }
}

$repo = Invoke-Clone -destRoot $destRoot
$script:GitShort = Get-GitShort -repo $repo

# --- settings (SDD D20): every question BEFORE the heavy work --------------
$noAsk = $false
if ($SkipModel -and -not $Model) {
    $noAsk = $true
}
if ($ModelFile -and -not $Model -and -not $noAsk) {
    ExitCode 1 '-ModelFile requires -Model (qwen38|qwen36|flint|apex35) for recipe selection'
}

if ($noAsk) {
    # AC-5 shape: env + clone + build only, nothing to ask. The build runs
    # sequentially here (there is no download leg to run in parallel with).
    $exe = Invoke-Build -repo $repo
    Ok "done: environment + repository + build (model/recipe steps skipped: -SkipModel without -Model)"
    Ok "next: re-run with -Model qwen38|qwen36|flint|apex35 to pick a recipe and emit run-server.ps1"
    exit 0
}

Step 'Settings (asked before the build and the download)'
$m = Pick-Model -key $Model
$mdir = $ModelDir
if (-not $mdir) { $mdir = $m.DefaultDir }
$local = $null
if ($SkipModel) {
    $local = Join-Path $mdir $m.File
    Warn "-SkipModel: assuming the model is at $local (not checked)"
}
elseif ($ModelFile) {
    if (-not (Test-Path $ModelFile)) { ExitCode 1 "-ModelFile does not exist: $ModelFile" }
    $local = $ModelFile
    $script:DOWNLOAD_TARGET = $null
    Ok "model: $local (-ModelFile)"
}
else {
    Resolve-ModelFile -m $m -dir $mdir
    $local = $script:LOCAL_FILE
    if ($local) {
        Ok "model: $local"
        if ($local -notmatch [regex]::Escape($m.File)) {
            Warn "using a different file than the table's $($m.File); recipe rows assume the table file (SDD D7/D13 risks)"
        }
    }
    # disk check needs the model size; it runs here, before the download
    # starts, and only when a download is actually going to happen
    if ($script:DOWNLOAD_TARGET) { Test-DiskSpace -dir $mdir -sizeGib $m.SizeGib }
}
$r = Pick-Recipe -m $m
$script:RecipeId = $r.Id
Say ("recipe $($r.Id): $($r.Measured)")
if ($r.Note) { Say ("note: $($r.Note)") }
# D22-C: the server port, asked with the other settings (before the heavy
# work). Default 8090; an explicit -Port/--port skips the question.
if (-not $Port) {
    $Port = (Ask 'Server port?' '8090').Trim()
    while ($Port -notmatch '^\d+$' -or [int]$Port -lt 1 -or [int]$Port -gt 65535) {
        Warn "port must be a number 1-65535 (got '$Port')"
        $Port = (Ask 'Server port?' '8090').Trim()
    }
}
# VOD-6: vision support. The projector is fetched from the SAME Hugging Face repo as the model
# file and the emitted command gets --mmproj $mmproj --mmproj-on-demand (the projector then sits
# in host RAM and on the GPU only while a request's images are encoded). Default yes; -NoVision
# suppresses the question; -MmprojFile uses a local projector.
$script:Vision = $false
$script:MmprojPath = ''
$visionOffered = Test-VisionOffered -m $m -r $r
if ($MmprojFile) {
    if ($NoVision) { ExitCode 1 '-MmprojFile and -NoVision contradict each other' }
    if (-not $visionOffered) { ExitCode 1 "-MmprojFile: vision is not offered for $($m.Key) / recipe $($r.Id) (no projector is published for this model, or the recipe already carries --mmproj)" }
    if (-not (Test-Path $MmprojFile)) { ExitCode 1 "-MmprojFile does not exist: $MmprojFile" }
    $script:Vision = $true
    $script:MmprojPath = $MmprojFile
    Ok "vision: $MmprojFile (-MmprojFile)"
}
elseif ($visionOffered -and -not $NoVision) {
    if ($SkipModel) {
        Say 'vision: skipped with -SkipModel (no download; pass -MmprojFile to use a local projector)'
    }
    else {
        $mmMiB = [math]::Round($m.MmprojBytes / 1MB)
        $mmFrom = if ($m.PSObject.Properties['MmprojRepo'] -and $m.MmprojRepo) { "$($m.MmprojRepo) (a separate repo: the model's own repo publishes none)" } else { "$($m.Repo) (the model's repo)" }
        $script:Vision = Confirm "Vision support? Also fetches the projector $($m.Mmproj) ($mmMiB MiB) from $mmFrom; it is parked in host RAM between image requests" $true
        if ($script:Vision) {
            $mmDir = $mdir
            if ($ModelFile) { $mmDir = Split-Path $ModelFile -Parent }
            $script:MmprojPath = Join-Path $mmDir $m.Mmproj
            if (Test-Path $script:MmprojPath) { Ok "projector: $($script:MmprojPath)" }
            else { Say "projector: $($script:MmprojPath) (downloaded in the parallel phase)" }
        }
    }
}
elseif ($m.PSObject.Properties['Mmproj'] -and $m.Mmproj -and -not $visionOffered) {
    Say 'vision: not offered for this recipe (it already carries --mmproj)'
}
elseif (-not $NoVision -and -not $MmprojFile) {
    Say "vision: not available for $($m.Key) (no vision projector is published for it; pass -MmprojFile to use your own)"
}
# D22-A/B: complete the row's flags (pure; see Complete-RowFlags). The
# compute capability is for the card the run will use; on failure the row
# is emitted as measured (the int8 tile is a performance approximation, not
# a correctness requirement).
$script:GpuCc = Detect-GpuArch
if (-not $script:GpuCc) { Warn 'compute capability unavailable (nvidia-smi); the row flags stay as measured (no D22 additions)' }
else { Say "card compute capability: $script:GpuCc" }
$script:RowFlags = Complete-RowFlags -Flags $r.Flags -EnvText $r.Env -ComputeCap $script:GpuCc -Vision $script:Vision
$script:RowFlagsAddedInt8 = $script:RowFlags.AddedInt8
# D12 (SDD 4.6): warning only, threshold by VRAM target.
# D17: for device-pinned rows the check runs against the card the row will
# actually use (Resolve-GpuIndex), not just nvidia-smi's first line.
$gpuLine = $script:GpuLine
if ($r.Device) {
    $gpuDev = Resolve-GpuIndex -VramTarget $r.Vram
    $ns = Get-NvidiaSmiPath
    if ($ns) {
        $mibs = @(& $ns --query-gpu=memory.total --format=csv,noheader 2>$null)
        if ($mibs.Count -gt $gpuDev) { $gpuLine = $mibs[$gpuDev] }
    }
}
if ($gpuLine -match '([\d,]+)\s*MiB') {
    $gpuMib = [int]($Matches[1].Replace(',', ''))
    $need = 0
    if ($r.Vram -eq 'full') { $need = 22528 }
    elseif ($r.Vram -eq '16g') { $need = 10240 }
    elseif ($r.Vram -eq '8g') { $need = 8192 }
    if ($gpuMib -gt 0 -and $gpuMib -lt $need) {
        $card = 'a 24 GiB card'
        if ($r.Id -eq 'R7') { $card = 'an 8 GiB card (GTX 1070)' }
        Warn "GPU has $gpuMib MiB; row $($r.Id) was measured on $card (SDD D12: warning only, the server's memory guard fails loudly)"
    }
}

# --- parallel phase (SDD D20): build leg and download leg ------------------
$binPath = Join-Path $repo 'build\llama-cuda\bin\Release\llama-server.exe'
$buildNeeded = $true
if ($SkipBuild) {
    $buildNeeded = $false
    Say '(skipped: -SkipBuild)'
} elseif (-not $FullBuild -and (Test-BinaryFresh $repo $binPath)) {
    $buildNeeded = $false
    Ok "build skipped (binary newer than sources): $binPath"
}

if ($DR) {
    # dry-run: no processes. The documented commands stay visible (same text
    # as the sequential Invoke-Build prints, so AC-2 output is unchanged).
    if ($buildNeeded) {
        Step 'Build llama-server'
        Say ('(dry-run: cmake -S ' + (Join-Path $repo 'third_party\llama.cpp') + ' -B ' + (Join-Path $repo 'build\llama-cuda') + ' -DGGML_CUDA=ON -DLLAMA_LAYER_STREAMING=ON)')
        Say ('(dry-run: cmake --build ' + (Join-Path $repo 'build\llama-cuda') + ' --config Release --target llama-server --parallel ' + [Environment]::ProcessorCount + ')')
    }
    if ($script:DOWNLOAD_TARGET) {
        Say '(dry-run: not downloading)'
        # the emitted script must name the file the real run would produce
        if (-not $script:LOCAL_FILE) { $script:LOCAL_FILE = $script:DOWNLOAD_TARGET }
    }
    if ($script:Vision -and $script:MmprojPath -and -not (Test-Path $script:MmprojPath)) {
        Say "(dry-run: not downloading the projector $($m.Mmproj) from $(if ($m.PSObject.Properties['MmprojRepo'] -and $m.MmprojRepo) { $m.MmprojRepo } else { $m.Repo }))"
    }
    $exe = $binPath
}
elseif ($buildNeeded -and $script:DOWNLOAD_TARGET) {
    Step 'Build + download (in parallel; Ctrl-C keeps the .part for resume)'
    $script:BUILD_LEG = Start-BuildLeg -repo $repo
    $script:BUILD_LEG_ACTIVE = $script:BUILD_LEG
    $t0 = Get-Date
    # the download runs in the foreground (resume/Range/token logic as
    # before); its 500 ms tick doubles as the combined status line. The
    # callback takes no parameters: it reads $script:BUILD_LEG / .part size
    # itself (script-scope references from a nested scriptblock are resolved
    # against the GLOBAL scope, so no-parameter access to $script: vars is
    # the reliable form on PS 5.1).
    $script:DL_STATUS = {
        if ($script:DL_DONE) { return }
        $part = Join-Path $script:DL_DIR ($script:DL_NAME + '.part')
        if (Test-Path $part) {
            $kb = [math]::Floor((Get-Item $part).Length / 1MB)
            $el = (Get-Date) - $script:DL_T0
            $leg = $script:BUILD_LEG
            $btxt = 'not running (fresh or -SkipBuild)'
            if ($leg) {
                if (Test-Path $leg.CfgRc) { $btxt = 'building (cmake --build) ... ' + [string][math]::Floor($el.TotalMinutes) + ' min' }
                else { $btxt = 'configuring ... ' + [string][math]::Floor($el.TotalMinutes) + ' min' }
            }
            Write-Host ("`r  build: " + $btxt + "   |   model: " + $kb + " MiB downloaded") -NoNewline
        }
        # fail-fast (D20): a cmake stage that exited nonzero stops the phase
        # now, without waiting for the download. Wait-BuildLeg prints the log
        # tail and exits 4.
        $leg = $script:BUILD_LEG
        if ($leg) {
            if (Test-Path $leg.CfgRc) {
                $rc1 = [int](Get-Content $leg.CfgRc | Select-Object -First 1)
                if ($rc1 -ne 0) { Write-Host ''; Wait-BuildLeg -leg $leg }
            }
            if (Test-Path $leg.BldRc) {
                $rc2 = [int](Get-Content $leg.BldRc | Select-Object -First 1)
                if ($rc2 -ne 0) { Write-Host ''; Wait-BuildLeg -leg $leg }
            }
        }
    }
    $script:DL_DIR = $mdir
    $script:DL_NAME = $m.File
    $script:DL_T0 = $t0
    try {
        Download-Model -m $m -dir $mdir -Status $script:DL_STATUS
    } finally {
        # download failure already exits 5 (fail-fast); stop the build leg so
        # a 22-minute build does not keep running after an aborted download.
        if ($script:DL_FAILMSG -and $script:BUILD_LEG) { Stop-BuildLeg -leg $script:BUILD_LEG }
    }
    # VOD-6: the projector (0.84 GiB) follows the model in the foreground while the build leg keeps running
    try {
        Download-Mmproj -m $m
    } finally {
        if ($script:DL_FAILMSG -and $script:BUILD_LEG) { Stop-BuildLeg -leg $script:BUILD_LEG }
    }
    # the download succeeded (a failure already exited 5 with the build leg
    # stopped); collect the build result now - Wait-BuildLeg tails + exits 4
    # on a failed stage, prints 'built' on success.
    if ($script:BUILD_LEG) {
        $rcBad = $false
        if (Test-Path $script:BUILD_LEG.CfgRc) {
            $rc1 = [int](Get-Content $script:BUILD_LEG.CfgRc | Select-Object -First 1)
            if ($rc1 -ne 0) { $rcBad = $true }
        }
        if (-not $rcBad -and (Test-Path $script:BUILD_LEG.BldRc)) {
            $rc2 = [int](Get-Content $script:BUILD_LEG.BldRc | Select-Object -First 1)
            if ($rc2 -ne 0) { $rcBad = $true }
        }
        if ($rcBad -or (Test-BuildLegRunning $script:BUILD_LEG)) {
            Wait-BuildLeg -leg $script:BUILD_LEG
        }
        elseif (Test-Path $script:BUILD_LEG.Bin) { Ok 'built ' + $script:BUILD_LEG.Bin }
        else { Wait-BuildLeg -leg $script:BUILD_LEG }
    }
    $script:BUILD_LEG_ACTIVE = $null
    $exe = $binPath
}
elseif ($buildNeeded) {
    if ($script:DOWNLOAD_TARGET -and $SkipDownload) {
        ExitCode 5 '-SkipDownload set but no local model found (expected ' + $script:DOWNLOAD_TARGET + ')'
    }
    # no download leg: build sequentially in the foreground (the documented
    # command shape of Invoke-Build; same logs, same failure tails)
    $exe = Invoke-Build -repo $repo
    Download-Mmproj -m $m     # VOD-6: the projector, when vision is on and the file is not there yet
}
else {
    # build fresh/skipped; download only (no combined line needed)
    if ($script:DOWNLOAD_TARGET) {
        if ($SkipDownload) {
            ExitCode 5 '-SkipDownload set but no local model found (expected ' + $script:DOWNLOAD_TARGET + ')'
        }
        Step 'Download model (Hugging Face; Ctrl-C keeps the .part for resume)'
        Download-Model -m $m -dir $mdir
    }
    Download-Mmproj -m $m     # VOD-6
    $exe = $binPath
}

Test-RowFlags -r $r -exe $exe
Emit-RunScript -r $r -modelFile $local -repo $repo -exe $exe
Start-ServerNow -script (Join-Path $repo 'run-server.ps1') -repo $repo

Ok 'wizard finished'
exit 0
