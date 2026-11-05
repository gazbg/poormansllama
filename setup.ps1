# setup.ps1 - poormansllama bootstrap + setup wizard (Windows)
#
# One-liner (no local clone needed):
#   iex (irm 'https://raw.githubusercontent.com/gazbg/poormansllama/claude/flint-support/setup.ps1')
#
# What it does: checks the environment (git, CMake, CUDA, MSVC, NVIDIA GPU),
# offers to install missing tools via winget, clones the repository if not
# already inside one, builds llama-server with the fork's options, downloads
# the chosen GGUF model from Hugging Face if absent, and emits a ready-to-run
# run-server.ps1 built from a MEASURED recipe (no unmeasured flag combos).
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
    [string]$Model = '',            # qwen38 | qwen36 | flint | 1/2/3
    [ValidateSet('full', '16g', '8g')][string]$Vram = '',
    [ValidateSet('262k', '128k')][string]$Context = '',
    [string]$Port = '8033',
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
    [int]$MaxDownloadMib = 0        # 0 = full file; test hook, see docs/SETUP.md
)

# --- state -------------------------------------------------------------
$ErrorActionPreference = 'Stop'
$NI = $NonInteractive
$DR = $DryRun
$script:AbortExit = 6

trap {
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
$script:MODELS = @(
    [pscustomobject]@{ Key='qwen38'; Label='Qwen3.8-27B dense hybrid (unsloth UD Q4_K_M)'
        Repo='unsloth/Qwen3.8-27B-GGUF'; File='Qwen3.8-27B-UD-Q4_K_M.gguf'; SizeGib=15.4
        DefaultDir='C:/Models/ALL/other/qwen-3.8'; Kind='dense' },
    [pscustomobject]@{ Key='qwen36'; Label='Qwen3.6-35B-A3B MoE (unsloth UD Q4_K_M)'
        Repo='unsloth/Qwen3.6-35B-A3B-GGUF'; File='Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'; SizeGib=20.6
        DefaultDir='C:/Models/ALL/other/qwen-3.6'; Kind='moe' },
    [pscustomobject]@{ Key='flint'; Label='Flint SWA MoE (unsloth UD Q5_K_XL)'
        Repo='unsloth/flint-GGUF'; File='flint-UD-Q5_K_XL.gguf'; SizeGib=27.9
        DefaultDir='C:/Models/ALL/other/flint'; Kind='moe-swa' }
)

# Recipe rows: MEASURED, verbatim (SDD section 4.6, decision D4/D11).
# Flags = the exact command from the source run, minus the exe path, -m, and
# the host/port the wizard re-asks (defaults keep the row's own values).
$script:RECIPES = @(
    [pscustomobject]@{ Id='R1'; Model='qwen38'; Vram='full'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-37 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        Bind='127.0.0.1'; Measured='PP 1093.25 t/s, TG 14.95 t/s (owner 2026-10-29, RTX 5090 24 GiB)' },
    [pscustomobject]@{ Id='R2'; Model='qwen38'; Vram='16g'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --spec-type draft-mtp --spec-draft-p-min 0.6 --spec-draft-n-min 0 --spec-draft-n-max 3 --spec-mtp-defer -ctkd q8_0 -ctvd q8_0 -lv 4'
        Bind='127.0.0.1'; Measured='PP 1007.87 t/s, TG 9.96 t/s (owner 2026-10-29, RTX 5090 24 GiB)' },
    [pscustomobject]@{ Id='R3'; Model='qwen38'; Vram='8g'; Ctx='128k'
        Flags='-c 128000 -ctk q4_0 -ctv q4_0 -fa on -np 1 -t 16 --no-warmup --lm-prefill-host-layers 0-63 --lm-decode-stream dynamic --lm-decode-slots 2 --lm-prefill --lm-prefill-prearm always -lv 4'
        Bind='127.0.0.1'; Measured='PP 36.95 t/s, TG 9.92 t/s (owner 2026-10-29, RTX 5090 24 GiB)' },
    [pscustomobject]@{ Id='R4'; Model='flint'; Vram='full'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --lm-prefill-copy-piece 64'
        Bind='0.0.0.0'; Measured='PP 2422.94 t/s, TG 53.80 t/s (owner 2026-11-02, RTX 5090 24 GiB)' },
    [pscustomobject]@{ Id='R5'; Model='qwen36'; Vram='full'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 24 -b 262000 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 3 -lv 4 --hybrid-moe-prefill-free-all'
        Device='auto'; Bind='0.0.0.0'; Measured='owner working recipe 2026-11-02 (RTX 5090 24 GiB)' },
    [pscustomobject]@{ Id='R6'; Model='qwen36'; Vram='16g'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe --lm-prefill --lm-prefill-canonical --lm-prefill-fuse --lm-prefill-prearm auto --lm-prefill-ubatch 2048 --lm-prefill-segment 32768 -b 32768 --hybrid-moe-prefill-ubatch 0 -fitt 16000 --ctx-checkpoints 32'
        Bind='127.0.0.1'; Measured='G42 canonical: TG 3.0-3.7x stock, PP 3.65-4.11x stock (docs/guide/USAGE.md, build s53)'
        Note='measured on Qwen3.6-35B-A3B-MTP-UD-Q4_K_S; the wizard offers the closest HF file, Q4_K_M (SDD D13)' },
    [pscustomobject]@{ Id='R7'; Model='qwen36'; Vram='8g'; Ctx='262k'
        Flags='-c 262144 -ctk q8_0 -ctv q8_0 -fa on -np 1 --hybrid-moe -fitt 16 --lm-prefill --lm-prefill-ckpt-chunk 32 -b 32768 -ub 2048 --lm-prefill-ubatch 2048 --hybrid-moe-prefill-ubatch 0 --lm-prefill-slots 2 -lv 4 --hybrid-moe-prefill-free-all --lm-prefill-segment 32768 -sm none -mg 0'
        Device='auto'; Bind='0.0.0.0'; Measured='owner working recipe 2026-11-05 (8 GiB target, binds 0.0.0.0)'
        Env=@{ 'LLAMA_FATTN_TILE_Q8QK' = '1' }
        Note='the emitted script sets env LLAMA_FATTN_TILE_Q8QK=1 (int8 attention; part of the 8g row); the old LINUX-PASCAL.md fuse/prearm recipe is historical' }
)

function RecipeFor([string]$m, [string]$v, [string]$c) {
    return ($script:RECIPES | Where-Object { $_.Model -eq $m -and $_.Vram -eq $v -and $_.Ctx -eq $c } | Select-Object -First 1)
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
    Step '1. Environment check'
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
    Step '2. Install missing tools (winget)'
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
    Step '3. Repository'
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
    param([string]$repo)
    Step '4. Build llama-server'
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

# --- model (SDD 4.5) ------------------------------------------------------
function Pick-Model {
    param([string]$key)
    if ($key) {
        $m = $script:MODELS | Where-Object { $_.Key -eq $key } | Select-Object -First 1
        if ($m) { return $m }
        if ($key -match '^[1-3]$') { return $script:MODELS[[int]$key - 1] }
        ExitCode 1 "unknown model '$key' (choose qwen38, qwen36, flint, or 1-3)"
    }
    $opts = @()
    for ($i = 0; $i -lt $script:MODELS.Count; $i++) {
        $m = $script:MODELS[$i]
        $opts += [pscustomobject]@{ Key=$m.Key; Label="$($m.Key): $($m.Label) - $($m.File), $($m.SizeGib) GiB" }
    }
    $sel = AskChoice 'Which model?' $opts 'qwen36'
    return $script:MODELS | Where-Object { $_.Key -eq $sel } | Select-Object -First 1
}

function Resolve-ModelFile {
    param([object]$m, [string]$dir)
    if (-not $dir) { $dir = $m.DefaultDir }
    if (-not (Test-Path $dir)) { $null = New-Item -ItemType Directory -Path $dir -Force }
    $local = Join-Path $dir $m.File
    if (Test-Path $local) { return $local }
    # offer other ggufs in the dir
    $others = Get-ChildItem -Path $dir -Filter *.gguf -File -ErrorAction SilentlyContinue
    if ($others -and $others.Count -gt 0) {
        Say "no $($m.File) in $dir, but found:"
        for ($i = 0; $i -lt $others.Count; $i++) { Say "    $($i + 1)) $($others[$i].Name)" }
        $a = Ask "use one of these (number), 'y' to download $($m.File), or type a full path" 'y'
        if ($a -match '^[0-9]+$' -and [int]$a -ge 1 -and [int]$a -le $others.Count) { return $others[[int]$a - 1].FullName }
        if ($a -match '^y' -or $a -eq '') { return $null }   # null => download
        if ($a -match '\.gguf$' -and (Test-Path $a)) { return $a }
        return $null
    }
    return $null
}

function Download-Model {
    param([object]$m, [string]$dir)
    $file = Join-Path $dir $m.File
    $part = "$file.part"
    $url = "https://huggingface.co/$($m.Repo)/resolve/main/$($m.File)"
    Step "5. Download model ($($m.SizeGib) GiB) from Hugging Face"
    Say $url
    if ($SkipDownload) { ExitCode 5 "-SkipDownload set but no local model found (expected $file)" }
    $maxBytes = 0
    if ($MaxDownloadMib -gt 0) { $maxBytes = [int64]$MaxDownloadMib * 1MB; Say "(test hook: -MaxDownloadMib $MaxDownloadMib, stopping at $MaxDownloadMib MiB)" }
    if ($DR) { Say '(dry-run: not downloading)'; return $file }
    $hdr = @{}
    if ($HfToken) { $hdr['Authorization'] = "Bearer $HfToken"; Say '(using HF token from --hf-token/HF_TOKEN; never printed)' }

    $resume = $false
    $have = 0
    if (Test-Path $part) { $have = (Get-Item $part).Length; $resume = $true; Say "resuming: $($have / 1GB) GiB already in $part" }

    if (-not $resume -and (Test-Path $file)) {
        # a complete file exists (e.g. user placed it): verify size only
        return $file
    }

    Say 'downloading ... (Ctrl-C aborts; partial data is kept for resume)'
    # manual Range request on every PS version: -Resume is unreliable for >2GB
    # files and 416s on some servers; HttpWebRequest works on 5.1 and 7.
    try {
            $req = [System.Net.HttpWebRequest]::Create($url)
            $req.Method = 'GET'
            if ($have -gt 0) { $req.AddRange([int64]$have); $req.Method = 'GET' }
            if ($HfToken) { $req.Headers.Add('Authorization', "Bearer $HfToken") }
            $resp = $req.GetResponse()
            $len = [int64]$resp.ContentLength
            if ($have -gt 0 -and $resp.StatusCode -eq [System.Net.HttpStatusCode]::PartialContent) {
                Say "server accepted Range (continuing at $have)"
            } elseif ($have -gt 0) {
                Say 'server ignored Range; restarting from scratch'
                $have = 0
                $fs0 = [System.IO.File]::Create($part)
                $fs0.Close()
            }
            $mode = 'Create'
            if ($have -gt 0) { $mode = 'Append' }
            $fs = [System.IO.File]::Open($part, $mode)
            $buf = New-Object byte[] 8MB
            $read = 0
            while ($true) {
                $n = $resp.GetResponseStream().Read($buf, 0, $buf.Length)
                if ($n -le 0) { break }
                if ($maxBytes -gt 0 -and ($have + $read + $n) -gt $maxBytes) {
                    $n = [int]($maxBytes - $have - $read)
                    if ($n -le 0) { break }
                }
                $fs.Write($buf, 0, $n)
                $read += $n
                $done = $have + $read
                $pct = if ($len -gt 0) { [math]::Floor($done * 100 / $len) } else { 0 }
                $kb = [math]::Floor($done / 1MB)
                Write-Host ("`r    {0,8} MiB  {1,3}%" -f $kb, $pct) -NoNewline
                Start-Sleep -Milliseconds 500
            }
            $fs.Close()
            Write-Host ''
        } catch {
            ExitCode 5 "download failed: $($_.Exception.Message) (manual: $url)"
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

# --- recipe + emission (SDD 4.6/4.7) --------------------------------------
function Pick-Recipe {
    param([object]$m)
    Step '6. Recipe (measured only)'
    # promote the -Vram/-Context parameters into script scope so main can see them
    if ($Vram) { $script:Vram = $Vram }
    if ($Context) { $script:Context = $Context }
    $avail = $script:RECIPES | Where-Object { $_.Model -eq $m.Key }
    if ($script:Vram -and $script:Context) {
        $r = RecipeFor $m.Key $script:Vram $script:Context
        if ($r) { return $r }
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
    $r = RecipeFor $m.Key $script:Vram $script:Context
    if (-not $r) {
        Warn "no measured recipe for $($m.Key) / $($script:Vram) / $($script:Context) - available rows for $($m.Key):"
        foreach ($a in $avail) {
            Write-Host ("    {0}  {1,-5} {2,-6} {3}" -f $a.Id, $a.Vram, $a.Ctx, $a.Measured)
        }
        ExitCode 3 'no measured recipe for the chosen model/VRAM/context (rows above are the available combinations; the wizard only emits measured rows)'
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
    if ($missing.Count -gt 0) {
        ExitCode 1 ("binary does not support row $($r.Id) flags: " + ($missing -join ', ') +
            ' — rebuild from a branch that contains them, or pick another row')
    }
    Ok "binary supports all $($r.Id) row flags"
}

function Emit-RunScript {
    param([object]$r, [string]$modelFile, [string]$repo, [string]$exe)
    Step '7. Emit run-server.ps1'
    $bind = $Bind
    if (-not $bind) { $bind = $r.Bind }
    $port = $Port
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
    $lines += '$port = ' + $port
    $lines += '$log = "' + $logName + '"'
    if ($r.Env) {
        foreach ($k in $r.Env.Keys) {
            $lines += "# env (row $($r.Id), verbatim): $k=$($r.Env[$k])"
            $lines += '$env:' + $k + ' = "' + $r.Env[$k] + '"'
        }
    }
    $lines += ''
    $lines += 'if (-not (Split-Path $log -Parent | Test-Path)) { New-Item -ItemType Directory -Path (Split-Path $log -Parent) -Force | Out-Null }'
    $flags = $r.Flags
    if ($r.Device) { $flags = $flags + ' --device CUDA' + (Resolve-GpuIndex -VramTarget $r.Vram) }
    $lines += '& $exe -m $model --host ' + $bind + ' --port $port ' + $flags + ' 2>&1 | Tee-Object -FilePath $log'
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
$exe = Invoke-Build -repo $repo

$script:GitShort = Get-GitShort -repo $repo

if ($SkipModel -and -not $Model) {
    # AC-5 shape: env + clone + build only, nothing to ask
    Ok "done: environment + repository + build (model/recipe steps skipped: -SkipModel without -Model)"
    Ok "next: re-run with -Model qwen38|qwen36|flint to pick a recipe and emit run-server.ps1"
    exit 0
}

if ($ModelFile -and -not $Model) { ExitCode 1 '-ModelFile requires -Model (qwen38|qwen36|flint) for recipe selection' }
$m = Pick-Model -key $Model
$mdir = $ModelDir
if (-not $mdir) { $mdir = $m.DefaultDir }
if (-not $SkipModel) {
    if ($ModelFile) {
        if (-not (Test-Path $ModelFile)) { ExitCode 1 "-ModelFile does not exist: $ModelFile" }
        $local = $ModelFile
        Ok "model: $local (-ModelFile)"
    }
    else {
        $local = Resolve-ModelFile -m $m -dir $mdir
        if ($local) {
            Ok "model: $local"
            if ($local -notmatch [regex]::Escape($m.File)) {
                Warn "using a different file than the table's $($m.File); recipe rows assume the table file (SDD D7/D13 risks)"
            }
        }
        else {
            $local = Download-Model -m $m -dir $mdir
        }
    }
}
else {
    $local = Join-Path $mdir $m.File
    Warn "-SkipModel: assuming the model is at $local (not checked)"
}
$r = Pick-Recipe -m $m
$script:RecipeId = $r.Id
Say ("recipe $($r.Id): $($r.Measured)")
if ($r.Note) { Say ("note: $($r.Note)") }
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
Test-RowFlags -r $r -exe $exe
Emit-RunScript -r $r -modelFile $local -repo $repo -exe $exe
Start-ServerNow -script (Join-Path $repo 'run-server.ps1') -repo $repo

Ok 'wizard finished'
exit 0
