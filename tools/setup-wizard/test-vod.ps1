# tools/setup-wizard/test-vod.ps1
# Host-only test for the vision additions of setup.ps1 (VOD-6, widened by SDD D27; no GPU, no build, no model, no
# download): the projector columns of the model table, Test-VisionOffered, the vision append of
# Complete-RowFlags, and the dry-run emission with vision on / off / refused, for every model entry.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\test-vod.ps1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$setupPath = Join-Path $repo 'setup.ps1'

$cutAt = (Select-String -Path $setupPath -Pattern '^# --- main' | Select-Object -First 1).LineNumber
if (-not $cutAt) { Write-Host '[FAIL] main marker not found in setup.ps1' -ForegroundColor Red; exit 1 }
$cutFile = Join-Path $env:TEMP 'swz-vod-cut.ps1'
Set-Content -Path $cutFile -Value (Get-Content $setupPath | Select-Object -First ($cutAt - 1))
. $cutFile

$fail = 0
function Check([string]$what, [bool]$cond) {
    if ($cond) { Write-Host "  PASS $what" }
    else { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fail++ }
}
function Count([string]$text, [string]$pat) { return @([regex]::Matches($text, $pat)).Count }

# --- the table: projector sizes from the Hugging Face tree API (2026-11-09: Qwen3.6; 2026-11-12: Qwen3.8 and Apex, SDD D27)
$q36 = @($script:MODELS | Where-Object { $_.Key -eq 'qwen36' })
Check 'three qwen36 table rows' ($q36.Count -eq 3)
foreach ($m in $script:MODELS) {
    $own = -not ($m.PSObject.Properties['MmprojRepo'] -and $m.MmprojRepo)
    if ($m.Key -eq 'qwen36') {
        Check "qwen36/$($m.Quant) lists mmproj-BF16.gguf" ($m.Mmproj -eq 'mmproj-BF16.gguf')
        $want = 902822624
        if ($m.Repo -eq 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF') { $want = 902822528 }
        Check "qwen36/$($m.Quant) projector size matches the repo it comes from ($($m.Repo))" ($m.MmprojBytes -eq $want)
        Check "qwen36/$($m.Quant) projector comes from the model's own repo" $own
    }
    elseif ($m.Key -eq 'qwen38') {
        Check "qwen38/$($m.Quant) lists mmproj-BF16.gguf (unsloth/Qwen3.8-27B-GGUF)" (($m.Mmproj -eq 'mmproj-BF16.gguf') -and ($m.MmprojBytes -eq 931146432) -and $own -and ($m.Repo -eq 'unsloth/Qwen3.8-27B-GGUF'))
    }
    elseif ($m.Key -eq 'apex35') {
        Check "apex35/$($m.Quant) lists the Apex projector from the bartowski repo (the model's own repo publishes none)" (($m.Mmproj -eq 'mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf') -and ($m.MmprojBytes -eq 902822560) -and ($m.MmprojRepo -eq 'bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF'))
    }
    else {
        Check "$($m.Key)/$($m.Quant) lists no projector (text-only model)" (-not ($m.PSObject.Properties['Mmproj'] -and $m.Mmproj))
    }
}

# --- Test-VisionOffered over every recipe row ----------------------------------------------------------
foreach ($r in $script:RECIPES) {
    $m = @($script:MODELS | Where-Object { $_.Key -eq $r.Model -and $_.Quant -eq $r.Quant })[0]
    if (-not $m) { $m = @($script:MODELS | Where-Object { $_.Key -eq $r.Model -and $_.Default })[0] }
    $want = ($r.Model -ne 'flint')    # SDD D27: every model with a projector, the MTP / draft rows included
    Check "$($r.Id) ($($r.Model)) vision offered = $want" ((Test-VisionOffered -m $m -r $r) -eq $want)
}

# --- Complete-RowFlags with Vision: once, idempotent, and the D22 additions are untouched --------------
foreach ($r in $script:RECIPES) {
    $envText = ''
    if ($r.Env) { $envText = (($r.Env.Keys | ForEach-Object { "$_=$($r.Env[$_])" }) -join ' ') }
    foreach ($cc in @('12.0', '6.1', '')) {
        $off = Complete-RowFlags -Flags $r.Flags -EnvText $envText -ComputeCap $cc
        $on  = Complete-RowFlags -Flags $r.Flags -EnvText $envText -ComputeCap $cc -Vision $true
        Check "$($r.Id) cc='$cc' Vision off: no vision flag" ((Count $off.Flags '--mmproj') -eq 0)
        Check "$($r.Id) cc='$cc' Vision on: --mmproj `$mmproj --mmproj-on-demand exactly once" ((Count $on.Flags '--mmproj \$mmproj --mmproj-on-demand') -eq 1 -and (Count $on.Flags '--mmproj') -eq 2)
        Check "$($r.Id) cc='$cc' Vision on: the rest of the flags equal the Vision-off flags" ($on.Flags -eq ($off.Flags + ' --mmproj $mmproj --mmproj-on-demand'))
        $again = Complete-RowFlags -Flags $on.Flags -EnvText $envText -ComputeCap $cc -Vision $true
        Check "$($r.Id) cc='$cc' idempotent with Vision" ($again.Flags -eq $on.Flags)
    }
}

# --- dry-run end to end (setup.ps1 itself; -DryRun starts nothing, downloads nothing) ----------------------
function DryRun([string[]]$extra) {
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $setupPath -DryRun -NonInteractive -Model qwen36 -Vram 16g -Context 262k -SkipBuild -ModelDir (Join-Path $env:TEMP 'swz-vod-models') -Port 8090 @extra 2>&1 | Out-String
    return [pscustomobject]@{ Text = $out; Code = $LASTEXITCODE }
}
$d = DryRun @()
Check 'dry run (default): exit 0' ($d.Code -eq 0)
Check 'dry run (default): the emitted script defines $mmproj' ($d.Text -match '\$mmproj = ".*mmproj-BF16\.gguf"')
$launch = @($d.Text -split "`n" | Where-Object { $_ -match '^\s*& \$exe ' })
Check 'dry run (default): exactly one launch line, and it carries --mmproj $mmproj --mmproj-on-demand once' (($launch.Count -eq 1) -and ((Count $launch[0] '--mmproj \$mmproj --mmproj-on-demand') -eq 1))
Check 'dry run (default): the projector download is announced from the model''s repo' ($d.Text -match 'projector mmproj-BF16\.gguf from unsloth/Qwen3\.6-35B-A3B-GGUF')
$d = DryRun @('-NoVision')
Check 'dry run -NoVision: exit 0, no vision flag, no projector line' (($d.Code -eq 0) -and ((Count $d.Text '--mmproj') -eq 0) -and ($d.Text -notmatch 'mmproj-BF16'))
$d = DryRun @('-MmprojFile', (Join-Path $env:TEMP 'swz-vod-does-not-exist.gguf'))
Check 'dry run -MmprojFile with a missing file: exit 1' ($d.Code -eq 1)
$fakeProj = Join-Path $env:TEMP 'swz-vod-fake-mmproj.gguf'
Set-Content -Path $fakeProj -Value 'x'
$d = DryRun @('-MmprojFile', $fakeProj)
Check 'dry run -MmprojFile: the local file is used and nothing is downloaded' (($d.Code -eq 0) -and ($d.Text -match [regex]::Escape($fakeProj)) -and ($d.Text -notmatch 'not downloading the projector'))
$d = DryRun @('-MmprojFile', $fakeProj, '-NoVision')
Check 'dry run -MmprojFile with -NoVision: exit 1 (contradiction)' ($d.Code -eq 1)
Remove-Item $fakeProj -Force -ErrorAction SilentlyContinue
function DryRunFor([string]$model, [string]$vram, [string]$ctx, [string]$quant, [string[]]$extra) {
    $a = @('-DryRun', '-NonInteractive', '-Model', $model, '-Vram', $vram, '-Context', $ctx, '-SkipBuild', '-ModelDir', (Join-Path $env:TEMP 'swz-vod-models'), '-Port', '8090')
    if ($quant) { $a += @('-Quant', $quant) }
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $setupPath @a @extra 2>&1 | Out-String
    return [pscustomobject]@{ Text = $out; Code = $LASTEXITCODE }
}
# SDD D27: every model entry with a projector, default (vision on) and -NoVision; the MTP row R5 and the dense rows included
$cases = @(
    @{ M='qwen38'; V='16g';  C='262k'; Q='q4k_m';  File='mmproj-BF16.gguf'; Repo='unsloth/Qwen3.8-27B-GGUF' },
    @{ M='qwen38'; V='full'; C='262k'; Q='q5k_xl'; File='mmproj-BF16.gguf'; Repo='unsloth/Qwen3.8-27B-GGUF' },
    @{ M='qwen36'; V='full'; C='262k'; Q='q8k_xl'; File='mmproj-BF16.gguf'; Repo='unsloth/Qwen3.6-35B-A3B-MTP-GGUF' },
    @{ M='apex35'; V='full'; C='262k'; Q='q5k_m';  File='mmproj-Apexintelligence-AI_Apex-35B-V1.0-bf16.gguf'; Repo='bartowski/Apexintelligence-AI_Apex-35B-V1.0-GGUF' }
)
foreach ($c in $cases) {
    $tag = "$($c.M)/$($c.Q) at $($c.V)"
    $d = DryRunFor $c.M $c.V $c.C $c.Q @()
    $launch = @($d.Text -split "`n" | Where-Object { $_ -match '^\s*& \$exe ' })
    Check "dry run $tag (default): exit 0, one launch line carrying --mmproj `$mmproj --mmproj-on-demand once" (($d.Code -eq 0) -and ($launch.Count -eq 1) -and ((Count $launch[0] '--mmproj \$mmproj --mmproj-on-demand') -eq 1))
    Check "dry run $tag (default): the download is announced as $($c.File) from $($c.Repo)" ($d.Text -match [regex]::Escape("projector $($c.File) from $($c.Repo)"))
    $d = DryRunFor $c.M $c.V $c.C $c.Q @('-NoVision')
    Check "dry run $tag -NoVision: exit 0, no vision flag, no projector line" (($d.Code -eq 0) -and ((Count $d.Text '--mmproj') -eq 0) -and ($d.Text -notmatch 'projector'))
}
# the emitted script says when --mmproj-on-demand is not a measured combination (Qwen3.6-35B-A3B and its Apex fine-tune are)
$d = DryRunFor 'qwen38' '16g' '262k' 'q4k_m' @()
Check 'dry run qwen38: the emitted script says that vision is not measured on this model' ($d.Text -match 'vision on this model is NOT measured')
$d = DryRunFor 'qwen36' '16g' '262k' 'q4k_m' @()
Check 'dry run qwen36: no "not measured" line' ($d.Text -notmatch 'vision on this model is NOT measured')
$d = DryRunFor 'apex35' 'full' '262k' 'q5k_m' @()
Check 'dry run apex35: no "not measured" line (same architecture as Qwen3.6-35B-A3B)' ($d.Text -notmatch 'vision on this model is NOT measured')
# flint is text-only: no projector is published anywhere
$d = DryRunFor 'flint' 'full' '262k' 'q5k_xl' @()
Check 'dry run flint: exit 0, "vision: not available", no vision flag' (($d.Code -eq 0) -and ($d.Text -match 'vision: not available for flint') -and ((Count $d.Text '--mmproj') -eq 0))
$d = DryRunFor 'flint' 'full' '262k' 'q5k_xl' @('-MmprojFile', (Join-Path $env:TEMP 'swz-vod-does-not-exist.gguf'))
Check 'dry run flint -MmprojFile: exit 1 (no projector is published for it)' ($d.Code -eq 1)
$out = & powershell -NoProfile -ExecutionPolicy Bypass -File $setupPath --dry-run --non-interactive --no-vision --model qwen36 --vram 16g --context 262k --skip-build --model-dir (Join-Path $env:TEMP 'swz-vod-models') --port 8090 2>&1 | Out-String
Check 'dashed spelling --no-vision is accepted' (($LASTEXITCODE -eq 0) -and ((Count $out '--mmproj') -eq 0))

Write-Host ''
if ($fail -eq 0) { Write-Host "test-vod.ps1: all checks passed" -ForegroundColor Green; exit 0 }
Write-Host "test-vod.ps1: $fail check(s) FAILED" -ForegroundColor Red
exit 1
