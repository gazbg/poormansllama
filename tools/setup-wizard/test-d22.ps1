# tools/setup-wizard/test-d22.ps1
# Host-only unit test for the D22 row-flag completion (no GPU, no build,
# no model): cut setup.ps1 at the '# --- main' marker, dot-source the
# function part, and check Complete-RowFlags / Detect-GpuArch against every
# recipe row and a matrix of compute capabilities.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\test-d22.ps1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$setupPath = Join-Path $repo 'setup.ps1'

$cutAt = (Select-String -Path $setupPath -Pattern '^# --- main' | Select-Object -First 1).LineNumber
if (-not $cutAt) { Write-Host '[FAIL] main marker not found in setup.ps1' -ForegroundColor Red; exit 1 }
$cutFile = Join-Path $env:TEMP 'swz-d22-cut.ps1'
Set-Content -Path $cutFile -Value (Get-Content $setupPath | Select-Object -First ($cutAt - 1))
. $cutFile

$fail = 0
function Check([string]$what, [bool]$cond) {
    if ($cond) { Write-Host "  PASS $what" }
    else { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fail++ }
}
function PrearmAlways([string]$f) {
    # exactly one --lm-prefill-prearm, and its value is always
    $hits = @([regex]::Matches($f, '--lm-prefill-prearm\s+(\S+)'))
    if ($hits.Count -ne 1) { return $false }
    return $hits[0].Groups[1].Value -eq 'always'
}
function HasInt8([string]$f) { return $f -split '\s+' | Where-Object { $_ -eq '--lm-prefill-attn-int8' } | Select-Object -First 1 }

Write-Host "  rows: $($script:RECIPES.Count), main marker at line $cutAt"

foreach ($r in $script:RECIPES) {
    $envText = ''
    if ($r.Env) { $envText = (($r.Env.Keys | ForEach-Object { "$_=$($r.Env[$_])" }) -join ' ') }
    foreach ($cc in @('12.0', '8.9', '6.1', '5.2', '3.5', '')) {
        $res = Complete-RowFlags -Flags $r.Flags -EnvText $envText -ComputeCap $cc
        Check ("$($r.Id) cc='$cc' prearm=always, exactly once") (PrearmAlways $res.Flags)
        $wantInt8 = @('6.1', '5.2', '3.5') -contains $cc
        if ($wantInt8) { Check ("$($r.Id) cc='$cc' int8 flag added") ($res.AddedInt8 -and (HasInt8 $res.Flags)) }
        else { Check ("$($r.Id) cc='$cc' no int8 flag") ((-not $res.AddedInt8) -and (-not (HasInt8 $res.Flags))) }
        # idempotence: completing the completed output changes nothing
        $res2 = Complete-RowFlags -Flags $res.Flags -EnvText $envText -ComputeCap $cc
        Check ("$($r.Id) cc='$cc' idempotent") ($res2.Flags -eq $res.Flags)
    }
}

# the "env already set" branch (AC-3): no current row carries the env, so
# drive the pure function with a synthetic row
$oldFlags = $script:RECIPES[6].Flags
$res = Complete-RowFlags -Flags $oldFlags -EnvText 'LLAMA_FATTN_TILE_Q8QK=1 LLAMA_FATTN_GQA8_MAXROWS=4' -ComputeCap '6.1'
Check 'env-set: no int8 flag appended' (-not $res.AddedInt8 -and -not (HasInt8 $res.Flags))
Check 'env-set: prearm still normalized' (PrearmAlways $res.Flags)
$res = Complete-RowFlags -Flags $oldFlags -EnvText 'LLAMA_FATTN_TILE_Q8QK=1' -ComputeCap '12.0'
Check 'env-set + new arch: flags unchanged except prearm' ((PrearmAlways $res.Flags) -and (-not (HasInt8 $res.Flags)))

# Detect-GpuArch test hook
$env:SETUP_WIZARD_GPU_CC = '6.1'
Check 'Detect-GpuArch honors the hook' ((Detect-GpuArch) -eq '6.1')
Remove-Item env:SETUP_WIZARD_GPU_CC

Write-Host ''
if ($fail -gt 0) { Write-Host "D22 PS: $fail check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "D22 PS: all $($script:RECIPES.Count * 6 + 5) checks passed"
exit 0
