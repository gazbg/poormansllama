# tools/setup-wizard/test-d24.ps1
# Host-only unit test for SDD D24: every MoE model shares the 8g profile (R7), whatever its name and
# whether or not it has a benchmark. No GPU, no build, no model: cut setup.ps1 at the '# --- main'
# marker, dot-source the function part, and check RecipeFor / IsMoeModel against a matrix.
#
# Run: pwsh -NoProfile -File tools\setup-wizard\test-d24.ps1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$setupPath = Join-Path $repo 'setup.ps1'

$cutAt = (Select-String -Path $setupPath -Pattern '^# --- main' | Select-Object -First 1).LineNumber
if (-not $cutAt) { Write-Host '[FAIL] main marker not found in setup.ps1' -ForegroundColor Red; exit 1 }
$cutFile = Join-Path $env:TEMP 'swz-d24-cut.ps1'
Set-Content -Path $cutFile -Value (Get-Content $setupPath | Select-Object -First ($cutAt - 1))
. $cutFile

$fail = 0
$total = 0
function Check([string]$what, [bool]$cond) {
    $script:total++
    if ($cond) { Write-Host "  PASS $what" }
    else { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fail++ }
}
function Case([string]$m, [string]$q, [string]$v, [string]$c, [string]$want) {
    $r = RecipeFor $m $v $c $q
    $got = 'none'
    if ($r) { $got = $r.Id }
    Check "$m/$q $v $c -> $want (got $got)" ($got -eq $want)
}

# which models are MoE
Check 'qwen38 is not MoE' (-not (IsMoeModel 'qwen38'))
Check 'qwen36 is MoE' (IsMoeModel 'qwen36')
Check 'flint is MoE' (IsMoeModel 'flint')
Check 'apex35 is MoE' (IsMoeModel 'apex35')
Check 'unknown model is not MoE' (-not (IsMoeModel 'nosuch'))

# 8g, 262k: every MoE model gets R7 (exact, default-quant fallback, or the shared profile)
Case 'qwen36' 'q4k_m'  '8g' '262k' 'R7'
Case 'qwen36' 'q5k_xl' '8g' '262k' 'R7'
Case 'qwen36' 'q8k_xl' '8g' '262k' 'R7'
Case 'flint'  'q5k_xl' '8g' '262k' 'R7'
Case 'apex35' 'q5k_m'  '8g' '262k' 'R7'

# the profile is the same row object for every MoE model
$ids = @(@('qwen36','q4k_m'), @('flint','q5k_xl'), @('apex35','q5k_m')) | ForEach-Object { (RecipeFor $_[0] '8g' '262k' $_[1]).Flags } | Select-Object -Unique
Check 'all MoE models at 8g/262k get the identical flag text' (@($ids).Count -eq 1)

# not widened: other targets and contexts keep their rows or refuse
Case 'apex35' 'q5k_m'  '8g'   '128k' 'none'
Case 'flint'  'q5k_xl' '8g'   '128k' 'none'
Case 'apex35' 'q5k_m'  '16g'  '262k' 'none'
Case 'flint'  'q5k_xl' '16g'  '262k' 'none'
Case 'apex35' 'q5k_m'  'full' '262k' 'R9'
Case 'flint'  'q5k_xl' 'full' '262k' 'R4'
Case 'qwen36' 'q4k_m'  '16g'  '262k' 'R8'
Case 'qwen36' 'q8k_xl' 'full' '262k' 'R5'

# D25: the apex35 Q6_K_L build is no longer offered (the flat list has 7 entries, picks are 1-7)
Check 'model table: 7 entries' (@($script:MODELS).Count -eq 7)
Check 'model table: no q6k_l entry' (-not (@($script:MODELS | Where-Object { $_.Quant -eq 'q6k_l' }).Count))
Check 'model table: apex35 has only q5k_m' ((@($script:MODELS | Where-Object { $_.Key -eq 'apex35' } | ForEach-Object { $_.Quant }) -join ',') -eq 'q5k_m')

# dense models are not MoE: their 8g row stays R3 (128k only)
Case 'qwen38' 'q4k_m'  '8g' '128k' 'R3'
Case 'qwen38' 'q4k_m'  '8g' '262k' 'none'
Case 'qwen38' 'q5k_xl' '8g' '262k' 'none'

# an unknown model never gets a profile
Case 'nosuch' 'q4k_m' '8g' '262k' 'none'

Write-Host ''
if ($fail -gt 0) { Write-Host "D24 PS: $fail of $total check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "D24 PS: all $total checks passed"
exit 0
