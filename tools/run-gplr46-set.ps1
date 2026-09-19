# T-GPLR-46 driver: the five arms back-to-back, ONE session. Order follows the
# contract: the trigger A/B first (gate decides whether the rest may run), then
# the growth-law pair and the saturation arm.
#
#   A 50k-on     default reference      (excess 3.0-4.6 s expected)
#   B 50k-off    rebuild off            (gate: excess <= 0.5 s -> rebuild confirmed)
#   C 100k-on    growth law             (band 8.7-15.6 s)
#   E 100k-off   growth-law control     (<= 0.5 s)
#   D 50k-3pass  saturation test        (pass3-pass2 vs pass2-pass1, GROWING or not)
#
# Run this DETACHED; completion is signalled by the five gplr46-*.done sentinels.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr46-rebuild-ab.ps1'
if (-not (Test-Path -LiteralPath $arm)) { Write-Host "MISSING: $arm"; exit 1 }

$arms = @('50k-on', '50k-off', '100k-on', '100k-off', '50k-3pass')
$verdictDir = Join-Path (Join-Path $root 'build\llama-cuda') 'verdicts'

foreach ($name in $arms) {
    Write-Host ("[gplr46] ===== arm {0} =====" -f $name)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Arm $name
    $code = $LASTEXITCODE
    Write-Host ("[gplr46] arm {0} exit: {1}" -f $name, $code)
    if ($name -eq '50k-on' -and $code -ne 0) {
        Write-Host '[gplr46] reference arm failed; stopping (gate cannot be evaluated)'
        exit 2
    }
    if ($name -eq '50k-off' -and $code -ne 0) {
        Write-Host '[gplr46] trigger arm failed; stopping before the growth-law arms'
        exit 3
    }
    if ($name -eq '50k-off') {
        $v = Get-Content (Join-Path $verdictDir 'gplr46-50k-off.verdict.json') -Raw | ConvertFrom-Json
        $ex = [double]$v.fields.excess_2v1_s
        Write-Host ("[gplr46] GATE: 50k-off excess {0:N2} s" -f $ex)
        if ($ex -gt 0.5) {
            Write-Host '[gplr46] GATE REFUTES the rebuild as the trigger (excess > 0.5 s with rebuild off); stopping per contract - escalate to the binary bisect.'
            exit 4
        }
        Write-Host '[gplr46] GATE PASSED: rebuild confirmed as the trigger; proceeding to the growth-law arms.'
    }
}
Write-Host '[gplr46] all five arms finished'
