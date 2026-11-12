# tools/setup-wizard/test-d28.ps1
# Host-only test for SDD D28 in setup.ps1 (no GPU, no build, no model, no download): the emitted run-server.ps1 carries a
# default cap of 16000 new tokens per response (-n), the $maxTokens variable at the top, and a notice line printed just
# before the server starts that tells the user to edit it.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File tools\setup-wizard\test-d28.ps1
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$repo = Split-Path -Parent $repo
$setupPath = Join-Path $repo 'setup.ps1'

$cutAt = (Select-String -Path $setupPath -Pattern '^# --- main' | Select-Object -First 1).LineNumber
if (-not $cutAt) { Write-Host '[FAIL] main marker not found in setup.ps1' -ForegroundColor Red; exit 1 }
$cutFile = Join-Path $env:TEMP 'swz-d28-cut.ps1'
Set-Content -Path $cutFile -Value (Get-Content $setupPath | Select-Object -First ($cutAt - 1))
. $cutFile

$fail = 0
function Check([string]$what, [bool]$cond) {
    if ($cond) { Write-Host "  PASS $what" }
    else { Write-Host "  FAIL $what" -ForegroundColor Red; $script:fail++ }
}
function Count([string]$text, [string]$lit) { return @([regex]::Matches($text, [regex]::Escape($lit))).Count }

# --- Test-HasPredictFlag (pure)
Check 'Test-HasPredictFlag: -n 4000'          (Test-HasPredictFlag '-c 262144 -ctk q8_0 -n 4000 -fa on')
Check 'Test-HasPredictFlag: --n-predict 4000' (Test-HasPredictFlag '-c 1 --n-predict 4000')
Check 'Test-HasPredictFlag: --predict=4000'   (Test-HasPredictFlag '-c 1 --predict=4000')
Check 'Test-HasPredictFlag: -n=4000'          (Test-HasPredictFlag '-c 1 -n=4000')
Check 'Test-HasPredictFlag: -np / -ngl / -ncmoe / --no-warmup / -nkvo are not -n' (-not (Test-HasPredictFlag '-c 262144 -np 1 -ngl 99 -ncmoe 37 --no-warmup -t 16 -nkvo'))
Check 'Test-HasPredictFlag: empty flags'      (-not (Test-HasPredictFlag ''))

# --- every kind of row: the emitted script (dry run prints it)
$cases = @(
    @{ M='qwen38'; V='16g';  C='262k'; Q='q4k_m' },
    @{ M='qwen38'; V='8g';   C='128k'; Q='q4k_m' },
    @{ M='flint';  V='full'; C='262k'; Q='q5k_xl' },
    @{ M='qwen36'; V='8g';   C='262k'; Q='q4k_m' },
    @{ M='qwen36'; V='16g';  C='262k'; Q='q8k_xl' },
    @{ M='apex35'; V='full'; C='262k'; Q='q5k_m' }
)
foreach ($c in $cases) {
    $tag = "$($c.M)/$($c.Q) at $($c.V)"
    $out = & powershell -NoProfile -ExecutionPolicy Bypass -File $setupPath -DryRun -NonInteractive -Model $c.M -Vram $c.V -Context $c.C -Quant $c.Q -SkipBuild -ModelDir (Join-Path $env:TEMP 'swz-d28-models') -Port 8090 2>&1 | Out-String
    $code = $LASTEXITCODE
    $lines = @($out -split "`n" | ForEach-Object { $_.TrimEnd("`r") })
    Check "${tag}: dry run exit 0" ($code -eq 0)
    Check "${tag}: `$maxTokens = 16000 defined once at the top of the script" ((Count $out '$maxTokens = 16000') -eq 1)
    $launchIdx = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -match '^\s*& \$exe ' })
    Check "${tag}: one launch line, carrying -n `$maxTokens once" (($launchIdx.Count -eq 1) -and ((Count $lines[$launchIdx[0]] '-n $maxTokens') -eq 1))
    $noticeIdx = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -match 'Write-Host "max new tokens per response: \$maxTokens' })
    $defIdx = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -match '^\s*\$maxTokens = 16000' })
    Check "${tag}: the definition comes first, the edit notice directly before the launch line" (($noticeIdx.Count -eq 1) -and ($defIdx.Count -eq 1) -and ($launchIdx.Count -eq 1) -and ($defIdx[0] -lt $noticeIdx[0]) -and ($noticeIdx[0] -lt $launchIdx[0]) -and (($launchIdx[0] - $noticeIdx[0]) -le 2))
    Check "${tag}: the notice tells the user to edit `$maxTokens" (($noticeIdx.Count -eq 1) -and ($lines[$noticeIdx[0]] -match 'Edit `\$maxTokens at the top of this script if you need more'))
}

Write-Host ''
if ($fail -eq 0) { Write-Host "test-d28.ps1: all checks passed" -ForegroundColor Green; exit 0 }
Write-Host "test-d28.ps1: $fail check(s) FAILED" -ForegroundColor Red
exit 1
