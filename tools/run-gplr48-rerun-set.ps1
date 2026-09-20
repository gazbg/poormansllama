# T-GPLR-48 RE-RUN driver: the three arms whose FIRST attempt was corrupted by the
# LAUNCHER, not by the engine.
#
# What happened (measured 2026-09-19): the first re-run was launched as
# `(setsid nohup bash script.sh &)`. Under that wrapper PowerShell does NOT wait
# for a native child: `& $exe @a > $log 2>&1` returns in 0 s, `$LASTEXITCODE` is
# $null and the child's redirected output never reaches the log -- the engine
# then runs on, orphaned, and the harness verdicts a run that is still in
# flight. Reproduced host-only in three shapes with the same 6 s child:
#   direct                exit=0   dur=5.1s logbytes=1056
#   setsid nohup bash -c  exit=<>  dur=0.0s logbytes=0     <-- the broken shape
#   plain nohup bash -c   exit=0   dur=5.1s logbytes=1056
# So `setsid` is the culprit, and this driver is launched with Start-Process
# (no setsid, no -Wait) instead. The engine work in those three arms was
# discarded rather than re-read, because the verdict could not observe it.
param([string] $LogPath = '')
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr48-arm.ps1'
$base = Join-Path $root 'build\llama-cuda'
if (-not $LogPath) { $LogPath = Join-Path $base 'gplr48-rerun2.log' }
Start-Transcript -Path $LogPath -Force | Out-Null

$plan = @(
    @{ Tag = 'gplr48-a3-50k-q8-pbp0-control'; Shape = '50k'; Kv = 'q8';  Rebuild = '0'; Bin = 'control'; Compare = ''; ExpectCrash = $true }
    @{ Tag = 'gplr48-a3-50k-q8-pbp0-changed'; Shape = '50k'; Kv = 'q8';  Rebuild = '0'; Bin = 'changed'; Compare = 'gplr48-a3-50k-q8-pbp0-control'; ExpectCrash = $true }
    @{ Tag = 'gplr48-a3-50k-f16-pbp1-changed'; Shape = '50k'; Kv = 'f16'; Rebuild = '1'; Bin = 'changed'; Compare = 'gplr48-a3-50k-f16-pbp1-control'; ExpectCrash = $false }
)

$failed = @()
foreach ($s in $plan) {
    $armArgs = @('-Tag', $s.Tag, '-Shape', $s.Shape, '-Kv', $s.Kv, '-Rebuild', $s.Rebuild, '-Bin', $s.Bin)
    if ($s.Compare)     { $armArgs += @('-CompareTag', $s.Compare) }
    if ($s.ExpectCrash) { $armArgs += '-ExpectCrash' }
    Write-Host ("[gplr48-rerun] ===== {0} =====" -f $s.Tag)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm @armArgs
    $code = $LASTEXITCODE
    Write-Host ("[gplr48-rerun] arm {0} exit: {1}" -f $s.Tag, $code)
    if ($code -ne 0) { $failed += $s.Tag }
}
if ($failed.Count -gt 0) { Write-Host ("[gplr48-rerun] FAILED arms: {0}" -f ($failed -join ', ')) }
else { Write-Host '[gplr48-rerun] all three arms PASSED' }
Stop-Transcript | Out-Null
exit $(if ($failed.Count -gt 0) { 1 } else { 0 })
