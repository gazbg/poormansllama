# T-GPLR-47 validation set: the contract's V1/V2/V4/V3 arms, one mutex hold per
# arm (the arm runner itself acquires the mutex), launched detached per the
# AGENT_ENVIRONMENT detached-job conventions (Start-Process, NOT setsid; the
# driver dot-sources arm-verdict.ps1 and runs Clear-ArmVerdict for each tag
# BEFORE Start-Process).
#
# Arm plan (contract "Validation (exact)"):
#   V1: 4k layer-major f16 rebuild-ON pair fresh-vs-control (ids, AC1, AC7)
#       plus a same-session rebuild-OFF arm for the byte-identity reference.
#   V2: 50k ON/OFF pair on the fresh binary (AC2) - the T-GPLR-46 recorded shape.
#   V4: q8 B1 rebuild-ON (AC4) - the T-GPLR-43 arm shape.
#   V3: 200k rebuild-ON on the fresh binary with --pp-recon 1 (AC3).
#
# Order note: V1 first (identity + semantics on the smallest shape), then the
# OFF references, then 50k, B1, 200k (longest last).
param([string] $LogPath = '')
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr47-arm.ps1'
$base = Join-Path $root 'build\llama-cuda'
if (-not $LogPath) { $LogPath = Join-Path $base 'gplr47-set.log' }
. (Join-Path $root 'tools\arm-verdict.ps1')
Start-Transcript -Path $LogPath -Force | Out-Null

$plan = @(
    # ---- V1: 4k trio
    @{ Tag = 'gplr47-v1-4k-on-control';  Shape = '4k';  Bin = 'control'; Rebuild = '1' }
    @{ Tag = 'gplr47-v1-4k-on-changed';  Shape = '4k';  Bin = 'changed'; Rebuild = '1' }
    @{ Tag = 'gplr47-v1-4k-off-changed'; Shape = '4k';  Bin = 'changed'; Rebuild = '0' }
    # ---- V2: 50k pair on the fresh binary
    @{ Tag = 'gplr47-v2-50k-on';         Shape = '50k'; Bin = 'changed'; Rebuild = '1' }
    @{ Tag = 'gplr47-v2-50k-off';        Shape = '50k'; Bin = 'changed'; Rebuild = '0' }
    # ---- V4: q8 B1 rebuild ON (the T-GPLR-43 rescue must survive the swap)
    @{ Tag = 'gplr47-v4-b1-q8-on';       Shape = 'b1';  Bin = 'changed'; Rebuild = '1' }
    # ---- V3: 200k confirmation, rebuild ON
    @{ Tag = 'gplr47-v3-200k-on';        Shape = '200k'; Bin = 'changed'; Rebuild = '1' }
)

$failed = @()
foreach ($s in $plan) {
    $armArgs = @('-Tag', $s.Tag, '-Shape', $s.Shape, '-Bin', $s.Bin, '-Rebuild', $s.Rebuild)
    Write-Host ("[gplr47-set] ===== {0} =====" -f $s.Tag)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm @armArgs
    $code = $LASTEXITCODE
    Write-Host ("[gplr47-set] arm {0} exit: {1}" -f $s.Tag, $code)
    if ($code -ne 0) { $failed += $s.Tag }
}
if ($failed.Count -gt 0) { Write-Host ("[gplr47-set] FAILED arms: {0}" -f ($failed -join ', ')) }
else { Write-Host '[gplr47-set] ALL ARMS PASSED' }
Stop-Transcript | Out-Null
exit $(if ($failed.Count -gt 0) { 1 } else { 0 })
