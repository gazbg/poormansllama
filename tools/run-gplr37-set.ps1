# T-GPLR-37 set driver: the five arms, one detached mutex session.
#
#   C1  4k control default  -> pinned, NO new keys, NO announcement  (INV1)
#   A1  4k changed default  -> mmap-registered, source=default, announced,
#                              40 init-once registers before first compute,
#                              decode PASS 5/5, parity clean
#   A2  4k changed explicit --copy-source pinned  -> byte-identical BEHAVIOR
#                              to C1 (pair identity), no announcement
#   A3  4k CONTROL explicit --copy-source pinned  -> the explicit pinned pair
#                              for A2 (identity across binaries: same flags,
#                              same work, both binaries agree)
#   A4  b1 changed default  -> mmap-registered on the 143k q8 shape (the
#                              recorded T-47 clean reference shape)
#
# Timing is recorded but never gated: the A1-vs-A2 comparison IS the measured
# effect of the default (same binary, same session, same shape, only the flag
# differs).
param()
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr37-arm.ps1'
$tag  = 'gplr37-set'

. (Join-Path $root 'tools\arm-verdict.ps1')
$verdictDir = Join-Path $root 'build\llama-cuda\verdicts'
Clear-ArmVerdict -Dir $verdictDir -Tag $tag | Out-Null
Write-ArmStart -Dir $verdictDir -Tag $tag -MutexName 'Global\AI-GPU-EXCLUSIVE-v1' -LogPath (Join-Path $root 'build\llama-cuda\gplr37-set.log') | Out-Null

Write-Host '=== C1: control binary, bare default (INV1: no keys, no announcement) ==='
& powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Tag gplr37-c1-control-default -Bin control -Shape 4k -Src default
Write-Host ("C1 EXIT {0}" -f $LASTEXITCODE)

Write-Host '=== A3: CONTROL explicit pinned (the pair anchor) ==='
& powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Tag gplr37-a3-control-pinned -Bin control -Shape 4k -Src pinned
Write-Host ("A3 EXIT {0}" -f $LASTEXITCODE)

Write-Host '=== A1: changed binary, bare default (the new default measured) ==='
& powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Tag gplr37-a1-changed-default -Bin changed -Shape 4k -Src default
Write-Host ("A1 EXIT {0}" -f $LASTEXITCODE)

Write-Host '=== A2: changed binary, explicit pinned (identity vs C1/A3) ==='
& powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Tag gplr37-a2-changed-pinned -Bin changed -Shape 4k -Src pinned -CompareTag gplr37-c1-control-default
Write-Host ("A2 EXIT {0}" -f $LASTEXITCODE)

Write-Host '=== A4: changed binary, b1 143k q8, bare default ==='
& powershell -NoProfile -ExecutionPolicy Bypass -File $arm -Tag gplr37-a4-b1-default -Bin changed -Shape b1 -Src default
Write-Host ("A4 EXIT {0}" -f $LASTEXITCODE)

# ---- set summary
$tags = @('gplr37-c1-control-default','gplr37-a3-control-pinned','gplr37-a1-changed-default','gplr37-a2-changed-pinned','gplr37-a4-b1-default')
$allPass = $true
foreach ($t in $tags) {
    $v = Read-ArmVerdict -Dir $verdictDir -Tag $t
    $st = if ($null -eq $v) { 'MISSING' } else { $v.status }
    Write-Host ("  {0,-32} {1}" -f $t, $st)
    if ($st -ne 'PASS') { $allPass = $false }
}
Write-Host ("ALL ARMS: {0}" -f $(if ($allPass) { 'PASS' } else { 'FAIL' }))

Write-ArmVerdict -Dir $verdictDir -Tag $tag -ExitCode $(if ($allPass) { 0 } else { 1 }) `
    -StartedUtc ((Get-Date).ToUniversalTime().ToString('o')) -LogPath (Join-Path $root 'build\llama-cuda\gplr37-set.log') `
    -Fields @{ arms = ($tags -join ','); all_pass = $allPass } | Out-Null
Write-Host 'DONE'
exit $(if ($allPass) { 0 } else { 1 })
