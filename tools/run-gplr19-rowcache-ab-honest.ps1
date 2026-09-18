# SUPERSEDED 2026-09-18 -- FORWARDS to tools/run-gplr19-rowcache-ab.ps1.
#
# This file was the honest-vehicle A/B driver that produced the K=6 and K=8
# evidence recorded in docs/currentstate.md (2026-09-18, "ROW-CACHE A/B ON THE
# HONEST STREAM") and on disk under build/llama-cuda/gplr19-honest-k6-ab plus the
# nested k8 root (see that entry for why the k8 evidence is nested).
#
# The non-echoing vehicle is now the DEFAULT of the unified driver, so this
# script does not implement an arm loop any more: keeping ONE implementation is
# what stops the honest path and the recorded path from drifting apart, which is
# the failure this whole vehicle change exists to prevent.
#
# Equivalent invocations (any of these runs the honest vehicle at K=6):
#   tools/run-gplr19-rowcache-ab.ps1 -RetainK 6                    # default vehicle
#   tools/run-gplr19-rowcache-ab.ps1 -RetainK 6 -Vehicle honest    # explicit
#
# The shim is kept rather than deleted so the recorded evidence keeps a
# resolvable producer path and any old command line keeps working. Use the
# unified driver for anything new.
param(
    # Depths to run, as a comma or space separated string (e.g. '6,8' or '6').
    [string] $Rungs = '6,8',

    # Optional base output root; each rung is written under its own subdirectory.
    [string] $OutRoot = ''
)
$ErrorActionPreference = 'Stop'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$driver = Join-Path $root 'tools\run-gplr19-rowcache-ab.ps1'
if (-not (Test-Path -LiteralPath $driver)) { throw "missing unified driver: $driver" }

Write-Host 'NOTE: run-gplr19-rowcache-ab-honest.ps1 is superseded; forwarding to'
Write-Host '      run-gplr19-rowcache-ab.ps1 -Vehicle honest (now that vehicle is the default).'

$ks = @($Rungs -split '[,\s]+' | Where-Object { $_ -ne '' })
if ($ks.Count -eq 0) { throw "-Rungs gave no depths; got '$Rungs'" }

foreach ($k in $ks) {
    $driverArgs = @('-RetainK', "$k", '-Vehicle', 'honest')
    if ($OutRoot) { $driverArgs += @('-OutRoot', (Join-Path $OutRoot "gplr19-honest-k$k-ab")) }
    Write-Host ("--- forwarding K={0}: run-gplr19-rowcache-ab.ps1 {1}" -f $k, ($driverArgs -join ' '))
    & powershell -NoProfile -ExecutionPolicy Bypass -File $driver @driverArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("FORWARDED RUN FAILED at K={0} with exit {1}" -f $k, $LASTEXITCODE)
        exit $LASTEXITCODE
    }
}
Write-Host 'ALL FORWARDED RUNGS COMPLETED'
exit 0
