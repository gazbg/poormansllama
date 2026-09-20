# T-GPLR-39 arm set: three arms back-to-back in ONE detached job (one mutex
# acquisition, same session -> timing comparable within the set).
#
#   D1  default  (changed binary, --route-telemetry UNSET)  - the new default:
#       resolves 0 with source 'default', announcement, ZERO route events
#   E1  on       (changed binary, --route-telemetry 1)      - explicit opt-in:
#       resolves 1 with source 'explicit', no announcement, route events present,
#       parity clean, tg ids identical to D1 (the route diagnostic must not
#       change the stream)
#   C1  default  (control = the T-GPLR-37 RESULT binary)    - INV1: carries no
#       route_telemetry key and no announcement at bare default
#
# Identity anchors: E1's tg ids must equal the recorded T-GPLR-37 default arm
# (gplr37-ron-s3 / A1) as well as D1's.
param(
    [ValidateSet('changed', 'control')] [string] $Bin = 'changed'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$ps   = Join-Path $root 'tools\run-gplr39-arm.ps1'

& $ps -Tag 'gplr39-D1-default' -Bin changed  -Mode default
& $ps -Tag 'gplr39-E1-on'      -Bin changed  -Mode on -CompareTag 'gplr39-D1-default'
& $ps -Tag 'gplr39-C1-control' -Bin control  -Mode default
exit $LASTEXITCODE
