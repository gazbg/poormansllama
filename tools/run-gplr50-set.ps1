# T-GPLR-50 arm set: the validation ladder's GPU arms, back-to-back in ONE
# session under ONE mutex acquisition per arm (the arm runner acquires its own
# mutex; sequential launches keep session spread comparable for pairs).
#
#   AC2a  gplr50-ac2-pw-off    changed binary, 4k, f16, pw (default), --pp-prebuild-per-pass 0
#         -> the echo trigger WITHOUT rebuild: must decode PASS (T-49b already
#            showed this; re-proves it on the current tree)
#   AC2b  gplr50-ac2-pw-on     changed binary, 4k, f16, pw (default), rebuild ON (default after flip)
#         -> the echo trigger WITH the fresh-context rebuild: the open question.
#            PASS here = AC2 discharged, flip stands. FAIL = escalate per contract.
#   AC1/AC5a gplr50-4k-f16-default  changed, 4k, f16, default -> resolved 1/default,
#            T-50 announcement, decode PASS, ids == AC2b's (same effective config)
#   AC4a  gplr50-50k-f16-default    changed, 50k, f16, default -> neutral-or-better ledger
#   AC4b  gplr50-b1-q8-default      changed, b1, q8, default  -> quantized path intact
#   INV1  gplr50-ctl-4k-f16-default control binary, 4k, f16, default -> resolved 0/default,
#            T-48 wording, NO T-50 marker
#   INV1b gplr50-ctl-4k-q8-default  control binary, 4k, q8, default -> resolved 1/default
#
# The 200k confirmation (AC3 pair) is NOT in this set: it is owner-invoked,
# run after this set passes, as a separate explicit pair.
param(
    [string[]] $Only = @()
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$arm  = Join-Path $root 'tools\run-gplr50-arm.ps1'

$arms = @(
    @{ Tag = 'gplr50-ac2-pw-off';       Bin = 'changed'; Shape = '4k';  Kv = 'f16'; Rebuild = '0' },
    @{ Tag = 'gplr50-ac2-pw-on';        Bin = 'changed'; Shape = '4k';  Kv = 'f16'; Rebuild = 'default' },
    @{ Tag = 'gplr50-4k-f16-default';   Bin = 'changed'; Shape = '4k';  Kv = 'f16'; Rebuild = 'default' },
    @{ Tag = 'gplr50-50k-f16-default';  Bin = 'changed'; Shape = '50k'; Kv = 'f16'; Rebuild = 'default' },
    @{ Tag = 'gplr50-b1-q8-default';    Bin = 'changed'; Shape = 'b1';  Kv = 'q8';  Rebuild = 'default' },
    @{ Tag = 'gplr50-ctl-4k-f16';       Bin = 'control'; Shape = '4k';  Kv = 'f16'; Rebuild = 'default' },
    @{ Tag = 'gplr50-ctl-4k-q8';        Bin = 'control'; Shape = '4k';  Kv = 'q8';  Rebuild = 'default' }
)

foreach ($armSpec in $arms) {
    if ($Only.Count -gt 0 -and $Only -notcontains $armSpec.Tag) { continue }
    Write-Host ("==== ARM {0} ====" -f $armSpec.Tag)
    & $arm -Tag $armSpec.Tag -Bin $armSpec.Bin -Shape $armSpec.Shape -Kv $armSpec.Kv -Rebuild $armSpec.Rebuild
    if ($LASTEXITCODE -ne 0) { Write-Host ("ARM {0} FAILED (exit {1})" -f $armSpec.Tag, $LASTEXITCODE) }
    else { Write-Host ("ARM {0} PASS" -f $armSpec.Tag) }
}
Write-Host 'SET COMPLETE'
