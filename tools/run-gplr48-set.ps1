# T-GPLR-48 driver: A1-A4 of the contract as ONE session, so the pairs that are
# compared against each other (explicit-vs-explicit, control-vs-changed) sit in
# one thermal/session window -- the recorded ~3% session spread makes
# cross-session timing comparisons worthless.
#
# Contract mapping (validation set, plan line ~3771):
#   A1  f16 50k repeat 2, NO explicit flag        AC1, AC2, AC5
#   A2  q8 B1 repeat 2,  NO explicit flag         AC3, AC5
#   A3  f16/q8 x explicit 0/1 at 50k, each paired
#       against the SAME command on the control    AC4 (AC6 for the f16 ON pair)
#   A4  f16 4k 96, NO explicit flag, decode-graded AC2 (4k form: the echo shape)
#
# Ordering is deliberate:
#   1. the f16 explicit-0 PAIR first, because its CHANGED arm is A1's noise-floor
#      reference (same binary, same session) -- better than the recorded
#      gplr46-50k-off, which ran on an older binary;
#   2. then A1 itself, whose real evidence is route parity flipping DIVERGED ->
#      clean (a telemetry key alone would not prove the default took effect);
#   3. the remaining explicit pairs;
#   4. the 4k decode arm;
#   5. the 143k q8 B1 arm LAST: it is the long one (~5 min) and the only arm whose
#      pass-2 must survive the crash the rebuild exists to fix.
#
# Run this DETACHED; completion is signalled by the eleven gplr48-*.done
# sentinels (see docs/AGENT_ENVIRONMENT.md, "Detached runs and waiting").
param(
    # 'all' | 'defaults' (A1,A2,A4 only) | 'explicit' (A3 only) -- staged so the
    # cheap gates can be read before the long B1 arm is paid for.
    [ValidateSet('all', 'defaults', 'explicit')] [string] $Set = 'all'
)
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base = Join-Path $root 'build\llama-cuda'
$arm  = Join-Path $root 'tools\run-gplr48-arm.ps1'
$verdictDir = Join-Path $base 'verdicts'
if (-not (Test-Path -LiteralPath $arm)) { Write-Host "MISSING: $arm"; exit 1 }
. (Join-Path $root 'tools\arm-verdict.ps1')

# Tag, Shape, Kv, Rebuild, Bin, CompareTag, RefTag
$A3 = @(
    @{ Tag = 'gplr48-a3-50k-f16-pbp0-control'; Shape = '50k'; Kv = 'f16'; Rebuild = '0'; Bin = 'control'; Compare = '' }
    @{ Tag = 'gplr48-a3-50k-f16-pbp0-changed'; Shape = '50k'; Kv = 'f16'; Rebuild = '0'; Bin = 'changed'; Compare = 'gplr48-a3-50k-f16-pbp0-control' }
    @{ Tag = 'gplr48-a3-50k-f16-pbp1-control'; Shape = '50k'; Kv = 'f16'; Rebuild = '1'; Bin = 'control'; Compare = '' }
    @{ Tag = 'gplr48-a3-50k-f16-pbp1-changed'; Shape = '50k'; Kv = 'f16'; Rebuild = '1'; Bin = 'changed'; Compare = 'gplr48-a3-50k-f16-pbp1-control' }
    @{ Tag = 'gplr48-a3-50k-q8-pbp0-control';  Shape = '50k'; Kv = 'q8';  Rebuild = '0'; Bin = 'control'; Compare = '' }
    @{ Tag = 'gplr48-a3-50k-q8-pbp0-changed';  Shape = '50k'; Kv = 'q8';  Rebuild = '0'; Bin = 'changed'; Compare = 'gplr48-a3-50k-q8-pbp0-control' }
    @{ Tag = 'gplr48-a3-50k-q8-pbp1-control';  Shape = '50k'; Kv = 'q8';  Rebuild = '1'; Bin = 'control'; Compare = '' }
    @{ Tag = 'gplr48-a3-50k-q8-pbp1-changed';  Shape = '50k'; Kv = 'q8';  Rebuild = '1'; Bin = 'changed'; Compare = 'gplr48-a3-50k-q8-pbp1-control' }
)
$A1 = @{ Tag = 'gplr48-a1-50k-f16-default'; Shape = '50k'; Kv = 'f16'; Rebuild = 'default'; Bin = 'changed'; Compare = ''; Ref = 'gplr48-a3-50k-f16-pbp0-changed' }
$A4 = @{ Tag = 'gplr48-a4-4k-f16-default';  Shape = '4k';  Kv = 'f16'; Rebuild = 'default'; Bin = 'changed'; Compare = '' }
$A2 = @{ Tag = 'gplr48-a2-b1-q8-default';   Shape = 'b1';  Kv = 'q8';  Rebuild = 'default'; Bin = 'changed'; Compare = '' }

$plan = switch ($Set) {
    'explicit' { $A3 + $A1 }
    'defaults' { $A1 + $A4 + $A2 }
    default    { @($A3[0], $A3[1], $A1) + $A3[2..7] + @($A4, $A2) }
}

Write-Host ("[gplr48] set={0} arms={1}" -f $Set, $plan.Count)
$failed = @()
foreach ($s in $plan) {
    $armArgs = @('-Tag', $s.Tag, '-Shape', $s.Shape, '-Kv', $s.Kv, '-Rebuild', $s.Rebuild, '-Bin', $s.Bin)
    if ($s.Compare) { $armArgs += @('-CompareTag', $s.Compare) }
    if ($s.Ref)     { $armArgs += @('-RefTag', $s.Ref) }
    Write-Host ("[gplr48] ===== {0} (shape={1} kv={2} rebuild={3} bin={4}) =====" -f $s.Tag, $s.Shape, $s.Kv, $s.Rebuild, $s.Bin)
    & powershell -NoProfile -ExecutionPolicy Bypass -File $arm @armArgs
    $code = $LASTEXITCODE
    Write-Host ("[gplr48] arm {0} exit: {1}" -f $s.Tag, $code)
    if ($code -ne 0) { $failed += $s.Tag }
}

# ---- summary: the resolved-default table plus the parity flip ----
$rows = @()
foreach ($s in $plan) {
    $v = Read-ArmVerdict -Dir $verdictDir -Tag $s.Tag
    if ($null -eq $v) { $rows += [pscustomobject]@{ arm = $s.Tag; status = 'NO-VERDICT' }; continue }
    $rows += [pscustomobject]@{
        arm        = $s.Tag
        status     = $v.status
        kv         = $v.fields.kv_cache
        requested  = $v.fields.rebuild_requested
        resolved   = $v.fields.resolved_pp_prebuild_per_pass
        source     = $v.fields.pp_prebuild_per_pass_source
        announced  = $v.fields.announcement_present
        parity     = $v.fields.route_parity
        parity_exp = $v.fields.parity_expected
        excess_2v1 = $v.fields.excess_2v1_s
        tg_ids     = $v.fields.tg_ids
        identity   = $v.fields.compare_identity_ok
        timing_3pct= $v.fields.compare_timing_within_3pct
        pass_count = $v.fields.pass_count
        exit       = $v.exit_code
    }
}
Write-Host ''
Write-Host '==== T-GPLR-48 summary ===='
$rows | Format-Table -AutoSize | Out-String -Width 400 | Write-Host
$summary = Join-Path $base 'gplr48-summary.json'
[System.IO.File]::WriteAllText($summary,
    (([ordered]@{ set = $Set; failed = $failed; rows = $rows } | ConvertTo-Json -Depth 6)),
    (New-Object System.Text.UTF8Encoding($false)))
Write-Host ("[gplr48] summary -> {0}" -f $summary)
if ($failed.Count -gt 0) {
    Write-Host ("[gplr48] FAILED arms: {0}" -f ($failed -join ', '))
    exit 1
}
Write-Host '[gplr48] all planned arms PASSED'
exit 0
