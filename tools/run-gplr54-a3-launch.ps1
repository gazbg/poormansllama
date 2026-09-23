# Detached launcher for the T-GPLR-54 A3 discriminator session (owner-invoked).
#
# Conventions enforced here (AGENTS.md, measured 2026-09-18):
#   * Clear-ArmVerdict for the tag BEFORE Start-Process, so Wait-ArmStart can
#     never read a previous run's marker or verdict.
#   * Plain Start-Process (no setsid/nohup shapes).
#   * Launch confirmation = the .started marker, not the process id; polling =
#     Wait-ArmVerdict / the verdict JSON, never log tailing.
#
# The inner command runs through the wrapper's `cmd.exe /d /s /c`, so it must be
# a cmd-safe string: the model/repo paths on this machine have no spaces, so no
# inner quotes are needed at all.
$ErrorActionPreference = 'Continue'
$root = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
. (Join-Path $root 'tools\arm-verdict.ps1')

$vDir = Join-Path $root 'build\llama-cuda\verdicts'
$tag  = 'gplr54-a3'
$log  = Join-Path $root 'build\llama-cuda\gplr54-a3-session.log'
$err  = Join-Path $root 'build\llama-cuda\gplr54-a3-session.err'
$model = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'

if (-not (Test-Path $model)) { Write-Host "LAUNCHER: MISSING MODEL $model"; exit 1 }

Clear-ArmVerdict -Dir $vDir -Tag $tag   # invalidate FIRST, before the child exists

$session = Join-Path $root 'tools\run-gplr54-a3-session.ps1'
$wrapper = Join-Path $root 'tools\with-gpu-lock.ps1'
# Outer child = a -Command child: powershell re-joins -Command tokens with
# spaces before parsing, so Start-Process's unreliable quoting of spaced array
# elements cannot corrupt the parse (measured this session: a -File child leaked
# its inner -NoProfile into with-gpu-lock.ps1 as a parameter). The & call
# operator gives the wrapper real named-parameter binding, and the inner command
# binds to -Command as one single-quoted token. NO double quotes anywhere in
# $cmd, so no layer can tear it.
$inner   = 'powershell -NoProfile -ExecutionPolicy Bypass -File ' + $session
$cmd     = "& '$wrapper' -TimeoutSeconds 7200 -VerdictTag $tag -Command '$inner'"
$p = Start-Process -FilePath 'powershell' -WindowStyle Hidden -PassThru `
       -WorkingDirectory $root `
       -RedirectStandardOutput $log -RedirectStandardError $err `
       -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $cmd)
Write-Host ("LAUNCHER: wrapper child pid {0}" -f $p.Id)

$m = Wait-ArmStart -Dir $vDir -Tag $tag -TimeoutSeconds 30 -Quiet
if ($null -eq $m) {
    Write-Host 'LAUNCHER: start marker NOT seen in 30s - the wrapper died before marking (see the log/err files).'
    exit 1
}
Write-Host ("LAUNCHER: session started {0} (marker pid {1})" -f $m.started_utc, $m.pid)
Write-Host ("LAUNCHER: verdict will land at {0}" -f (Join-Path $vDir ($tag + '.verdict.json')))
exit 0
