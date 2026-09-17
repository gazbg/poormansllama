# T-GPLR-40 addendum: build the stock-libllama ownership probe and run its two
# arms (q8 = expected crash if upstream; f16 = control) under the GPU mutex.
# Failure isolation only -- no performance claims.
$ErrorActionPreference = 'Continue'
$root   = Split-Path (Split-Path $PSCommandPath -Parent) -Parent
$base   = Join-Path $root 'build\llama-cuda'
$model  = 'C:/Models/ALL/other/qwen-3.6/Qwen3.6-35B-A3B-MTP-UD-Q4_K_S.gguf'
$prompt = Join-Path $base 'gplr07-qa-prompts/anchor-100000.txt'
$src    = Join-Path $root 'tools\gplr40-stock-probe.cpp'
$exe    = Join-Path $base 'gplr40-stock-probe.exe'

if (-not (Test-Path $prompt)) { Write-Host "MISSING ANCHOR: $prompt"; exit 1 }

# --- build (no CUDA in the probe itself; links the already-built llama.lib). ---
# VS 18 BuildTools is the toolset that produced llama.lib; the 2022 install's
# vcvars is broken here (its vswhere dependency fails).
$vcvars = 'C:/Program Files (x86)/Microsoft Visual Studio/18/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
if (-not (Test-Path $vcvars)) { Write-Host "vcvars64.bat not found at $vcvars"; exit 1 }
$incLlama = Join-Path $root 'third_party\llama.cpp\include'
$incGgml  = Join-Path $root 'third_party\llama.cpp\ggml\include'
$libDir   = Join-Path $base 'src\Release'
$cmd = "`"$vcvars`" >nul && cl /nologo /O2 /EHsc /I`"$incLlama`" /I`"$incGgml`" `"$src`" /link /LIBPATH:`"$libDir`" llama.lib /OUT:`"$exe`""
Write-Host "[gplr40-stock] building..."
cmd /c $cmd 2>&1 | Select-Object -Last 4 | ForEach-Object { Write-Host $_ }
if (-not (Test-Path $exe)) { Write-Host 'BUILD FAILED'; exit 1 }

# --- run both arms under the shared-GPU mutex. ---
$lock  = Join-Path $root 'tools\with-gpu-lock.ps1'
$inner = Join-Path $root 'tools\run-gplr40-stock-locked.ps1'
& $lock -TimeoutSeconds 2400 -Command ("powershell -NoProfile -ExecutionPolicy Bypass -File `"" + $inner + "`" -Exe `"" + $exe + "`" -Model `"" + $model + "`" -Prompt `"" + $prompt + "`"")
Write-Host ("WRAPPER EXIT: {0}" -f $LASTEXITCODE)
