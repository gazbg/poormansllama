# Build the T-GPLR-40 stock-libllama probe with MSVC against the built llama lib.
$ErrorActionPreference = 'Stop'
$root      = 'C:/Development/AI/qwen38GZI'
$base      = Join-Path $root 'build/llama-cuda'
$vcvars    = 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
$incLlama  = Join-Path $root 'third_party/llama.cpp/include'
$incGgml   = Join-Path $root 'third_party/llama.cpp/ggml/include'
$libDir    = Join-Path $base 'bin/Release'
$src       = Join-Path $root 'tools/gplr40-stock-probe.cpp'
$exe       = Join-Path $base 'gplr40-stock-probe.exe'

$bat = Join-Path $env:TEMP 'gplr40-build.bat'
@"
call "$vcvars" >nul
cl /nologo /O2 /EHsc /I"$incLlama" /I"$incGgml" "$src" /link /LIBPATH:"$libDir" llama.lib /OUT:"$exe"
"@ | Set-Content $bat -Encoding Ascii

cmd /c $bat 2>&1 | Select-Object -Last 6 | ForEach-Object { Write-Host $_ }
if (Test-Path $exe) { Write-Host "BUILD OK: $exe"; exit 0 } else { Write-Host 'BUILD FAILED'; exit 1 }
