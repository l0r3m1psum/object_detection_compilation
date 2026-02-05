@echo off

REM TODO: provide a way to restore the previous PATH
set errorlevel=0
call .\config.bat
if not exist .\config.bat set errorlevel=1
if %errorlevel% neq 0  (
	echo Please create 'config.bat' file that sets the installdir environment variable. 1>&2
	echo Example: 1>&2
	echo set "installdir=D:" 1>&2
	exit /b 1
)

FOR /F "tokens=*" %%a in ('where git') do (
	set "gitpath=%%~dpa"
	goto :break
)
:break
if %errorlevel% neq 0 (
	echo Could not find git.
	exit /b 1
)

REM We set the path to the Windows' default to avoid "interferences" from outside programs
set "PATH=%SystemRoot%\system32;%SystemRoot%;%SystemRoot%\System32\Wbem;%SYSTEMROOT%\System32\WindowsPowerShell\v1.0\"

set "PATH=%gitpath%;%installdir%\Programs\TVM\lib;%installdir%\Programs\LLVM\bin;%installdir%\Programs\Python;%installdir%\Appdata\Roaming\Python\Python311\Scripts;%PATH%"

FOR /F "tokens=*" %%a in ('"%PROGRAMFILES(X86)%\Microsoft Visual Studio\Installer\vswhere.exe" -version [17.0^,18.0^) -latest -property installationPath') do SET vspath=%%a
if %errorlevel% neq 0 (
	echo Can't find vswhere, probably Visual Studio is not installed. 1>&2
	exit /b 1
)
if not defined vspath (
    echo Can't find Visual Studio 2022 (Version 17^). 1>&2
    exit /b 1
)
call "%vspath%\VC\Auxiliary\Build\vcvarsall.bat" x64 || exit /b 1

set "PYTHONUSERBASE=%installdir%\AppData\Roaming\Python"

REM Needed by vtar.Environment
REM The bitsream should be inside "zcu104\0_0_1\1x16_i8w8a32_15_15_18_17.bit"
set "VTA_HW_PATH=%cd%\submodules\tvm-vta"
REM Needed by vtar.bitstream.get_bitstream_path
set "VTA_CACHE_PATH=%installdir%\Programs\bitstreams"
set "HOME=workaround_for_get_bitstream_path"

where /q python
if %ERRORLEVEL% equ 0 (
	python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt || exit /b 1
	REM Optional dependencies
	python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq2.txt
	python -m site
	python -m sysconfig
)