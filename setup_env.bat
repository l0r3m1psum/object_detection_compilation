@echo off

REM TODO: provide a way to restore the previous PATH
set errorlevel=0
call .\config.bat
if %errorlevel% neq 0 (
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

set "PATH=%gitpath%;%installdir%\Programs\LLVM\bin;%installdir%\Programs\Python;%installdir%\Appdata\Roaming\Python\Python311\Scripts;%PATH%"

FOR /F "tokens=*" %%a in ('"%PROGRAMFILES(X86)%\Microsoft Visual Studio\Installer\vswhere.exe" -property installationPath') do SET vspath=%%a
if %errorlevel% neq 0 (
	echo Can't find vswhere, probably Visual Studio is not installed. 1>&2
	exit /b 1
)
call "%vspath%\VC\Auxiliary\Build\vcvarsall.bat" x64 || exit /b 1

set "PYTHONUSERBASE=%installdir%\AppData\Roaming\Python"

where /q python
if %ERRORLEVEL% equ 0 (
	python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" -r projreq.txt || exit /b 1
	python -m pip install --no-index --find-links "%installdir%\Programs\wheelhouse" torch torchvision || exit /b 1
	python -m site
	python -m sysconfig
)