@echo off
REM Build all dependencies for the project

setlocal

REM TODO: take them as parameters from the command line
set installdir=D:
set ncores=24

pushd llvm-project
    if not exist llvm\build (mkdir llvm\build || goto :exit)
    pushd llvm\build || goto :exit
        cmake -DCMAKE_INSTALL_PREFIX=%installdir%\Programs\LLVM ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd

    if not exist clang\build (mkdir clang\build || goto :exit)
    pushd clang\build || goto :exit
        cmake -DCMAKE_INSTALL_PREFIX=%installdir%\Programs\LLVM ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
            -DLLVM_INCLUDE_TESTS=OFF .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd
popd

pushd cpython || goto :exit
    pushd PCbuild || goto :exit
        call build.bat || goto :exit
    popd || goto :exit

    call python.bat PC\layout ^
        --include-stable ^
        --include-pip ^
        --include-pip-user ^
        --include-distutils ^
        --include-venv ^
        --include-dev ^
        --copy %installdir%\Programs\Python || goto :exit
popd || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%