@echo off
REM Build all dependencies for the project

setlocal

call config.bat

pushd submodules

    if not exist tvm\3rdparty\dlpack\build (mkdir tvm\3rdparty\dlpack\build || goto :exit)
    pushd tvm\3rdparty\dlpack\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dlpack" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd || goto :exit

    if not exist tvm\3rdparty\dmlc-core\build (mkdir tvm\3rdparty\dmlc-core\build || goto :exit)
    pushd tvm\3rdparty\dmlc-core\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dmlc" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd || goto :exit

    if not exist safetensors-cpp\build (mkdir safetensors-cpp\build || goto :exit)
    pushd safetensors-cpp\build
        cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\safetensors" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
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
            --copy "%installdir%\Programs\Python" || goto :exit
    popd || goto :exit

    set "PATH=%installdir%\Programs\Python;%PATH%"

    pushd llvm-project
        if not exist llvm\build (mkdir llvm\build || goto :exit)
        pushd llvm\build || goto :exit
            cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\LLVM" ^
                -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
            cmake --build . --target install --parallel %ncores% || goto :exit
        popd

        if not exist clang\build (mkdir clang\build || goto :exit)
        pushd clang\build || goto :exit
            cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\LLVM" ^
                -DCMAKE_BUILD_TYPE=RelWithDebInfo ^
                -DLLVM_INCLUDE_TESTS=OFF .. || goto :exit
            cmake --build . --target install --parallel %ncores% || goto :exit
            if not exist "%installdir%\Programs\LLVM\bin\gcc.exe" (
                mklink /h "%installdir%\Programs\LLVM\bin\gcc.exe" ^
                    "%installdir%\Programs\LLVM\bin\clang.exe"
            )
        popd
    popd

popd

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%