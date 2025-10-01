@echo off
REM Build all dependencies for the project

setlocal

set target=install

pushd submodules

    if not exist tvm\3rdparty\dlpack\build (mkdir tvm\3rdparty\dlpack\build || goto :exit)
    pushd tvm\3rdparty\dlpack\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dlpack" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% || goto :exit
    popd || goto :exit

    if not exist tvm\3rdparty\dmlc-core\build (mkdir tvm\3rdparty\dmlc-core\build || goto :exit)
    pushd tvm\3rdparty\dmlc-core\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dmlc" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% || goto :exit
    popd || goto :exit

    if not exist safetensors-cpp\build (mkdir safetensors-cpp\build || goto :exit)
    pushd safetensors-cpp\build
        cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\safetensors" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% || goto :exit
    popd

    pushd cpython || goto :exit
        pushd PCbuild || goto :exit
            REM TODO: This should work with the -E flag i.e. we should include
            REM all python's essential dependencies as submodules. To do so add
            REM flags:
            REM     --no-ctypes --no-ssl --no-tkinter
            REM and download
            REM     bzip2 sqlite xz zlib
            REM as get_externals.bat does.
            call build.bat || goto :exit
            REM TODO: call clean.bat when %target% is clean
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

    pushd llvm-project
        if not exist llvm\build (mkdir llvm\build || goto :exit)
        pushd llvm\build || goto :exit
            cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\LLVM" ^
                -DLLVM_ENABLE_PROJECTS=clang ^
                -DLLVM_INCLUDE_TESTS=OFF ^
                -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
            cmake --build . --target %target% --parallel %NUMBER_OF_PROCESSORS% || goto :exit
            if not exist "%installdir%\Programs\LLVM\bin\gcc.exe" (
                mklink /h "%installdir%\Programs\LLVM\bin\gcc.exe" ^
                    "%installdir%\Programs\LLVM\bin\clang.exe" || goto :exit
            )
        popd
    popd

popd

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%