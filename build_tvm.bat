@echo off
REM https://tvm.apache.org/docs/install/from_source.html

REM We expect to be in a Python's venv virutal environment and that
REM vcvarsall.bat has been invoked to correctly setup the build environment.

setlocal

call config.bat

REM pip install -r requirements_tvm.txt || goto :exit
pushd submodules\tvm || goto :exit
    REM git submodule update --init --recursive || goto :exit
    if not exist build (mkdir build || goto :exit)
    pushd build || goto :exit
        copy /y ..\cmake\config.cmake . || goto :exit
        echo set(CMAKE_BUILD_TYPE RelWithDebInfo) >> config.cmake || goto :exit
        echo set(USE_LLVM "llvm-config --ignore-libllvm --link-static") >> config.cmake || goto :exit
        echo set(HIDE_PRIVATE_SYMBOLS ON) >> config.cmake || goto :exit
        echo set(USE_CUDA ON) >> config.cmake || goto :exit
        echo set(USE_CUBLAS ON) >> config.cmake || goto :exit

        REM echo set(SUMMARIZE ON) >> config.cmake || goto :exit
        REM echo set(CMAKE_C_COMPILER "%installdir:\=\\%\\Programs\\LLVM\\bin\\clang-cl.exe") >> config.cmake || goto :exit
        REM echo set(CMAKE_CXX_COMPILER "%installdir:\=\\%\\Programs\\LLVM\\bin\\clang-cl.exe") >> config.cmake || goto :exit

        cmake "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\TVM" .. || goto :exit
        REM Needed because otherwise LINK.EXE cannot find tvm.lib
        set "LINK=/LIBPATH:%CD%\Debug /LIBPATH:%installdir%\Programs\Python\libs" || goto :exit
        cmake --build . --verbose --target install --parallel %ncores% || goto :exit
        set "TVM_LIBRARY_PATH=%CD%\Debug" || goto :exit
    popd || goto :exit
    set "LINK=/LIBPATH:%installdir%\Programs\Python\libs /LIBPATH:%installdir%\Programs\TVM\lib" || goto :exit
    REM --no-build-isolation makes it faster
    pip install --verbose --no-build-isolation --no-index --find-links "%installdir%\Programs\wheelhouse" .\python || goto :exit
    REM python -c "import tvm; print(tvm.__file__); print(tvm._ffi.base._LIB); print('\n'.join(f'{k}: {v}' for k, v in tvm.support.libinfo().items()))" || goto :exit

    if not exist 3rdparty\dlpack\build (mkdir 3rdparty\dlpack\build || goto :exit)
    pushd 3rdparty\dlpack\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dlpack" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd || goto :exit

    if not exist 3rdparty\dmlc-core\build (mkdir 3rdparty\dmlc-core\build || goto :exit)
    pushd 3rdparty\dmlc-core\build || goto :exit
        cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ^
            -DBUILD_MOCK=no ^
            "-DCMAKE_INSTALL_PREFIX=%installdir%\Programs\dmlc" ^
            -DCMAKE_BUILD_TYPE=RelWithDebInfo .. || goto :exit
        cmake --build . --target install --parallel %ncores% || goto :exit
    popd || goto :exit
popd || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%
