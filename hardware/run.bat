@echo off

setlocal
set "VTA_RPC_HOST=192.168.2.99"
set "VTA_RPC_PORT=9091"
set "PYTHONPATH=%CD%\..\submodules\tvm\vta\python"
REM The bitsream should be inside "zcu104\0_0_1\1x16_i8w8a32_15_15_18_17.bit"
set "VTA_CACHE_PATH=%installdir%\Programs\bitstreams"
REM The code expect the HOME environment variable to exists.
set "HOME=workaround"

pushd ..\submodules\tvm
	copy /y 3rdparty\vta-hw\config\zcu104_sample.json ^
		3rdparty\vta-hw\config\vta_config.json >NUL
	REM python vta\tests\python\integration\test_benchmark_topi_conv2d.py
	REM copy /y 3rdparty\vta-hw\config\fsim_sample.json ^
	REM 	3rdparty\vta-hw\config\vta_config.json >NUL
popd

pushd ..
	python -i python\test_relay.py
popd

endlocal