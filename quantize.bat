@echo off

setlocal

set "imagenet_train=%installdir%\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train"
set "imagenet_val=%installdir%\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\val"
set "model=build\gluon_resnet18_v1b_Opset18.onnx"

rem python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 10000 -t axis -o build\resnet18-v2-7_int8_axis.onnx || goto :exit
rem python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 10000 -t tensor -o build\resnet18-v2-7_int8_tensor.onnx || goto :exit
REM python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 10000 -t network -o build\resnet18-v2-7_int8_network.onnx || goto :exit

rem python python\compile_model.py -m build\resnet18-v2-7_int8_axis.onnx -f requantize -o build\resnet18-v2-7_int8_axis.dll || goto :exit
rem python python\compile_model.py -m build\resnet18-v2-7_int8_tensor.onnx -f requantize -o build\resnet18-v2-7_int8_tensor.dll || goto :exit
REM python python\compile_model.py -m build\resnet18-v2-7_int8_tensor.onnx -f requantize -o build\resnet18-v2-7_int8_tensor.dll || goto :exit

python python\eval_models.py -m ^
	"%model%" ^
	build\resnet18-v2-7_int8_axis.onnx ^
	build\resnet18-v2-7_int8_axis.dll ^
	build\resnet18-v2-7_int8_tensor.onnx ^
	build\resnet18-v2-7_int8_tensor.dll ^
	-s 50000 ^
	-d "%imagenet_val%" || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%
