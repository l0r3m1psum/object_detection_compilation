@echo off

setlocal

set "imagenet_train=%installdir%\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\train"
set "imagenet_val=%installdir%\imagenet-object-localization-challenge\ILSVRC\Data\CLS-LOC\val"
set "model=build\gluon_resnet18_v1b_Opset18.onnx"

python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 1000 -t axis -o build\resnet18-v2-7_int8_axis.onnx || goto :exit
python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 1000 -t tensor -o build\resnet18-v2-7_int8_tensor.onnx || goto :exit
python python\quantize_model.py -m "%model%" -d "%imagenet_train%" -s 1000 -t network -o build\resnet18-v2-7_int8_network.onnx || goto :exit
python python\eval_models.py -m ^
	"%model%" ^
	build\resnet18-v2-7_int8_axis.onnx ^
	build\resnet18-v2-7_int8_tensor.onnx ^
	build\resnet18-v2-7_int8_network.onnx ^
	-d "%imagenet_val%" -s 1000 || goto :exit

endlocal

:exit
if %ERRORLEVEL% neq 0 echo An error occurred!
exit /b %ERRORLEVEL%
