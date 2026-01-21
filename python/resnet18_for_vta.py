import torch
from torchvision.models import resnet18, ResNet18_Weights
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod, quant_pre_process
import numpy
import onnxruntime as ort
from torchvision import datasets, transforms
import os
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import vtar.relax.frontend.onnx
import vtar.relax.transform
import tvm
from tvm import relax
import onnx

# From the ONNX model zoo there are already quantized models available they all
# use per channel scale and zero_point. YOLOv3 has the pesky quantized leaky
# ReLU. YOLO cant be supported because relax.op.all_class_non_max_suppression is
# not available in TVM 0.20.0 and the ONNX Loop operator is also not supported.
# https://github.com/apache/tvm/issues/17767
# LocalResponseNormalization (LRN) is needed by inception and it is available in
# future versions of TVM
# Supportable
# https://huggingface.co/onnxmodelzoo
# https://huggingface.co/onnxmodelzoo/mnist-12-int8/resolve/main/mnist-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/vgg16-12-int8/resolve/main/vgg16-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/resnet50-v1-12-int8/resolve/main/resnet50-v1-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/densenet-12-int8/resolve/main/densenet-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/squeezenet1.0-12-int8/resolve/main/squeezenet1.0-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/shufflenet-v2-12-int8/resolve/main/shufflenet-v2-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/mobilenetv2-12-int8/resolve/main/mobilenetv2-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/efficientnet-lite4-11-int8/resolve/main/efficientnet-lite4-11-int8.onnx
# Supported in future version of TVM
# https://huggingface.co/onnxmodelzoo/caffenet-12-int8/resolve/main/caffenet-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/googlenet-12-int8/resolve/main/googlenet-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/zfnet512-12-int8/resolve/main/zfnet512-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/inception-v1-12-int8/resolve/main/inception-v1-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/ssd-12-int8/resolve/main/ssd-12-int8.onnx
# Can it run???
# https://huggingface.co/onnxmodelzoo/shufflenet-v2-12-int8/resolve/main/shufflenet-v2-12-int8.onnx
# Has Loop
# https://huggingface.co/onnxmodelzoo/yolov3-12-int8/resolve/main/yolov3-12-int8.onnx
# https://huggingface.co/onnxmodelzoo/ssd_mobilenet_v1_12-int8/resolve/main/ssd_mobilenet_v1_12-int8.onnx

class MyCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_loader):
        super().__init__()
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            batch = next(self.data_iter)
            return {'input': batch[0].numpy()}
        except StopIteration:
            return None

def run_inference(session, input_tensor):
    ort_inputs = {session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = session.run(None, ort_inputs)
    return numpy.array(ort_outs[0])

def main():

    # https://stackoverflow.com/q/58151507
    imagenet_rgb_mean = (0.485, 0.456, 0.406)
    imagenet_rgb_std = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_rgb_mean, std=imagenet_rgb_std),
    ])

    if not os.path.exists("build/resnet18.onnx"):
        try:
            train_dataset = datasets.Imagenette(root='build/dataset',  split='train', download=True, transform=transform)
        except RuntimeError:
            train_dataset = datasets.Imagenette(root='build/dataset',  split='train', download=False, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.train()
        model.to(device)

        num_epochs = 5

        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

        # torch.save(model.state_dict(), "build/resnet18_imagenette.pth")

        model.eval()
        model.to('cpu')
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model, dummy_input, "build/resnet18.onnx",
                        input_names=["input"], output_names=["output"],
                        opset_version=11)

    if not os.path.exists("build/resnet18_int8.onnx"):
        calib_dataset = datasets.Imagenette(root='build/dataset', split='train', download=False, transform=transform)
        calib_data_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)
        calib_data_reader = MyCalibrationDataReader(calib_data_loader)

        # TODO: make arguments of this function explicit (and understand them)
        quant_pre_process("build/resnet18.onnx", "build/resnet18_pre_proc.onnx")
        quantize_static(
            model_input="build/resnet18_pre_proc.onnx",
            model_output="build/resnet18_int8.onnx",
            calibration_data_reader=calib_data_reader,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=None,
            per_channel=False,
            reduce_range=False,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            nodes_to_quantize=None,
            nodes_to_exclude=None,
            use_external_data_format=False,
            calibrate_method=CalibrationMethod.MinMax,
            calibration_providers=None,
            extra_options={
                "ActivationSymmetric": True,
                "WeightSymmetric": True,
                "QDQKeepRemovableActivations": False,
            },

        )

    import ctypes
    vta_fsim = ctypes.CDLL("vta_fsim")
    env = vtar.get_env()
    target = tvm.target.Target(env.target, host=env.target_host)
    dev = tvm.device(str(env.target))
    # dev = tvm.runtime.device('cpu')
    if not os.path.exists("build/resnet18_int8.dll"):
        onnx_model = onnx.load("build/resnet18_int8.onnx")
        mod = vtar.relax.frontend.onnx.from_onnx(onnx_model)
        mod = vtar.relax.transform.RemoveUnnecessaryDequantizeQuantizeWrapping()(mod)
        mod = vtar.relax.transform.GraphPack()(mod)
        mod = relax.get_pipeline('vtar_zero')(mod)
        target = tvm.target.Target.from_device(dev)
        ex = relax.build(mod, target)
        ex.export_library("build/resnet18_int8.dll")
    else:
        ex = tvm.runtime.load_module("build/resnet18_int8.dll")

    vm = relax.VirtualMachine(ex, dev)

    test_dataset = datasets.Imagenette(root='build/dataset', split='val', download=False, transform=transform)
    test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset))[:200])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    sess_fp32 = ort.InferenceSession("build/resnet18.onnx")
    sess_int8 = ort.InferenceSession("build/resnet18_int8.onnx")

    correct_fp32 = 0
    correct_int8 = 0
    correct_tvm = 0
    total = 0

    for images, labels in test_loader:
        images = images.cpu()

        outputs_fp32 = run_inference(sess_fp32, images)
        pred_fp32 = numpy.argmax(outputs_fp32, axis=1)

        outputs_int8 = run_inference(sess_int8, images)
        pred_int8 = numpy.argmax(outputs_int8, axis=1)

        outputs_tvm = vm["main"](tvm.nd.array(images.numpy(), dev)).numpy()
        pred_tvm = numpy.argmax(outputs_tvm, axis=1)

        correct_fp32 += (pred_fp32 == labels.numpy()).sum()
        correct_int8 += (pred_int8 == labels.numpy()).sum()
        correct_tvm += (pred_tvm == labels.numpy()).sum()
        total += labels.size(0)
        print(".", end="")
        sys.stdout.flush()
    print("")

    print(f"Accuracy FP32: {correct_fp32/total*100:.2f}%")
    print(f"Accuracy INT8: {correct_int8/total*100:.2f}%")
    print(f"Accuracy tvm: {correct_tvm/total*100:.2f}%")

if __name__ == "__main__":
    main()
