import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms

import onnx
from onnx import numpy_helper
import onnxruntime
from onnxruntime.quantization import (
    quantize_static, CalibrationDataReader, QuantType, QuantFormat,
    CalibrationMethod, quant_pre_process,
    QuantizationMode,
)
from onnxruntime.quantization.registry import QLinearOpsRegistry, QDQRegistry
from onnxruntime.quantization.quant_utils import (
    save_and_reload_model_with_shape_infer,
    load_model_with_shape_infer,
    model_has_pre_process_metadata,
    update_opset_version,
)
from onnxruntime.quantization.quantize import check_static_quant_arguments
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.calibrate import (
    MinMaxCalibrater, create_calibrator, CalibrationMethod, TensorsData
)

import numpy

import os
import sys
import pathlib
import tempfile
import collections
import copy
from typing import Container

import tvm
from tvm import relax
import vtar.relax.frontend.onnx

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

# Other model zoos
# https://huggingface.co/opencv
# https://github.com/STMicroelectronics/stm32ai-modelzoo/
# https://aihub.qualcomm.com/models/

# Quantized ResNet 18 (the Qualcomm one requires uint quantization)
# https://aihub.qualcomm.com/models/resnet18
# https://github.com/STMicroelectronics/stm32ai-modelzoo/blob/main/image_classification/resnet18_pt/Public_pretrainedmodel_public_dataset/Imagenet/resnet18wd4_pt_224/resnet18wd4_pt_224_qdq_int8.onnx
# This could be good for live real-time demo on face detection
# https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar_int8.onnx

def main():

    # https://stackoverflow.com/q/58151507
    imagenet_rgb_mean = (0.485, 0.456, 0.406)
    imagenet_rgb_std = (0.229, 0.224, 0.225)
    imagenet_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_rgb_mean, std=imagenet_rgb_std),
    ])

    if not os.path.exists("build/resnet18.onnx"):
        try:
            train_dataset = datasets.Imagenette(root='build/dataset',  split='train', download=True, transform=imagenet_transform)
        except RuntimeError:
            train_dataset = datasets.Imagenette(root='build/dataset',  split='train', download=False, transform=imagenet_transform)

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
                        opset_version=13)

if __name__ == "__main__":
    main()
