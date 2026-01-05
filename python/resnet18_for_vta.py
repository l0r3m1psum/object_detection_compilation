import torch
from torchvision.models import resnet18, ResNet18_Weights
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import vtar.relax.frontend.onnx
import vtar.relax.transform
import tvm
from tvm import relax
import onnx

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
    return np.array(ort_outs[0])

def main():

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
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

        quantize_static(
            model_input="build/resnet18.onnx",
            model_output="build/resnet18_int8.onnx",
            calibration_data_reader=calib_data_reader,
            quant_format=QuantFormat.QOperator,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8
        )

    dev = tvm.runtime.device('cpu')
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
        pred_fp32 = np.argmax(outputs_fp32, axis=1)

        outputs_int8 = run_inference(sess_int8, images)
        pred_int8 = np.argmax(outputs_int8, axis=1)

        outputs_tvm = vm["main"](tvm.nd.array(images.numpy(), dev)).numpy()
        pred_tvm = np.argmax(outputs_tvm, axis=1)

        correct_fp32 += (pred_fp32 == labels.numpy()).sum()
        correct_int8 += (pred_int8 == labels.numpy()).sum()
        correct_tvm += (pred_tvm == labels.numpy()).sum()
        total += labels.size(0)
        print(".", end="")
    print("")

    print(f"Accuracy FP32: {correct_fp32/total*100:.2f}%")
    print(f"Accuracy INT8: {correct_int8/total*100:.2f}%")
    print(f"Accuracy tvm: {correct_tvm/total*100:.2f}%")

if __name__ == "__main__":
    main()
