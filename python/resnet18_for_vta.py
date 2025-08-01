import torch
from torchvision.models import resnet18
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader
import torch.nn as nn


class CalibrationDataReader:
    def __init__(self, data_loader):
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            batch = next(self.data_iter)
            return {'input': batch[0].numpy()}
        except StopIteration:
            return None

    


# Funzione per inferenza con onnxruntime
def run_inference(session, input_tensor):
    ort_inputs = {session.get_inputs()[0].name: input_tensor.numpy()}
    ort_outs = session.run(None, ort_inputs)
    return np.array(ort_outs[0])
    



if not (os.path.exists("resnet18_int8.onnx")):

    # Setup dataset (es. ImageNet valida anche per ResNet18)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    # calibrazione per la quantizzazione su CIFAR100
    train_dataset = datasets.Imagenette(root='./dataset',  split='train', download=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # carico il modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))  # Adatta alle 10 classi di Imagenette

    # preparo il train
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    model.to(device)

    num_epochs = 5  # <-- Qui specifichi il numero di epoche

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

    # Salva modello PyTorch
    torch.save(model.state_dict(), "resnet18_imagenette.pth")

    # valutazione del modello
    model.eval()
    model.to('cpu')
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, "resnet18.onnx",
                    input_names=["input"], output_names=["output"],
                    opset_version=11)


    # calibrazione del dataset per la quantizzazione statica
    calib_dataset = datasets.Imagenette(root='./dataset', split='val', download=False, transform=transform)
    calib_data_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True)
    calib_data_reader = CalibrationDataReader(calib_data_loader)

    # Quantizza il modello
    quantize_static(
        model_input="resnet18.onnx",
        model_output="resnet18_int8.onnx",
        calibration_data_reader=calib_data_reader,
        quant_format=QuantFormat.QOperator,  # più compatibile con TVM perchè utilizza quantizzazioni native ONNX
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )


    test_dataset = datasets.Imagenette(root='./dataset', split='val', download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)


    # Carica modello FP32 e INT8
    sess_fp32 = ort.InferenceSession("resnet18.onnx")
    sess_int8 = ort.InferenceSession("resnet18_int8.onnx")

    # Valutazione
    correct_fp32 = 0
    correct_int8 = 0
    total = 0

    for images, labels in test_loader:
        images = images.cpu()

        # inference FP32
        outputs_fp32 = run_inference(sess_fp32, images)
        pred_fp32 = np.argmax(outputs_fp32, axis=1)

        # inference INT8
        outputs_int8 = run_inference(sess_int8, images)
        pred_int8 = np.argmax(outputs_int8, axis=1)

        correct_fp32 += (pred_fp32 == labels.numpy()).sum()
        correct_int8 += (pred_int8 == labels.numpy()).sum()
        total += labels.size(0)

    print(f"Accuracy FP32: {correct_fp32/total*100:.2f}%")
    print(f"Accuracy INT8: {correct_int8/total*100:.2f}%")



















"""
import tvm
from tvm import relay
import onnx

onnx_model = onnx.load("resnet18_int8.onnx")

shape_dict = {"input": (1, 3, 224, 224)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

# Compila per VTA
from vta import environment
env = environment.get_env()

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=env.target, target_host=env.target_host, params=params)
"""
