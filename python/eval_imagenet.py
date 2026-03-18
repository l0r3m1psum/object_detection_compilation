import os
import numpy as np
import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys

# ================= Configuration =================
ONNX_MODEL_PATH = "build/resnet18wd4_pt_224_qdq_int8.onnx"
# Path to your ImageNet validation directory (must contain subfolders for each class)
IMAGENET_VAL_DIR = "D:/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/val"
BATCH_SIZE = 64
NUM_WORKERS = 4
# =================================================

def main():
    imagenet_rgb_mean = (0.485, 0.456, 0.406)
    imagenet_rgb_std = (0.229, 0.224, 0.225)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_rgb_mean, imagenet_rgb_std),
    ])

    print(f"Loading ImageNet validation dataset from {IMAGENET_VAL_DIR}...")
    val_dataset = datasets.ImageFolder(IMAGENET_VAL_DIR, transform=preprocess)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    import tvm
    from tvm import relax
    dev = tvm.device('cpu')
    ex = tvm.runtime.load_module("build/resnet18wd4_pt_224_qdq_int8.dll")
    vm = relax.VirtualMachine(ex, dev)

    print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
    providers =['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    
    # Get the name of the model's input layer
    input_name = session.get_inputs()[0].name

    correct_tvm = 0
    correct_top1 = 0
    total_samples = 0

    print("Starting evaluation...")
    i = 0
    for images, labels in val_loader:
        images_np = images.numpy()
        labels_np = labels.numpy()

        outputs = session.run(None, {input_name: images_np})[0]
        predictions = np.argmax(outputs, axis=1)

        outputs_tvm = vm["main"](tvm.nd.array(images_np, dev)).numpy()
        pred_tvm = np.argmax(outputs_tvm, axis=1)

        correct_tvm += (pred_tvm == labels.numpy()).sum()
        correct_top1 += np.sum(predictions == labels_np)

        total_samples += labels_np.shape[0]
        print(".", end='')
        if i == 100:
            break
        i+=1
        sys.stdout.flush()
    print("")

    top1_accuracy = correct_top1 / total_samples
    top1_accuracy_tvm = correct_tvm / total_samples
    print("\n" + "="*40)
    print(f"Total Images Evaluated : {total_samples}")
    print(f"Top-1 Accuracy         : {top1_accuracy * 100:.2f}%")
    print(f"Top-1 Accuracy (TVM)   : {top1_accuracy_tvm * 100:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()
