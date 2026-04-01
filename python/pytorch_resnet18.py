import torch
import torchvision.models.quantization as quantized_models

def export_quantized_resnet18_to_onnx(onnx_file_path="quantized_resnet18.onnx"):
    # 1. Set the quantization engine.
    # 'fbgemm' is typically used for x86 CPUs, 'qnnpack' is used for ARM.
    # torchvision's pre-trained quantized models are mostly fbgemm-based.
    # Note: If you are on an Apple Silicon (M1/M2) Mac, you may need to use 'qnnpack'.
    backend = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
    torch.backends.quantized.engine = backend
    print(f"Using quantization engine: {backend}")

    # 2. Load the pre-trained quantized ResNet18 model
    print("Loading pre-trained quantized ResNet-18...")
    weights = quantized_models.ResNet18_QuantizedWeights.DEFAULT
    model = quantized_models.resnet18(weights=weights, quantize=True)

    # Set the model to evaluation mode (essential before exporting)
    model.eval()

    # 3. Create a dummy input tensor matching the input shape of ResNet-18
    # Shape: (Batch Size, Channels, Height, Width) -> (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 4. Export the model to ONNX
    print(f"Exporting model to {onnx_file_path}...")
    torch.onnx.export(
        model,                         # The PyTorch model
        dummy_input,                   # Dummy input tensor
        onnx_file_path,                # Destination file path
        export_params=True,            # Store the trained parameter weights inside the model file
        opset_version=14,              # Opset >= 13 is REQUIRED for quantized ONNX export
        do_constant_folding=True,      # Execute constant folding for optimization
        input_names=['input'],         # Define the name of the input tensor
        output_names=['output'],       # Define the name of the output tensor
        dynamic_axes={                 # Enable dynamic batch sizes (optional but recommended)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Export complete!")

if __name__ == "__main__":
    export_quantized_resnet18_to_onnx()
