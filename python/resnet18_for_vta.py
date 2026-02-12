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

def quantize_relay_for_vta(
    model_input: str | pathlib.Path | onnx.ModelProto,
    model_output: str | pathlib.Path,
    calibration_data_reader: CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibration_providers=None,
):
    """Performs per-network symmetric quantization for the activations using
    MinMax calibration and power of two symmetric quantization for the weights
    (the biases are quantized according to the ONNX specification).

    Implementation stolen and adapted from onnxruntime.quantization.quantize_static

    https://github.com/apache/tvm/blob/v0.18.0/python/tvm/relay/quantize/quantize.py
    """

    per_channel = False
    reduce_range = False
    activation_type = QuantType.QInt8
    weight_type = QuantType.QInt8
    calibrate_method = CalibrationMethod.MinMax
    nodes_to_exclude = nodes_to_exclude or []
    nodes_to_quantize = nodes_to_quantize or []
    op_types_to_quantize = op_types_to_quantize or []
    mode = QuantizationMode.QLinearOps

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    model = (
        save_and_reload_model_with_shape_infer(model_input)
        if isinstance(model_input, onnx.ModelProto)
        else load_model_with_shape_infer(pathlib.Path(model_input))
    )

    pre_processed: bool = model_has_pre_process_metadata(model)
    if not pre_processed:
        logging.warning(
            "Please consider to run pre-processing before quantization. Refer to example: "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

    extra_options = extra_options={
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }

    calib_extra_options_keys = [
        ("CalibTensorRangeSymmetric", "symmetric"), # ???
    ]
    calib_extra_options = {
        # key: extra_options.get(name) for (name, key) in calib_extra_options_keys if name in extra_options
    }

    updated_model = update_opset_version(model, weight_type)
    is_model_updated = updated_model is not model
    if is_model_updated:
        model = updated_model

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as quant_tmp_dir:
        if is_model_updated:
            # Update model_input and avoid to use the original one
            model_input = copy.deepcopy(model)

        if isinstance(model_input, onnx.ModelProto):
            output_path = pathlib.Path(quant_tmp_dir).joinpath("model_input.onnx").as_posix()
            onnx.save_model(
                model_input,
                output_path,
                save_as_external_data=True,
            )
            model_input = output_path

        calibrator = create_calibrator(
            pathlib.Path(model_input),
            op_types_to_quantize,
            augmented_model_path=pathlib.Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=use_external_data_format,
            providers=calibration_providers,
            extra_options=calib_extra_options,
        )

        calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_data()
        if not isinstance(tensors_range, TensorsData):
            raise TypeError(
                f"Unexpected type {type(tensors_range)} for tensors_range and calibrator={type(calibrator)}."
            )
        del calibrator

    global_min = float("+inf")
    global_max = float("-inf")
    for tensor, data in tensors_range.items():
        global_min = min(global_min, data.lowest)
        global_max = max(global_max, data.highest)
    # global_scale = max(abs(global_min), abs(global_max)) / 127.0
    for data in tensors_range.values():
        data.lowest = global_min
        data.highest = global_max


    overrides = {}
    initializer_map = {
        init.name: init
        for init in model.graph.initializer
    }
    for node in model.graph.node:
        if (node.op_type in ["Conv", "Gemm"]
            and len(node.input) > 2
            and node.input[1] in initializer_map
            and node.input[2] in initializer_map):

            weight = numpy_helper.to_array(initializer_map[node.input[1]])
            # bias = numpy_helper.to_array(initializer_map[node.input[2]])
            weight_scale = 2.**(numpy.ceil(numpy.log2(numpy.max(numpy.abs(weight)))) - (8-1))

            overrides[node.input[1]] = [{
                "quant_type": QuantType.QInt8,
                "scale": weight_scale,
                "zero_point": 0,
            }]

            # This should be done automatically
            # overrides[node.input[2]] = [{
            #     "quant_type": QuantType.QInt8,
            #     "scale": weight_scale*global_scale,
            #     "zero_point": 0,
            # }]

    extra_options["TensorQuantOverrides"] = overrides

    check_static_quant_arguments(quant_format, activation_type, weight_type)

    if quant_format is QuantFormat.QOperator:
        quantizer = ONNXQuantizer(
            model,
            per_channel,
            reduce_range,
            mode,
            True,  # static
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )
    else:
        quantizer = QDQQuantizer(
            model,
            per_channel,
            reduce_range,
            weight_type,
            activation_type,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            extra_options,
        )

    quantizer.quantize_model()
    quantizer.model.save_model_to_file(model_output, use_external_data_format)
    if not pre_processed:
        logging.warning(
            "Please consider pre-processing before quantization. See "
            "https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification"
            "/cpu/ReadMe.md "
        )

def is_input_or_initializer(name: str, map: Container[str]):
    # assert name in init_map or node in input_map
    return name not in map

def fuse_qlinear_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    All the names in an onnx graph are either in
    input, output, initializer or node
    """
    # TODO: fuse also Gemm
    graph: onnx.GraphProto = model.graph

    node_map: Dict[str, onnx.NodeProto] = {node.output[0]: node for node in graph.node}
    init_map: Dict[str, onnx.TensorProto] = {init.name: init for init in graph.initializer}

    # Map output_name -> usage_count (for safety checks)
    usage_count = collections.defaultdict(int)
    for node in graph.node:
        for input_name in node.input:
            usage_count[input_name] += 1
    for output in graph.output:
        usage_count[output.name] += 1

    nodes_to_remove = set()
    nodes_to_add = []
    initializer_to_remove = set()
    # initializer_to_add = []

    for node in graph.node:
        # First we try to match the Q(DQ) pattern.

        q_node = node
        if node.op_type != "QuantizeLinear": continue

        conv_output_name = q_node.input[0]
        if is_input_or_initializer(conv_output_name, node_map): continue
        conv_node = node_map[conv_output_name]
        if conv_node.op_type != "Conv": continue

        if usage_count[conv_output_name] > 1:
            print(f"Skipping {conv_node.name}: Output used by multiple nodes.")
            continue

        if len(conv_node.input) < 2: continue

        dq_x_name = conv_node.input[0]
        if is_input_or_initializer(dq_x_name, node_map): continue
        dq_x_node = node_map[dq_x_name]
        if dq_x_node.op_type != "DequantizeLinear": continue

        dq_w_name = conv_node.input[1]
        if is_input_or_initializer(dq_w_name, node_map): continue
        dq_w_node = node_map[dq_w_name]
        if dq_w_node.op_type != "DequantizeLinear": continue

        if len(conv_node.input) > 2:
            dq_b_name = conv_node.input[2]
            if is_input_or_initializer(dq_b_name, node_map): continue
            dq_b_node = node_map[dq_b_name]
            if dq_b_node.op_type != "DequantizeLinear": continue

        # If we have matched the patter we try to create the fused node.

        x, x_s, x_zp = dq_x_node.input[0], dq_x_node.input[1], dq_x_node.input[2]
        w, w_s, w_zp = dq_w_node.input[0], dq_w_node.input[1], dq_w_node.input[2]
        y_s, y_zp = q_node.input[1], q_node.input[2]
        inputs = [x, x_s, x_zp, w, w_s, w_zp, y_s, y_zp]
        if len(conv_node.input) > 2:
            b, b_s, b_zp = dq_b_node.input[0], dq_b_node.input[1], dq_b_node.input[2]
            if b_zp not in init_map:
                pass
                # TODO: emit warning that must be set to 0
            initializer_to_remove.add(b_zp)
            if not (numpy_helper.to_array(init_map[b_zp]) == 0).all():
                print(f"Skipping {conv_node.name}: Bias zero point is not zero.")
                continue

            # TODO: check that:
            numpy_helper.to_array(init_map[b_s]) == numpy_helper.to_array(init_map[x_s]) * numpy_helper.to_array(init_map[x_s])
            # in some floating point robust way.
            initializer_to_remove.add(b_s)
            inputs.append(b)

        qlinear_node = onnx.helper.make_node(
            "QLinearConv",
            inputs=inputs,
            outputs=[q_node.output[0]],
            name=conv_node.name + "_fused",
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_node.attribute}
        )
        nodes_to_add.append(qlinear_node)

        # Now we can mark for removal the nodes we have fused.

        nodes_to_remove.add(conv_node.name)
        nodes_to_remove.add(q_node.name)

        usage_count[dq_x_node.output[0]] -= 1
        if usage_count[dq_x_node.output[0]] == 0:
            nodes_to_remove.add(dq_x_node.name)

        usage_count[dq_w_node.output[0]] -= 1
        if usage_count[dq_w_node.output[0]] == 0:
            nodes_to_remove.add(dq_w_node.name)

        if len(conv_node.input) > 2:
            usage_count[dq_b_node.output[0]] -= 1
            if usage_count[dq_b_node.output[0]] == 0:
                nodes_to_remove.add(dq_b_node.name)

    # We create the new model.

    new_model = copy.deepcopy(model)

    new_node_list = [node for node in graph.node if node.name not in nodes_to_remove]
    new_node_list.extend(nodes_to_add)

    new_initializer_list = [initializer for initializer in graph.initializer if initializer.name not in initializer_to_remove]
    # new_initializer_list.extend(initializer_to_add)

    del new_model.graph.node[:]
    del new_model.graph.initializer[:]
    new_model.graph.node.extend(new_node_list)
    new_model.graph.initializer.extend(new_initializer_list)

    # TODO: sort nodes topologically.
    # onnx.checker.check_model(new_model)

    return new_model

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
                        opset_version=13)

    not_resnet_per_channel = not os.path.exists("build/resnet18_int8_per_channel.onnx")
    not_resnet_per_tensor = not os.path.exists("build/resnet18_int8_per_tensor.onnx")
    not_resnet_per_network = not os.path.exists("build/resnet18_int8_per_network.onnx")
    if not_resnet_per_tensor or not_resnet_per_network or not_resnet_per_channel:
        calib_dataset = datasets.Imagenette(root='build/dataset', split='train', download=False, transform=transform)
        calib_data_loader = DataLoader(calib_dataset, batch_size=1, shuffle=False)

        if not os.path.exists("build/resnet18_pre_proc.onnx"):
            # TODO: make arguments of this function explicit (and understand them)
            quant_pre_process("build/resnet18.onnx", "build/resnet18_pre_proc.onnx")
        if not_resnet_per_channel:
            quantize_static(
                model_input="build/resnet18_pre_proc.onnx",
                model_output="build/resnet18_int8_per_channel.onnx",
                calibration_data_reader=MyCalibrationDataReader(calib_data_loader),
                quant_format=QuantFormat.QDQ,
                op_types_to_quantize=None,
                per_channel=True,
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
        if not_resnet_per_tensor:
            quantize_static(
                model_input="build/resnet18_pre_proc.onnx",
                model_output="build/resnet18_int8_per_tensor.onnx",
                calibration_data_reader=MyCalibrationDataReader(calib_data_loader),
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
        if not_resnet_per_network:
            quantize_relay_for_vta(
                "build/resnet18_pre_proc.onnx",
                "build/resnet18_int8_per_network.onnx",
                MyCalibrationDataReader(calib_data_loader),
                QuantFormat.QOperator,
            )

    import ctypes
    vta_fsim = ctypes.CDLL("vta_fsim")
    env = vtar.get_env()
    target = tvm.target.Target(env.target, host=env.target_host)
    dev = tvm.device(str(env.target))
    # TODO: if not present emit warning and compile model that always answers with one class.
    ex_pn = tvm.runtime.load_module("build/resnet18_int8_per_network.dll")
    vm_pn = relax.VirtualMachine(ex_pn, dev)
    ex_pt = tvm.runtime.load_module("build/resnet18_int8_per_tensor.dll")
    vm_pt = relax.VirtualMachine(ex_pt, dev)

    test_dataset = datasets.Imagenette(root='build/dataset', split='val', download=False, transform=transform)
    test_dataset = torch.utils.data.Subset(test_dataset, torch.randperm(len(test_dataset))[:200])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # For some reason this does not fuse all Q(DQ) convolutions.
    sess_opts = onnxruntime.SessionOptions()
    sess_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_opts.optimized_model_filepath = "build/resnet18_int8_per_channel_opt.onnx"

    m = onnx.load("build/resnet18_int8_per_channel.onnx")
    m = fuse_qlinear_conv(m)
    onnx.save(m, "build/resnet18_int8_per_channel_fused.onnx")

    sess_fp32 = onnxruntime.InferenceSession("build/resnet18.onnx")
    sess_int8_pc = onnxruntime.InferenceSession("build/resnet18_int8_per_channel_fused.onnx")
    sess_int8_pt = onnxruntime.InferenceSession("build/resnet18_int8_per_tensor.onnx")
    sess_int8_pn = onnxruntime.InferenceSession("build/resnet18_int8_per_network.onnx")

    correct_fp32 = 0
    correct_int8_pc = 0
    correct_int8_pt = 0
    correct_int8_pn = 0
    correct_tvm_pn = 0
    correct_tvm_pt = 0
    total = 0

    for images, labels in test_loader:
        images = images.cpu()

        outputs_fp32 = run_inference(sess_fp32, images)
        pred_fp32 = numpy.argmax(outputs_fp32, axis=1)

        outputs_int8_pc = run_inference(sess_int8_pc, images)
        pred_int8_pc = numpy.argmax(outputs_int8_pc, axis=1)

        outputs_int8_pt = run_inference(sess_int8_pt, images)
        pred_int8_pt = numpy.argmax(outputs_int8_pt, axis=1)

        outputs_int8_pn = run_inference(sess_int8_pn, images)
        pred_int8_pn = numpy.argmax(outputs_int8_pn, axis=1)

        outputs_tvm_pn = vm_pn["main"](tvm.nd.array(images.numpy(), dev)).numpy()
        pred_tvm_pn = numpy.argmax(outputs_tvm_pn, axis=1)

        outputs_tvm_pt = vm_pt["main"](tvm.nd.array(images.numpy(), dev)).numpy()
        pred_tvm_pt = numpy.argmax(outputs_tvm_pt, axis=1)

        correct_fp32 += (pred_fp32 == labels.numpy()).sum()
        correct_int8_pc += (pred_int8_pc == labels.numpy()).sum()
        correct_int8_pt += (pred_int8_pt == labels.numpy()).sum()
        correct_int8_pn += (pred_int8_pn == labels.numpy()).sum()
        correct_tvm_pn += (pred_tvm_pn == labels.numpy()).sum()
        correct_tvm_pt += (pred_tvm_pt == labels.numpy()).sum()
        total += labels.size(0)
        print(".", end="")
        sys.stdout.flush()
    print("")

    print(f"Accuracy FP32:                {correct_fp32/total*100:.2f}%")
    print(f"Accuracy INT8 per-channel:    {correct_int8_pc/total*100:.2f}%")
    print(f"Accuracy INT8 per-tensor:     {correct_int8_pt/total*100:.2f}%")
    print(f"Accuracy INT8 per-network:    {correct_int8_pn/total*100:.2f}%")
    print(f"Accuracy TVM IOA per-tensor:  {correct_tvm_pt/total*100:.2f}%")
    print(f"Accuracy TVM IOA per-network: {correct_tvm_pn/total*100:.2f}%")

if __name__ == "__main__":
    main()
