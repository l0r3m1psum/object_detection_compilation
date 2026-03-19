import argparse
import pathlib
import tempfile
import collections
import copy
from typing import Container, Optional

import onnx
import onnxruntime.quantization
import torch
import torchvision
import numpy

from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat,
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

class MyCalibrationDataReader(onnxruntime.quantization.CalibrationDataReader):
    def __init__(self, data_loader, input_name):
        super().__init__()
        self.input_name = input_name
        self.data_iter = iter(data_loader)

    def get_next(self):
        try:
            data, labels = next(self.data_iter)
            return {self.input_name: data.numpy()}
        except StopIteration:
            return None

import struct
# https://github.com/onnx/onnx/issues/2182#issuecomment-3616406449
def make_dynamic_batch(
    infile: str,
    outfile: Optional[str] = None,
    batch_size: str = 'N',
    export_batch_size: Optional[int] = None
) -> None:
    if outfile is None:
        outfile = infile
    model = onnx.load(infile) # type: ignore
    graph = model.graph

    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        dims = tensor.type.tensor_type.shape.dim
        if len(dims) == 0:
            continue
        dim0 = dims[0]
        if export_batch_size is not None and dim0.HasField('dim_value') and dim0.dim_value == export_batch_size:
            dim0.ClearField('dim_value')
            dim0.dim_param = batch_size
        elif export_batch_size is None:
            dim0.ClearField('dim_value')
            dim0.dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    # Replace export_batch_size with -1 in Constant node attributes
    if export_batch_size is not None:
        for node in graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.type == 7:  # INTS
                        if export_batch_size in attr.ints:
                            attr.ints[0] = -1

    onnx.save(model, outfile) # type: ignore


def quantize_relay_for_vta(
    model_input: str | pathlib.Path | onnx.ModelProto,
    model_output: str | pathlib.Path,
    calibration_data_reader: onnxruntime.quantization.CalibrationDataReader,
    quant_format=QuantFormat.QDQ,
    op_types_to_quantize=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    use_external_data_format=False,
    calibration_providers=None,
) -> None:
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

            weight = onnx.numpy_helper.to_array(initializer_map[node.input[1]])
            # bias = onnx.numpy_helper.to_array(initializer_map[node.input[2]])
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

def try_fuse_conv_or_gemm(
    node: onnx.NodeProto,
    node_map,
    init_map,
    usage_count,
    nodes_to_remove,
    nodes_to_add,
    initializer_to_remove
) ->  None:
    # First we try to match the Q(DQ) pattern.
    q_node = node
    if node.op_type != "QuantizeLinear": return

    conv_output_name = q_node.input[0]
    if is_input_or_initializer(conv_output_name, node_map): return
    conv_node = node_map[conv_output_name]
    if conv_node.op_type != "Conv" and conv_node.op_type != "Gemm": return

    if usage_count[conv_output_name] > 1:
        print(f"Skipping {conv_node.name}: Output used by multiple nodes.")
        return

    if len(conv_node.input) < 2: return

    dq_x_name = conv_node.input[0]
    if is_input_or_initializer(dq_x_name, node_map): return
    dq_x_node = node_map[dq_x_name]
    if dq_x_node.op_type != "DequantizeLinear": return

    dq_w_name = conv_node.input[1]
    if is_input_or_initializer(dq_w_name, node_map): return
    dq_w_node = node_map[dq_w_name]
    if dq_w_node.op_type != "DequantizeLinear": return

    if len(conv_node.input) > 2:
        dq_b_name = conv_node.input[2]
        if is_input_or_initializer(dq_b_name, node_map): return
        dq_b_node = node_map[dq_b_name]
        if dq_b_node.op_type != "DequantizeLinear": return

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
        if not (onnx.numpy_helper.to_array(init_map[b_zp]) == 0).all():
            print(f"Skipping {conv_node.name}: Bias zero point is not zero.")
            return

        # TODO: check that:
        onnx.numpy_helper.to_array(init_map[b_s]) == onnx.numpy_helper.to_array(init_map[x_s]) * onnx.numpy_helper.to_array(init_map[x_s])
        # in some floating point robust way.
        initializer_to_remove.add(b_s)
        inputs.append(b)

    if conv_node.op_type == "Gemm":
        # The bias goes in a different position...
        if len(conv_node.input) > 2:
            tmp = inputs.pop()
            inputs.insert(6, tmp)
        qlinear_node = onnx.helper.make_node(
            "QGemm",
            inputs=inputs,
            outputs=[q_node.output[0]],
            name=conv_node.name + "_fused",
            domain="com.microsoft",
            **{attr.name: onnx.helper.get_attribute_value(attr) for attr in conv_node.attribute if attr.name != "beta"}
        )
    else:
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

def try_fuse_add(
    node: onnx.NodeProto,
    node_map,
    init_map,
    usage_count,
    nodes_to_remove,
    nodes_to_add,
    initializer_to_remove
) -> None:
    # First we try to match the Q(DQ) pattern.
    q_node = node
    if node.op_type != "QuantizeLinear": return

    add_output_name = q_node.input[0]
    if is_input_or_initializer(add_output_name, node_map): return
    add_node = node_map[add_output_name]
    if add_node.op_type != "Add": return

    if usage_count[add_output_name] > 1:
        print(f"Skipping {add_node.name}: Output used by multiple nodes.")
        return

    if len(add_node.input) != 2: return

    dq_a_name = add_node.input[0]
    if is_input_or_initializer(dq_a_name, node_map): return
    dq_a_node = node_map[dq_a_name]
    if dq_a_node.op_type != "DequantizeLinear": return

    dq_b_name = add_node.input[1]
    if is_input_or_initializer(dq_b_name, node_map): return
    dq_b_node = node_map[dq_b_name]
    if dq_b_node.op_type != "DequantizeLinear": return

    # If we have matched the pattern, we try to create the fused node.

    a, a_s, a_zp = dq_a_node.input[0], dq_a_node.input[1], dq_a_node.input[2]
    b, b_s, b_zp = dq_b_node.input[0], dq_b_node.input[1], dq_b_node.input[2]
    c_s, c_zp = q_node.input[1], q_node.input[2]

    inputs = [a, a_s, a_zp, b, b_s, b_zp, c_s, c_zp]

    qlinear_add_node = onnx.helper.make_node(
        "QLinearAdd",
        inputs=inputs,
        outputs=[q_node.output[0]],
        name=add_node.name + "_fused",
        domain="com.microsoft",
        **{attr.name: onnx.helper.get_attribute_value(attr) for attr in add_node.attribute}
    )
    nodes_to_add.append(qlinear_add_node)

    # Now we can mark for removal the nodes we have fused.

    nodes_to_remove.add(add_node.name)
    nodes_to_remove.add(q_node.name)

    usage_count[dq_a_node.output[0]] -= 1
    if usage_count[dq_a_node.output[0]] == 0:
        nodes_to_remove.add(dq_a_node.name)

    usage_count[dq_b_node.output[0]] -= 1
    if usage_count[dq_b_node.output[0]] == 0:
        nodes_to_remove.add(dq_b_node.name)

def fuse_qoperators(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    All the names in an onnx graph are either in
    input, output, initializer or node
    """
    graph: onnx.GraphProto = model.graph

    node_map: Dict[str, onnx.NodeProto] = {node.output[0]: node for node in graph.node}
    init_map: Dict[str, onnx.TensorProto] = {init.name: init for init in graph.initializer}

    usage_count = collections.defaultdict(int)
    for node in graph.node:
        for input_name in node.input:
            usage_count[input_name] += 1
    # NOTE: is this necessary?
    for output in graph.output:
        usage_count[output.name] += 1

    nodes_to_remove = set()
    nodes_to_add = []
    initializer_to_remove = set()
    # initializer_to_add = []

    for node in graph.node:
        # NOTE: adding more fusion without checks could break the code is two
        # consecutively go of.
        try_fuse_conv_or_gemm(
            node,
            node_map,
            init_map,
            usage_count,
            nodes_to_remove,
            nodes_to_add,
            initializer_to_remove
        )
        try_fuse_add(
            node,
            node_map,
            init_map,
            usage_count,
            nodes_to_remove,
            nodes_to_add,
            initializer_to_remove
        )

    # We create the new model.

    new_model = copy.deepcopy(model)

    new_node_list = [node for node in graph.node if node.name not in nodes_to_remove]
    new_node_list.extend(nodes_to_add)

    new_initializer_list = [initializer for initializer in graph.initializer if initializer.name not in initializer_to_remove]
    # new_initializer_list.extend(initializer_to_add)

    # TODO: cleanup the topological sorting code...

    # The ONNX graph has its edge directions from the producer node to the
    # consumer node i.e. the edge is the used_by relation e.g. Conv -> Relu here
    # the inverted (or transposed) graph is created i.e. the edge direction is
    # from consumer to producer e.g. Conv <- Relu.
    node_by_output: Dict[str, onnx.NodeProto] = {}
    for node in new_node_list:
        for output in node.output:
            node_by_output[output] = node

    topo: List[onnx.NodeProto] = []
    visited = set()

    def build_topo(n: onnx.NodeProto):
        if id(n) not in visited:
            visited.add(id(n))
            # traverse inputs to find producer nodes
            for input_name in n.input:
                producer = node_by_output.get(input_name)
                if producer is not None:
                    build_topo(producer)
            topo.append(n)

    # Iterate over all nodes to ensure disconnected components are covered.
    for node in new_node_list:
        build_topo(node)

    del new_model.graph.node[:]
    del new_model.graph.initializer[:]
    new_model.graph.node.extend(topo)
    new_model.graph.initializer.extend(new_initializer_list)

    # QLinearAdd and QGemm needs this domain
    # TODO: check if this two nodes have actually been added
    has_ms_domain = any(opset.domain == "com.microsoft" for opset in new_model.opset_import)
    if not has_ms_domain and nodes_to_add:
        opset = new_model.opset_import.add()
        opset.domain = "com.microsoft"
        opset.version = 1

    onnx.checker.check_model(new_model)

    return new_model

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantizes a ONNX model with a given dataset."
    )
    parser.add_argument(
        "-m", "--model", help="The model to quantize.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-d", "--dataset", help="The dataset to use for calibration.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-o", "--output", help="The output path for the quantized model.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-s", "--subset", help="Size of the random subset to use for calibration.", default=0, type=int
    )
    parser.add_argument(
        "-t", "--type", help="The type of quantization to use.", required=True, type=str
    )
    # TODO: batch size
    NUM_WORKERS = 4
    BATCH_SIZE = 64

    args = parser.parse_args()

    imagenet_rgb_mean = (0.485, 0.456, 0.406)
    imagenet_rgb_std = (0.229, 0.224, 0.225)
    imagenet_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(imagenet_rgb_mean, imagenet_rgb_std),
    ])

    dataset = torchvision.datasets.ImageFolder(
        args.dataset, transform=imagenet_transform
    )
    if args.subset:
        dataset = torch.utils.data.Subset(
            dataset, torch.randperm(len(dataset))[:args.subset]
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    with tempfile.TemporaryDirectory(prefix="ort.quant.") as tmp_dir:
        tmp_dir_path = pathlib.Path(tmp_dir)
        model = onnx.load(args.model)
        input_name = model.graph.input[0].name
        onnx_opset_version = next(
            opset_import.version
            for opset_import in model.opset_import if not opset_import.domain
        )
        if onnx_opset_version < 13:
            converted_model = onnx.version_converter.convert_version(model, 13)
        else:
            converted_model = model
        model_converted = tmp_dir_path / "model_converted.onnx"
        onnx.save(converted_model, model_converted)

        model_pre_proc = tmp_dir_path / "model_pre_proc.onnx"
        quant_pre_process(model_converted, model_pre_proc, auto_merge=True)

        model_optimized = tmp_dir_path / "model_optimized.onnx"
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = str(model_optimized)
        _ = onnxruntime.InferenceSession(model_pre_proc, sess_options)

        make_dynamic_batch(model_optimized, model_optimized)
        # TODO: put this in the eval_models.py file together with creating a Python file of shared files.
        make_dynamic_batch(args.model, args.model)

        if args.type == "axis":
            quantize_static(
                model_input=model_optimized,
                model_output=args.output,
                calibration_data_reader=MyCalibrationDataReader(dataloader, input_name),
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
            m = onnx.load("build/resnet18_int8_per_channel_qdq.onnx")
            m = fuse_qoperators(m)
            onnx.save(m, "build/resnet18_int8_per_channel.onnx")
        elif args.type == "tensor":
            quantize_static(
                model_input=model_optimized,
                model_output=args.output,
                calibration_data_reader=MyCalibrationDataReader(dataloader, input_name),
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
        elif args.type == "network":
            quantize_relay_for_vta(
                model_optimized,
                args.output,
                MyCalibrationDataReader(dataloader, input_name),
                QuantFormat.QOperator,
            )

if __name__ == '__main__':
    main()
