import argparse
import enum
import pathlib

import onnx
import tvm
import vtar.relax.frontend.onnx

class Flags(enum.Enum):
    rescale = enum.auto()
    requantize = enum.auto()
    verbose = enum.auto()

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compiles a model for the CPU for easier evaluation.",
    )
    parser.add_argument(
        "-m", "--model", help="The model to compile.", required=True, type=pathlib.Path,
    )
    parser.add_argument(
        "-f", "--flags", nargs="+", help="The flags to customize the compilation.", type=Flags.__getitem__,
    )
    parser.add_argument(
        "-o", "--output", help="The output file of the compilation.", required=True, type=pathlib.Path,
    )

    args = parser.parse_args()

    transforms = []

    if Flags.verbose in args.flags:
        transforms.append(tvm.ir.transform.PrintIR())
    transforms.append(vtar.relax.transform.RewriteQDQPatterns())
    if Flags.requantize in args.flags:
        transforms.append(vtar.relax.transform.ReQuantize())
    if Flags.rescale in args.flags:
        transforms.append(vtar.relax.transform.ReScale())
    transforms.append(vtar.relax.transform.LowerQNNOps())
    transforms.append(tvm.relax.transform.FoldConstant())
    if Flags.verbose in args.flags:
        transforms.append(tvm.ir.transform.PrintIR())

    graph = onnx.load(args.model)
    mod = vtar.relax.frontend.onnx.from_onnx(graph)
    seq = tvm.ir.transform.Sequential(transforms)
    mod = seq(mod)
    ex = tvm.compile(mod)
    ex.export_library(args.output)

if __name__ == '__main__':
    main()
