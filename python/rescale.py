import tvm
from tvm import relax, ir, tir
from tvm.script import ir as I, relax as R

import vtar

import graphviz

@relax.expr_functor.visitor
class DataflowBlockGraphvizVisitor(relax.expr_functor.PyExprVisitor):
    def __init__(self):
        self.dot_graphs =[]

    def visit_dataflow_block_(self, block: relax.DataflowBlock) -> None:
        dot_lines =[
            "digraph DataflowBlock {",
            "    node[shape=record, style=filled, fillcolor=lightgrey];"
        ]

        for binding in block.bindings:
            var = binding.var
            val = binding.value
            var_name = var.name_hint

            op_name = val.__class__.__name__
            if isinstance(val, relax.Call) and hasattr(val.op, "name"):
                op_name = val.op.name

            dot_lines.append(f'    "{var_name}" [label="{{ {var_name} | {op_name} }}"];')

            used_vars = relax.analysis.free_vars(val)

            for used_var in used_vars:
                used_name = used_var.name_hint
                dot_lines.append(f'    "{used_name}" -> "{var_name}";')

            constants = set()
            def collect_consts(e: relax.Expr):
                if isinstance(e, relax.Constant):
                    constants.add(e)

            relax.analysis.post_order_visit(val, collect_consts)

            for c in constants:
                # Use memory address to uniquely identify the constant node in Graphviz.
                # If a constant is shared across multiple bindings, this ensures they
                # accurately point to the same Graphviz node.
                const_id = f"const_{id(c)}"

                # Attempt to build a descriptive label (e.g. "Const | [16, 32] | float32")
                label_parts = ["Const"]
                if getattr(c, "struct_info", None) is not None:
                    sinfo = c.struct_info
                    if getattr(sinfo, "shape", None) is not None:
                        # Sanitize curly braces to prevent breaking Graphviz record shapes
                        shape_str = str(sinfo.shape).replace("{", "[").replace("}", "]")
                        label_parts.append(shape_str)
                    if getattr(sinfo, "dtype", None) is not None:
                        label_parts.append(str(sinfo.dtype))

                label_str = " | ".join(label_parts)

                dot_lines.append(f'    "{const_id}"[label="{{ {label_str} }}", fillcolor=white];')
                dot_lines.append(f'    "{const_id}" -> "{var_name}";')

            # Traverse into the value's expression in case there are nested functions/blocks
            self.visit_expr(val)

        dot_lines.append("}")
        self.dot_graphs.append("\n".join(dot_lines))
        super().visit_dataflow_block_(block)

@I.ir_module
class CascadedAddsModule:
    @R.function
    def main(
        x: R.Tensor((1, 16, 32, 32), "int8"),
        w1: R.Tensor((16, 16, 3, 3), "int8"),
        w2: R.Tensor((16, 16, 3, 3), "int8"),
        w3: R.Tensor((16, 16, 3, 3), "int8")
    ):
        with R.dataflow():
            c1 = R.qnn.conv2d(
                x, R.const(1.0), R.const(0, "int8"),
                w1, R.const(1.0), R.const(0, "int8"),
                R.const(1.0), R.const(0, "int8"),
                padding=(1, 1)
            )
            c2 = R.qnn.conv2d(
                x, R.const(1.0), R.const(0, "int8"),
                w2, R.const(1.0), R.const(0, "int8"),
                R.const(1.0), R.const(0, "int8"),
                padding=(1, 1)
            )
            c3 = R.qnn.conv2d(
                x, R.const(1.0), R.const(0, "int8"),
                w3, R.const(1.0), R.const(0, "int8"),
                R.const(1.0), R.const(0, "int8"),
                padding=(1, 1)
            )
            add1 = R.nn.relu(
                R.qnn.add(
                    c2, R.const(1.0), R.const(0, "int8"),
                    c3, R.const(1.0), R.const(0, "int8"),
                    R.const(1.0), R.const(0, "int8"),
                )
            )
            add2 = R.qnn.add(
                c1, R.const(1.0), R.const(0, "int8"),
                add1, R.const(1.0), R.const(0, "int8"),
                R.const(1.0), R.const(0, "int8"),
            )
            R.output(add2)
        return add2

def show_dfgraph(func: relax.Function, name: str):
    visitor = DataflowBlockGraphvizVisitor()
    visitor.visit_expr(func)

    for i, dot_graph in enumerate(visitor.dot_graphs):
        # print(dot_graph)
        view_dot_graph_tkinter(dot_graph)
        # graph = graphviz.Source(dot_graph)
        # graph.view(name)

import tkinter
import subprocess

def view_dot_graph_tkinter(dot_string: str):
    try:
        process = subprocess.Popen(
            ("dot", "-Tpng", "-Gdpi=300"),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        png_data, err_data = process.communicate(input=dot_string.encode('utf-8'))

        if process.returncode != 0:
            raise RuntimeError(f"Graphviz failed to render: {err_data.decode('utf-8')}")

    except FileNotFoundError:
        raise RuntimeError("The 'dot' command was not found. Please install Graphviz on your system.")

    root = tkinter.Tk()
    root.title("TVM DataflowGraph Viewer")

    # FIXME: this is so shit.
    photo = tkinter.PhotoImage(data=png_data).subsample(4, 4)

    img_width, img_height = photo.width(), photo.height()

    root.geometry(f"{img_width + 40}x{img_height + 40}")

    label = tkinter.Label(root, image=photo, bg="white")
    label.pack(expand=True, fill="both")

    # TODO: open more multiple windows, this should be probably be called from another thread.
    root.mainloop()

if __name__ == "__main__":
    mod = CascadedAddsModule
    mod.show()
    # show_dfgraph(mod["main"], "before.gv")
    mod = vtar.relax.transform.ReScale()(mod)
    mod.show()
    # show_dfgraph(mod["main"], "after.gv")
