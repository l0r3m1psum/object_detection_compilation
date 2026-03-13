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
            c1 = R.nn.conv2d(x, w1, padding=(1, 1))
            c2 = R.nn.conv2d(x, w2, padding=(1, 1))
            c3 = R.nn.conv2d(x, w3, padding=(1, 1))
            add1 = R.nn.relu(
                R.qnn.add(
                    c2, R.const(1.0), R.const(0),
                    c3, R.const(1.0), R.const(0),
                    R.const(1.0), R.const(0),
                )
            )
            add2 = R.qnn.add(
                c1, R.const(1.0), R.const(0),
                add1, R.const(1.0), R.const(0),
                R.const(1.0), R.const(0),
            )
            R.output(add2)
        return add2

@R.function
def example_func(x: R.Tensor((10, 10), "float32"), y: R.Tensor((10, 10), "float32")):
    with R.dataflow():
        # These are bindings within a DataflowBlock
        lv0 = R.add(x, y)
        lv1 = R.multiply(lv0, x)
        gv = R.subtract(lv1, y)
        R.output(gv)
    return gv

def show_dfgraph(func: relax.Function, name: str):
    visitor = DataflowBlockGraphvizVisitor()
    visitor.visit_expr(func)

    for i, dot_graph in enumerate(visitor.dot_graphs):
        # print(dot_graph)
        graph = graphviz.Source(dot_graph)
        graph.view(name)

if __name__ == "__main__":
    mod = CascadedAddsModule
    mod.show()
    show_dfgraph(mod["main"], "before.gv")
    mod = vtar.relax.transform.ReScale()(mod)
    mod.show()
    show_dfgraph(mod["main"], "after.gv")
