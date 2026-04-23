import types

import tvm.script.relax

# Monkey patching TVM to use the new operators
if not hasattr(tvm.script.relax, "qnn"):
    tvm.script.relax.qnn = types.SimpleNamespace()

from . import build_module
from .environment import get_env, Environment
from . import intrin
from . import dlight
from . import tir
from . import relax
from . import topi
from . import utils

tvm.script.relax.bidi_shift = relax.op.bidi_shift
tvm.script.relax.qnn.add = relax.op.qnn.add
tvm.script.relax.qnn.conv2d = relax.op.qnn.conv2d
tvm.script.relax.qnn.avg_pool2d = relax.op.qnn.avg_pool2d
tvm.script.relax.qnn.linear = relax.op.qnn.linear

del tvm, types