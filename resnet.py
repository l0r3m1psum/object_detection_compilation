"""An implemention of ResNet-50

The original paper is
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Presentation by Microsoft
[Microsoft Vision Model ResNet-50: Pretrained vision model built with web-scale data](https://www.youtube.com/watch?v=oQqxkO3BN3Q)

The implementation is inspired by
[Pytorch ResNet implementation from Scratch](https://www.youtube.com/watch?v=DkNIBBBvcPs)
and
[ResNet-PyTorch](https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py)
only inspired because they are both wrong since they do not remove the bias from
the convolutions.
"""

from tvm.relax.frontend import nn
from tvm import relax

"""
class BatchNorm2D(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm2D, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters: scale (gamma) and shift (beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running stats (not learnable, but saved)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        # x shape: (N, C, H, W)
        if self.training:
            # Compute mean and variance across (N, H, W)
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
        else:
            # Use running stats during inference
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)

        # Normalize
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma.view(1, -1, 1, 1) * x_hat + self.beta.view(1, -1, 1, 1)

        return out
"""

# https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
# https://pytorch.org/docs/stable/generated/torch.nn.functional.batch_norm.html
# https://isaac-the-man.dev/posts/normalization-strategies/#batch-normalization
class BatchNorm2D(nn.Module):
	# num_features is the number of channels in the image
	# the running_mean is calculated per dimension
	def __init__(
			self,
			num_features: int,
			eps: float = 1e-5,
			momentum: float = 0.1
		) -> None:
		super().__init__()
		C = num_features,
		self.gamma = nn.Parameter(C)
		self.beta = nn.Parameter(C)
		self.moving_mean = nn.Parameter(C)
		self.moving_var = nn.Parameter(C)
		# https://discuss.pytorch.org/t/what-num-batches-tracked-in-the-new-bn-is-for/27097
		self.num_batches_tracked = nn.Parameter((1,))
		self.eps = eps
		self.momentum = momentum

	def forward(self, x: nn.core.Tensor) -> nn.core.Tensor:
		# x is a 4D tensor (N, C, H, W)
		# mean is calculated as across the batch and the height and width dimensions
		# moving_mean.shape == C,
		# +---+     +---+     +---+          |
		# |+---+....|+---+....|+---+     \   |
		# ||+---+   ||+---+   ||+---+ ->  \  |
		# +||   |   +||   |   +||   |      \ |
		#  +|   |....+|   |....+|   |        |
		#   +---+     +---+     +---+        |
		x: relax.expr.Call = relax.op.nn.batch_norm(
			data=x._expr,
			gamma=self.gamma._expr,
			beta=self.beta._expr,
			moving_mean=self.moving_mean._expr,
			moving_var=self.moving_var._expr,
			axis=1, # This should be the axis of the channel (N, C, H, W)
			epsilon=self.eps,
			center=True,
			scale=True,
			momentum=self.momentum,
			training=False
		)
		x: tuple[nn.core.Tensor] = nn.wrap_nested(x, 'batch_norm')
		return x[0]

class Identity(nn.Module):
	def forward(self, x: nn.core.Tensor) -> nn.core.Tensor:
		return x

class MaxPool2D(nn.Module):
	def __init__(
			self,
			kernel_size: int,
			stride: int | None = None,
			padding: int = 0,
			dilation: int = 1,
			ceil_mode: bool = False,
		) -> None:
		super().__init__()
		self.kernel_size = kernel_size
		self.stride = stride if stride is not None else kernel_size
		self.padding = padding
		self.dilation = dilation
		self.ceil_mode = ceil_mode
	def forward(self, x: nn.core.Tensor) -> nn.core.Tensor:
		x: relax.expr.Call = relax.op.nn.max_pool2d(
			x._expr,
			self.kernel_size,
			self.stride,
			self.padding,
			self.dilation,
			self.ceil_mode
		)
		x = nn.wrap_nested(x, 'max_pool2d')
		return x

class Sequential(nn.Module):
	def __init__(self, *args: list[nn.Module]) -> None:
		super().__init__()
		# Using a normal Python's list does not register the modules correctly.
		self.modules = nn.ModuleList(args)
	def forward(self, x):
		return self.modules(x)

class AdaptiveAvgPool2D(nn.Module):
	def __init__(self, output_size: int) -> None:
		super().__init__()
		self.output_size = output_size
	def forward(self, x):
		x: relax.expr.Call = relax.op.nn.adaptive_avg_pool2d(x._expr, self.output_size)
		x = nn.wrap_nested(x, 'adaptive_avg_pool2d')
		return x

class Bottleneck(nn.Module):
	expansion = 4
	def __init__(
			self,
			in_chans: int,
			out_chans: int,
			in_downsample: nn.Module | None = None,
			stride: int = 1
		) -> None:
		super().__init__()

		self.conv1 = nn.Conv2D(in_chans, out_chans, kernel_size=1, stride=1, padding=0, bias=False)
		self.batch_norm1 = BatchNorm2D(out_chans)
		self.conv2 = nn.Conv2D(out_chans, out_chans, kernel_size=3, stride=stride, padding=1, bias=False)
		self.batch_norm2 = BatchNorm2D(out_chans)
		self.conv3 = nn.Conv2D(out_chans, out_chans*self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
		self.batch_norm3 = BatchNorm2D(out_chans*self.expansion)

		self.in_downsample = in_downsample if in_downsample else Identity()
		self.relu = nn.ReLU()

	def forward(self, x):
		x_downsample = self.in_downsample(x)
		x = self.relu(self.batch_norm1(self.conv1(x)))
		x = self.relu(self.batch_norm2(self.conv2(x)))
		x = self.batch_norm3(self.conv3(x))
		x = self.relu(x + x_downsample)
		return x

class ResNet(nn.Module):
	def __init__(
			self,
			ResBlock: type[nn.Module],
			layer0_depth: int,
			layer1_depth: int,
			layer2_depth: int,
			layer3_depth: int,
			num_classes: int,
			in_chans: int = 3
		) -> None:
		super().__init__()
		self.out_chans = 64

		self.conv1 = nn.Conv2D(in_chans, self.out_chans, kernel_size=7, stride=2, padding=3, bias=False)
		self.batch_norm1 = BatchNorm2D(self.out_chans)
		self.relu = nn.ReLU()
		self.max_pool = MaxPool2D(kernel_size=3, stride=2, padding=1)

		self.layer1 = self._make_layer(ResBlock, layer0_depth, 64,  1)
		self.layer2 = self._make_layer(ResBlock, layer1_depth, 128, 2)
		self.layer3 = self._make_layer(ResBlock, layer2_depth, 256, 2)
		self.layer4 = self._make_layer(ResBlock, layer3_depth, 512, 2)

		self.avg_pool = AdaptiveAvgPool2D(1)
		self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

	def forward(self, x):
		x1 = self.max_pool(self.relu(self.batch_norm1(self.conv1(x))))

		x2 = self.layer1(x1)
		x3 = self.layer2(x2)
		x4 = self.layer3(x3)
		x5 = self.layer4(x4)

		y = self.avg_pool(x5)
		y = y.reshape(y.shape[0], -1)
		y = self.fc(y) # logits

		# Intermediate results are used for FPN.
		return x1, x2, x3, x4, x5, y

	def _make_layer(
			self,
			ResBlock: type[nn.Module],
			depth: int,
			planes: int,
			stride: int = 1,
		) -> nn.Module:

		new_out_chans = planes*ResBlock.expansion

		in_downsample = None
		if stride != 1 or self.out_chans != new_out_chans:
			in_downsample = Sequential(
				nn.Conv2D(self.out_chans, new_out_chans, kernel_size=1, stride=stride, bias=False),
				BatchNorm2D(new_out_chans)
			)

		layers = [ResBlock(self.out_chans, planes, in_downsample, stride)]
		self.out_chans = new_out_chans

		for _ in range(depth-1):
			layers.append(ResBlock(self.out_chans, planes))

		return Sequential(*layers)

def ResNet50() -> nn.Module:
	num_classes = 1000
	channels = 3
	return ResNet(Bottleneck, 3, 4, 6, 3, num_classes, channels)

# https://arxiv.org/abs/1612.03144
# https://rumn.medium.com/f573a889c7b1
# https://jonathan-hui.medium.com/45b227b9106c
class ResNet50FPN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.resnet50 = ResNet50()
		self.lateral_conv1 = nn.Conv2D(256, 256,  kernel_size=1, stride=1, padding=0, bias=False)
		self.lateral_conv2 = nn.Conv2D(512, 256,  kernel_size=1, stride=1, padding=0, bias=False)
		self.lateral_conv3 = nn.Conv2D(1024, 256, kernel_size=1, stride=1, padding=0, bias=False)
		self.lateral_conv4 = nn.Conv2D(2048, 256, kernel_size=1, stride=1, padding=0, bias=False)

	def forward(self, x):
		_, c2, c3, c4, c5, _ = self.resnet50(x)
		p5 = self.lateral_conv4(c5)
		ip5 = nn.interpolate(p5, scale_factor=2, mode='nearest')
		p4 = self.lateral_conv3(c4) + ip5
		ip4  = nn.interpolate(p4, scale_factor=2, mode='nearest')
		p3 = self.lateral_conv2(c3) + ip4
		ip3 = nn.interpolate(p3, scale_factor=2, mode='nearest')
		p2 = self.lateral_conv1(c2) + ip3
		return p2, p3, p4, p5

mod = ResNet50()
# print(list(mod.named_parameters()))
# print(list(mod.parameters()))
# print(list(mod.state_dict()))
# TODO: try mod.load_state_dict() con https://huggingface.co/microsoft/resnet-50
# https://huggingface.co/microsoft/resnet-50/resolve/main/model.safetensors
import safetensors.torch
state_dict = safetensors.torch.load_file('resnet-50\\model.safetensors')
print(len(mod.state_dict()), len(state_dict.keys()))
irmod, params_spec = mod.export_tvm(
	{"forward": {"x": nn.spec.Tensor((2, 3, 224, 224), "float32")}},
	debug=True
)

mod = ResNet50FPN()
irmod, params_spec = mod.export_tvm(
	{"forward": {"x": nn.spec.Tensor((2, 3, 224, 224), "float32")}},
	debug=True
)

# https://tvm.d2l.ai/chapter_common_operators/batch_norm.html
