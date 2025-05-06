import typing
import logging

import torch
import torchvision

logger = logging.getLogger(__name__)

# TODO: create an interator class that does the dfs
def make_relu_not_inplace(module: torch.nn.Module):
	for name, submodule in module.named_children():
		if name == 'relu':
			submodule.inplace = False
		make_relu_not_inplace(submodule)

# https://pytorch.org/docs/stable/torch.compiler_transformations.html
# TODO: maybe use https://pytorch.org/docs/stable/fx.html#subgraph-rewriting-with-replace-pattern instead
def replace_add_inplace_with_add(graph: torch.fx.Graph) -> None:
	for node in graph.nodes:
		if node.op == 'call_function' and node.target == torch.ops.aten.add_.Tensor:
			node.target = torch.ops.aten.add.Tensor
	graph.lint()

# TODO: verify that torch.flatten(x, 1) is equal to x.view(x.size(0), -1)
# Probably a better way to do it is use: https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
def replace_flatten_with_view(graph: torch.fx.Graph, input_shape: torch.Size) -> None:
	for node in graph.nodes:
		if (node.op == 'call_function'
			and node.target == torch.ops.aten.flatten.using_ints
			and node.args[1] == 1):
			node.target = torch.ops.aten.view.default
			input_node: torch.fx.node.Node = node.args[0]
			batch_size = input_shape[0]
			node.args = input_node, [batch_size, -1]
	graph.lint()

# COCO's dataset format for 'labels'
class Target(typing.TypedDict):
	boxes: torch.Tensor # torchvision.tv_tensors.BoundingBoxes
	labels: torch.Tensor
	image_id: int
	area: torch.Tensor
	iscrowd: torch.Tensor
	mask: typing.NotRequired[torch.Tensor] # torchvision.tv_tensors.Mask

class ListDataset(torch.utils.data.Dataset):

	def __init__(self, data: list, transforms) -> None:
		super().__init__()
		self.data = data
		self.transforms = transforms

	def __getitem__(self, idx: int):
		return self.transforms(self.data[idx])

	def __len__(self) -> int:
		return len(self.data)

# TODO: https://pytorch.org/vision/main/auto_examples/transforms/plot_custom_transforms.html#how-to-write-your-own-v2-transforms
class Identity():
	def __call__(self, x):
		return x

def collate_fn(batch):
	return tuple(zip(*batch))

# https://dsp.stackexchange.com/a/20342
class MovingAverage:
	def __init__(self, n: int) -> None:
		if n < 2: raise ValueError("n < 2")
		self.y = 0.
		self.xs: list[float] = [0. for _ in range(n)]
		self.n = n
		# Data in circular buffer with new and old pointer
		#  n o
		# |3|1|2|
		self.new = 0
		self.old = 1
		self.recip = 1/n
	def update(self, value: float) -> float:
		self.new = (self.new+1)%self.n
		self.old = (self.old+1)%self.n
		self.xs[self.new] = value
		self.y = self.y + self.recip * (value - self.xs[self.old])
		return self.y

# https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/dcb285e7dea7e73d9480937d58de0e9bdfc20051/lib/Evaluator.py
# https://github.com/cocodataset/cocoapi/blob/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9/PythonAPI/pycocotools/cocoeval.py
def mAP(bdts, bgts, labels_num, iou_thresh=0.5, method='all'):
	"""
	bdts = batch detections
	bgts = batch gtound truths

	list(bdts[0].keys()) == ['boxes', 'scores', 'labels']
	list(bgts[0].keys()) == ['boxes', 'labels']

	Since this function needs to sort it is advisable to run it on the CPU"""
	assert len(bdts) == len(bgts)

	# Stable sorting is for compatibility with COCO which is compatible with Matlab
	descending_coco_compatible = dict(descending=True, stable=True)

	APs = torch.zeros(labels_num)
	# We start from index 1 since label 0 is the background class.
	for label in range(1, labels_num+1):
		bTPs = torch.empty(0, dtype=torch.bool)
		bdt_num = 0
		bgt_num = 0
		bdt_scores = torch.empty(0)

		for dts, gts in zip(bdts, bgts):

			dt_boxes = dts['boxes'][dts['labels'] == label]
			dt_scores = dts['scores'][dts['labels'] == label]
			bdt_scores = torch.concat((bdt_scores, dt_scores))
			dt_num = dt_boxes.size()[0]
			bdt_num += dt_num

			gt_boxes = gts['boxes'][gts['labels'] == label]
			gt_num = gt_boxes.size()[0]
			bgt_num += gt_num
			gt_detected = torch.zeros(gt_num, dtype=torch.bool)

			dt_scores, sort_indices = torch.sort(dt_scores, **descending_coco_compatible)
			dt_boxes = dt_boxes[sort_indices]

			TPs = torch.empty(dt_num, dtype=torch.bool)
			common_box_num = min(gt_num, dt_num)
			if gt_num < dt_num:
				TPs[common_box_num:] = 0 # i.e. it is a FP
			else:
				pass # FN could be counted

			IoUs = torchvision.ops.box_iou(dt_boxes, gt_boxes)
			max_ious, max_ious_indices = torch.max(IoUs, dim=1) \
				if IoUs.numel() != 0 else (torch.tensor([]), torch.tensor([]))
			assert dt_boxes.size()[:1] == max_ious.size()
			for i in range(common_box_num):
				j = max_ious_indices[i]
				if max_ious[i] > iou_thresh:
					if gt_detected[j]:
						TPs[i] = 0 # i.e. it is a FP
					else:
						TPs[i] = 1
						gt_detected[j] = True
				else:
					TPs[i] = 0 # i.e. it is a FP
			bTPs = torch.concat((bTPs, TPs))

		bdt_scores, sort_indices = torch.sort(bdt_scores, **descending_coco_compatible)
		bTPs = bTPs[sort_indices]
		cumsum_bTPs = torch.cumsum(bTPs, dim=0) if bTPs.numel() > 0 else 0
		# precision = TP/detections
		precisions = cumsum_bTPs/torch.arange(start=1, end=bdt_num+1)
		# recall = TP/ground truths
		recalls = cumsum_bTPs/bgt_num if bgt_num > 0 else torch.tensor([])

		if method == 'all':
			heights = torch.flip(precisions, dims=(0,))
			heights, _ = torch.cummax(heights, dim=0)
			heights = torch.flip(heights, dims=(0,))
			heights = heights[:-1]
			widths = torch.diff(recalls)
			# because of one indexing
			APs[label-1] = torch.dot(heights, widths)
		else:
			raise ValueError(f'method not supported: {method}')

	print(APs)
	res = torch.mean(APs)
	return res
