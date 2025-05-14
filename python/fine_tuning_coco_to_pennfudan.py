import io
import logging
import os
import random
import urllib.request
import warnings
import zipfile

import utils

import torch
import torchvision.tv_tensors
from torchvision.transforms import v2 as T

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
random.seed(42)
# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(42)

"""
One note on the labels. The model considers class 0 as background. If your
dataset does not contain the background class, you should not have 0 in your
labels. For example, assuming you have just two classes, cat and dog, you can
define 1 (not 0) to represent cats and 2 to represent dogs. So, for instance,
if one of the images has both classes, your labels tensor should look like
[1, 2].

Additionally, if you want to use aspect ratio grouping during training (so that
each batch only contains images with similar aspect ratios), then it is
recommended to also implement a get_height_and_width method, which returns the
height and the width of the image. If this method is not provided, we query all
elements of the dataset via __getitem__ , which loads the image in memory and
is slower than if a custom method is provided.
"""

dataset_path = 'D:\\Datasets\\PennFudanPed.zip'

class PennFudanDataset(torch.utils.data.Dataset):

	def __init__(self) -> None:
		super().__init__()

		dataset_url = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'
		dataset_bytes = bytearray()

		try:
			if not os.path.exists(dataset_path):
				logger.info('Downloading dataset...')
				with urllib.request.urlopen(dataset_url) as url, \
					open(dataset_path, 'wb') as f:
					while True:
						dataset_chunk = url.read(1024)
						if not dataset_chunk: break
						dataset_bytes += dataset_chunk
						f.write(dataset_chunk)
			else:
				logger.info('Reading dataset...')
				with open(dataset_path, 'rb') as f:
					dataset_bytes += f.read()
		except Exception as ex:
			raise RuntimeError('Unable to load the dataset') from ex

		self.root = zipfile.ZipFile(io.BytesIO(dataset_bytes), 'r')

		sorted_namelist = self.root.namelist()
		sorted_namelist.sort()
		self.imgs = [name for name in sorted_namelist
			if name.startswith('PennFudanPed/PNGImages/')
				and name.endswith('.png')]
		self.masks = [name for name in sorted_namelist
			if name.startswith('PennFudanPed/PedMasks/')
				and name.endswith('.png')]

		self.transforms = T.Compose([
			T.ToDtype(torch.float, scale=True),
			T.ToPureTensor(),
		])

	def __getitem__(self, idx: int) -> tuple[torchvision.tv_tensors.Image, utils.Target]:
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=UserWarning)
			with self.root.open(self.imgs[idx]) as f:
				img = torch.frombuffer(f.read(), dtype=torch.uint8)
				img = torchvision.io.decode_image(img)
			with self.root.open(self.masks[idx]) as f:
				mask = torch.frombuffer(f.read(), dtype=torch.uint8)
				mask = torchvision.io.decode_image(mask)

		# Each color in the image represent an object
		obj_ids = torch.unique(mask)
		obj_ids = obj_ids[1:] # id=0 is always background
		num_objs = len(obj_ids)

		one_chan_per_obj_id = obj_ids[:, None, None]
		masks = (mask == one_chan_per_obj_id).to(torch.uint8)
		boxes = torchvision.ops.masks_to_boxes(masks)

		# all objects to be detected are pedestrian
		labels = torch.ones((num_objs, ), dtype=torch.int64)
		area = torchvision.ops.box_area(boxes)
		# all objects are not crowd
		iscrowd = torch.zeros((num_objs, ), dtype=torch.int64)

		target: utils.Target = {
			'boxes': boxes,
			'masks': masks,
			'labels': labels,
			'image_id': idx,
			'area': area,
			'iscrowd': iscrowd,
		}

		img, target = self.transforms(img, target)

		return img, target

	def __len__(self) -> int:
		return len(self.imgs)

def get_model() -> torch.nn.Module:
	# Mask R-CNN is based on Faster R-CNN

	# Trained on COCO
	model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

	num_classes = 2 # 0 = background, 1 = pedestrian
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn \
		.FastRCNNPredictor(in_features, num_classes)

	in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
	hidden_layers = 256
	model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn \
		.MaskRCNNPredictor(in_features_mask, hidden_layers, num_classes)

	return model


"""
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.771
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.946
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.690
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.787
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.815
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.815
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.792
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.828
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.737
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.989
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.907
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.385
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.756
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.717
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.791
"""
def coco_metrics(dts, gts):
	"""
	dts = detections
	gts = ground-truths
	"""



	#
	dts_prob = torchvision.ops.box_iou(dts['boxes'], targets['boxes'])
	torcheval.metrics.functional.multiclass_precision_recall_curve()

	IoUs = torchvision.ops.box_iou(dts['boxes'], targets['boxes'])
	matches = IoUs >= thresh
	correct = dts['labels'][matches] == gts['labels'][matches]
	# COCO calcola la precision reall curve


	pass

pennfudan = PennFudanDataset()
logger.info('Preprocessing dataset')
preprocessed_pennfudan = [pennfudan[i] for i in range(len(pennfudan))]
del pennfudan
random.shuffle(preprocessed_pennfudan)
dataset_train = utils.ListDataset(preprocessed_pennfudan[:-50], T.RandomHorizontalFlip(0.5))
dataset_test = utils.ListDataset(preprocessed_pennfudan[-50:], utils.Identity())

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Using device: {device}')

data_loader_train = torch.utils.data.DataLoader(
	dataset_train,
	batch_size=2,
	shuffle=True,
	collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=1,
    shuffle=False,
    collate_fn=utils.collate_fn
)

model = get_model()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 2

for epoch in range(num_epochs):
	model.train()

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1 / 1000
		warmup_iters = min(1000, len(data_loader_train) - 1)
		lr_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, start_factor=warmup_factor, total_iters=warmup_iters
		)

	for images, targets in data_loader_train:
		images = [image.to(device) for image in images]
		targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
				for k, v in target.items()}
			for target in targets]

		losses: dict = model(images, targets)
		loss = sum(losses.values())
		logger.info(f'epoch: {epoch} loss: {loss.item()}')

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if lr_scheduler: lr_scheduler.step()

	model.eval()

	with torch.no_grad():
		# outputs = dict[['boxes', 'labels', 'scores', 'masks']]

		test_outputs = []
		test_targets = []
		for images, targets in data_loader_test:
			images = [image.to(device) for image in images]
			outputs: list[dict] = model(images)
			outputs = [{k: (v.to('cpu') if isinstance(v, torch.Tensor) else v)
				for k, v in output.items()}
			for output in outputs]
			test_outputs.extend(outputs)
			test_targets.extend(targets)
		mAP = utils.mAP(test_outputs, test_targets, 1)
		print(f'mAP: {mAP}')

# TODO: preload/prefetch a few samples because the CPU graph is spiky

# https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
# https://pytorch.org/docs/stable/tensorboard.html

# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
