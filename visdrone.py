# Retina net backbone's is resnet
#

import multiprocessing.pool
import os
import io
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

import utils
import memory_limit

import numpy
import torch
import torchvision
from torchvision.transforms import v2
import torcheval.metrics
import ignite.metrics

# https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(42)

memory_limit.set_memory_limit(60*1024*1024*1024)
# hold = bytearray(40*1024*1024*1024)

def show(img: torch.Tensor | numpy.ndarray) -> None:
	torchvision.transforms.functional.to_pil_image(img).show()
	input()

def loadtxt(path: str) -> dict:

	# TODO: make it more streamlined
	try:
		res =  numpy.loadtxt(path, int, '#', ',')
	except ValueError:
		logger.warning(f'File "{path}" is not formatted correctly trying to fix...')
		with open(path) as f:
			txt = f.read()
			txt = txt.replace(",\n", "\n")
		txt_io = io.StringIO(txt)
		res =  numpy.loadtxt(txt_io, int, '#', ',')
	if res.ndim == 1:
		res = numpy.expand_dims(res, 0)
	res = torch.tensor(res)

	res[:,:4] = torchvision.ops.box_convert(res[:,:4], "xywh", "xyxy")

	# Putting it in COCO format.
	id = path[path.rfind('\\')+1:-len('.txt')]
	target = {
		'boxes': res[:,:4],
		'labels': res[:,4].to(torch.int64),
		'image_id': id, # TODO: convert to int?
		'area': torchvision.ops.box_area(res[:,:4]),
		'iscrowd': False,
	}

	degenerate_boxes = target['boxes'][:, 2:] <= target['boxes'][:, :2]
	degenerate_boxes = degenerate_boxes[:,0] | degenerate_boxes[:,1]
	if degenerate_boxes.any():
		logger.error(f'degenerate boxes: {target["boxes"][degenerate_boxes]}')

	return target

class LoadImage(torch.nn.Module):
	def forward(self, pair) -> tuple[torch.Tensor, object]:
		return torchvision.io.read_image(pair[0]), pair[1]

def get_model() -> torch.nn.Module:
	backbone = torchvision.models.mobilenet_v2(
		weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
	).features
	# We add this attribute for RetinaNet
	backbone.out_channels = 1280

	if False:
		for param in backbone.parameters():
			param.requires_grad = False

	anchor_generator = torchvision.models.detection.anchor_utils.AnchorGenerator(
		sizes=((32, 64, 128, 256, 512),),
		aspect_ratios=((0.5, 1.0, 2.0),)
	)
	model = torchvision.models.detection.retinanet.RetinaNet(
		backbone,
		num_classes=12,
		anchor_generator=anchor_generator,
	)

	return model

def get_paths(dir_path: str) -> list[str]:
	fnames = os.listdir(dir_path)
	fnames.sort()
	fnames = [os.path.join(dir_path, fname) for fname in fnames]
	return fnames

train_data_paths = get_paths(r"D:\Datasets\VisDrone2019-DET\train\images")
train_labels_paths = get_paths(r"D:\Datasets\VisDrone2019-DET\train\annotations")

test_data_paths = get_paths(r"D:\Datasets\VisDrone2019-DET\test-dev\images")
test_labels_paths = get_paths(r"D:\Datasets\VisDrone2019-DET\test-dev\annotations")

with multiprocessing.pool.ThreadPool() as pool:
	# train_data = pool.map(torchvision.io.read_image, train_data_paths)
	train_labels = pool.map(loadtxt, train_labels_paths)
	# test_data = pool.map(torchvision.io.read_image, test_data_paths)
	test_labels = pool.map(loadtxt, test_labels_paths)

train_visdrone = list(zip(train_data_paths, train_labels))
dataset_train = utils.ListDataset(train_visdrone, v2.Compose([LoadImage(), v2.ToDtype(torch.float32, scale=True)]))
test_visdrone = list(zip(test_data_paths, test_labels))
dataset_test = utils.ListDataset(test_visdrone, v2.Compose([LoadImage(), v2.ToDtype(torch.float32, scale=True)]))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.info(f'Using device: {device}')

# TODO: preload stuff
data_loader_train = torch.utils.data.DataLoader(
	dataset_train,
	batch_size=2,
	shuffle=True,
	collate_fn=utils.collate_fn
)
data_loader_test = torch.utils.data.DataLoader(
	dataset_test,
	batch_size=8,
	collate_fn=utils.collate_fn
)

model = get_model()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.AdamW(params, lr=0.005, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 50

ma = utils.MovingAverage(10)

for epoch in range(num_epochs):
	model.train()

	lr_scheduler = None
	if epoch == 0:
		warmup_factor = 1 / 1000
		warmup_iters = min(1000, len(data_loader_train) - 1)
		lr_scheduler = torch.optim.lr_scheduler.LinearLR(
			optimizer, start_factor=warmup_factor, total_iters=warmup_iters
		)

	if True:
		logger.info('loading model')
		model.load_state_dict(torch.load('model.pt', weights_only=True))
	else:
		train_batch_num = len(data_loader_train)
		for batch_i, (images, targets) in enumerate(data_loader_train):
			images = [image.to(device) for image in images]
			targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
					for k, v in target.items()}
				for target in targets]

			try:
				losses: dict = model(images, targets)
			except AssertionError:
				# print()
				# TODO: clean data from degenerate boxes
				# logging.exception("exception in model forward")
				continue

			loss = sum(losses.values())
			print(f'epoch: {epoch+1}/{num_epochs} train batch: {batch_i+1}/{train_batch_num} '
				f'ma loss: {ma.update(loss.item())}', end='\r')

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if lr_scheduler: lr_scheduler.step()
		torch.save(model.state_dict(), 'model.pt')

	print()

	model.eval()

	with torch.no_grad():
		test_batch_num = len(data_loader_test)
		test_outputs = []
		test_targets = []
		for batch_i, (images, targets) in enumerate(data_loader_test):
			images = [image.to(device) for image in images]
			outputs = model(images)
			outputs = [{k: (v.to('cpu') if isinstance(v, torch.Tensor) else v)
				for k, v in output.items()}
			for output in outputs]
			test_outputs.extend(outputs)
			test_targets.extend(targets)
			print(f'epoch: {epoch+1}/{num_epochs} test batch: {batch_i+1}/{test_batch_num}', end='\r')

		mAP = utils.mAP(test_outputs, test_targets, 12)
		print(f'\nmAP: {mAP}')

	print()

raise SystemExit(0) ############################################################

if False:
	# Checking if the data is loaded correctly
	data_0_bboxed = torchvision.utils.draw_bounding_boxes(
		data[0],
		labels[0][:, :4]
	)
	show(data_0_bboxed)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
	weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
)
model = model.eval()

import inspect
# print(type(model))
print(inspect.getmro(type(model)))
# print(model)

if False:
	with torch.no_grad():
		outputs = model([data[0]/255])

	output_viz = torchvision.utils.draw_bounding_boxes(
		data[0],
		outputs[0]['boxes'],
		[str(n.item()) for n in outputs[0]['labels']]
	)
	print(outputs)
	show(output_viz)
