import argparse
import pathlib
import sys

import numpy
import onnxruntime
import tvm
import torch
import torchvision

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluates a list of ONNX and TVM models on a given dataset.",
    )
    parser.add_argument(
        "-m", "--models", nargs="+", help="The list of models to evaluate.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-d", "--dataset", help="The dataset to use for the evaluation.", required=True, type=pathlib.Path
    )
    parser.add_argument(
        "-s", "--subset", help="Size of the random subset to use for evaluation.", default=0, type=int
    )
    # TODO: batch_size
    # TODO: num_workers
    # TODO: metric e.g. top-k

    args = parser.parse_args()

    NUM_WORKERS = 4
    BATCH_SIZE = min(max(1, args.subset), 64)

    tvm_dev = tvm.device("cpu")
    models = []
    for model_path in args.models:
        suffix = model_path.suffix
        if suffix == ".onnx":
            session = onnxruntime.InferenceSession(model_path)
            models.append(session)
        elif suffix == ".dll" or suffix == ".so" or suffix == ".tar":
            ex = tvm.runtime.load_module(model_path)
            vm = tvm.relax.VirtualMachine(ex, tvm_dev)
            models.append(vm)
        else:
            raise RuntimeError("Unsupported extension for %s" % model_path)

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

    models_correct = [0] * len(models)
    total_samples = 0
    for images, labels in dataloader:
        images_np = images.numpy()
        labels_np = labels.numpy()

        for i, model in enumerate(models):
            if isinstance(model, onnxruntime.InferenceSession):
                input_name = model.get_inputs()[0].name
                outputs = model.run(None, {input_name: images_np})[0]
            elif isinstance(model, tvm.relax.VirtualMachine):
                outputs = model["main"](tvm.nd.array(images_np, tvm_dev)).numpy()
            else:
                assert False

            predictions = numpy.argmax(outputs, axis=1)

            models_correct[i] += (predictions == labels_np).sum()
        total_samples += labels_np.shape[0]
        print(".", end='')
        sys.stdout.flush()
    print()

    for i, model_path in enumerate(args.models):
        top1_accuracy = models_correct[i] / total_samples
        print("Top-1 Accuracy %s: %f" % (model_path, top1_accuracy))

if __name__ == "__main__":
    main()
