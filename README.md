# Resources

## Visdrone

https://aiskyeye.com/home/ this seems to be the home of the VisDrone
challenge/dataset.
https://github.com/VisDrone/VisDrone-Dataset
https://github.com/VisDrone/VisDrone2018-DET-toolkit constains the description
of the labels.
https://datasetninja.com/vis-drone-2019-det a good exploration of the dataset.
https://paperswithcode.com/dataset/visdrone
https://www.kaggle.com/datasets/kushagrapandya/visdrone-dataset
https://docs.ultralytics.com/it/datasets/detect/visdrone/ a tutorial


A list of relevant models
https://github.com/facebookresearch/Detectron#introduction

## Metrics

[A Survey on Performance Metrics for Object-Detection
Algorithms](https://ieeexplore.ieee.org/document/9145130)
[Comparative Analysis of Object Detection Metrics with a Companion Open-Source
Toolkit](https://doi.org/10.3390/electronics10030279)

# Outdoor Python

After doing the build with `build.bat` in `PCbuild`

```
.\python.exe -m ensurepip
.\python.exe -m pip download -r requirements.txt -d .\wheelhouse
.\python.exe -m pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 -d .\wheelhouse
REM On the offline PC
.\python.exe -m venv myvenv
pip install --no-index --find-links .\wheelhouse -r requirements.txt
pip install --no-index --find-links .\wheelhouse torch torchvision
```

note that somehow the default sys.path is set to the module in cpython's source
tree. To build a windows installer you have to go to `Tools\msi\build.bat`