# Mixture of Volumetric Primitives -- Training and Evaluation

This repository contains code to train and render Mixture of Volumetric
Primitives (MVP) models.

If you use Mixture of Volumetric Primitives in your research, please cite:  
[Mixture of Volumetric Primitives for Efficient Neural Rendering](https://arxiv.org/abs/2103.01954)  
Stephen Lombardi, Tomas Simon, Gabriel Schwartz, Michael Zollhoefer, Yaser Sheikh, Jason Saragih  
ACM Transactions on Graphics (SIGGRAPH 2021) 40, 4. Article 59   

```
@article{Lombardi21,
author = {Lombardi, Stephen and Simon, Tomas and Schwartz, Gabriel and Zollhoefer, Michael and Sheikh, Yaser and Saragih, Jason},
title = {Mixture of Volumetric Primitives for Efficient Neural Rendering},
year = {2021},
issue_date = {August 2021},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {40},
number = {4},
issn = {0730-0301},
url = {https://doi.org/10.1145/3450626.3459863},
doi = {10.1145/3450626.3459863},
journal = {ACM Trans. Graph.},
month = {jul},
articleno = {59},
numpages = {13},
keywords = {neural rendering}
}
```

## Requirements

* Python (3.8+)
  * PyTorch
  * NumPy
  * SciPy
  * Pillow
  * OpenCV
* ffmpeg (in $PATH to render videos)
* CUDA 10 or higher

## Building

The repository contains two CUDA PyTorch extensions. To build, cd to each
directory and use `make`:
```
cd extensions/mvpraymarcher
make
cd -
cd extensions/utils
make
```

## How to Use

There are two main scripts in the root directory: train.py and render.py. The
scripts take a configuration file for the experiment that defines the dataset
used and the options for the model (e.g., the type of decoder that is used).

Download the latest release on Github to get the experiments directory.

To train the model:
```
python train.py experiments/dryice1/experiment1/config.py
```

To render a video of a trained model:
```
python render.py experiments/dryice1/experiment1/config.py
```

See ARCHITECTURE.md for more details.

## Training Data

See the latest Github release for data.

## Using your own Data

Implement your own Dataset class to return images and camera parameters. An
example is given in data.multiviewvideo. A dataset class will need to return
camera pose parameters, image data, and tracked mesh data.

## How to Extend

See ARCHITECTURE.md

## License

See the LICENSE file for details.
