# Mixture of Volumetric Primitives -- Code Architecture

## File organization

There are five main entry points:
* train.py -- for training models or resuming training
* render.py -- for rendering models from training/validation cameras or novel
view fly-throughs/fly-arounds
* mse.py -- for computing quantitative evaluations & performance
* speedtest.py -- for measuring evaluation run time
* trainspeedtest.py -- for measuring training run time

The root directory contains several subdirectories:
```
data/ --- custom PyTorch Dataset classes for loading included data
extensions/ --- CUDA PyTorch extensions
experiments/ --- location of input data and training and evaluation output
models/ --- PyTorch modules (encoders, decoders, raymarchers, etc.)
utils/ --- miscellaneous utilities
```

## Configuration files

Each entry point takes as input a configuration file that instantiates the
model, dataset, the optimizer, and other things used for training or
evaluation. The configuration file must define the following:

```
class Train():
    batchsize=6
    def __init__(self, maxiter=2000000, **kwargs):
        self.maxiter = maxiter
        self.otherargs = kwargs
    def get_autoencoder(self, dataset):
        """Returns a PyTorch Module that accepts inputs and produces a dict
        of output values. One of those output values should be 'losses', another
        dict with each of the separate losses. See models.volumetric for an example."""
        return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self):
        """A dict that is passed to the autoencoder telling it what values
        to compute (e.g., irgbrec for the rgb image reconstruction)."""
        return []
    def get_ae_args(self):
        """Any non-data arguments passed to the autoencoder's forward method."""
        return dict(renderoptions={**get_renderoptions(), **self.otherargs})
    def get_dataset(self):
        """A Dataset class that returns data for the autoencoder"""
        return get_dataset(subsampletype="stratified")
    def get_optimizer(self, ae):
        """The optimizer used to train the autoencoder parameters."""
        import itertools
        import torch.optim
        lr = 0.002
        aeparams = itertools.chain(
            [{"params": x} for k, x in ae.encoder.named_parameters()],
            [{"params": x} for k, x in ae.decoder.named_parameters()],
            [{"params": x} for x in ae.bgmodel.parameters()],
            )
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"irgbmse": 1.0, "vertmse": 0.1, "kldiv": 0.001, "primvolsum": 0.01}
```
```
class Eval():
    def __init__(self, outfilename=None, outfilesuffix=None,
            cam=None, camdist=768., camperiod=512, camrevs=0.25,
            segments=["S01_She_always_jokes_about_too_much_garlic_in_his_food"],
            maxframes=-1,
            keyfilter=[],
            **kwargs):
        self.outfilename = outfilename
        self.outfilesuffix = outfilesuffix
        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if len(segments) == 0 else x in segments
        self.maxframes = maxframes
        self.keyfilter = keyfilter
        self.otherargs = kwargs
    def get_autoencoder(self, dataset): return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={
        **get_renderoptions(),
        **self.otherargs})
    def get_dataset(self):
        import data.utils
        import data.camrotate as cameralib
        if self.cam == "all":
            camerafilter = lambda x: x.startswith("40")
        elif self.cam == "holdout":
            camerafilter = lambda x: x in holdoutcams
        else:
            camerafilter = lambda x: x == self.cam
        dataset = get_dataset(camerafilter=camerafilter,
                segmentfilter=self.segmentfilter,
                keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                maxframes=self.maxframes,
                **{k: v for k, v in self.otherargs.items() if k in get_dataset.__code__.co_varnames},
                )
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset), camdist=self.camdist,
                    period=self.camperiod, revs=self.camrevs,
                    **{k: v for k, v in self.otherargs.items() if k in cameralib.Dataset.__init__.__code__.co_varnames})
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    def get_writer(self):
        """A class that contains a method 'batch' that receives a batch of
        input and output data and 'finalize'"""
        import utils.videowriter as writerlib
        if self.outfilename is None:
            outfilename = (
                    "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
                    (self.outfilesuffix if self.outfilesuffix is not None else "") +
                    ".mp4")
        else:
            outfilename = self.outfilename
        return writerlib.Writer(
            os.path.join(os.path.dirname(__file__), outfilename),
            keyfilter=self.keyfilter,
            colcorrect=[1.35, 1.16, 1.5],
            bgcolor=[255., 255., 255.],
            **{k: v for k, v in self.otherargs.items() if k in ["cmap", "cmapscale", "colorbar"]})
```

The training script is called by passing the path to the configuration file.
Some options will be forwarded to the configuration file, e.g.:
```
# train with default options
python train.py experiments/example/config.py

# use smaller iteration limit, and don't output progress images
python train.py experiments/example/config.py --maxiter 100000 --noprogress

# resume training a previous experiment
python train.py experiments/example/config.py --resume
```

Some examples of using the render script options. Note that quotes are needed
for string types, and quotes around those quotes are necessary for bash to read
the command.
```
# by default, Eval configuration will render the scene from a novel camera trajectory
python render.py experiments/example/config.py --keyfilter "['irgbrec']" --maxframes 256

# select a segment to render
python render.py experiments/example/config.py --keyfilter "['irgbrec']" --segments "['segment_name']"

# specify a camera to render from one camera perspective
python render.py experiments/example/config.py --keyfilter "['irgbrec', 'image']" --maxframes 256 --cam "'400264'" --outfilesuffix "'_vsgt'"

# squared error can be visualized with a color map
python render.py experiments/example/config.py --keyfilter "['irgbsqerr']" --maxframes 256 --cmap "'viridis'" --cmapscale 10. --colorbar True --outfilesuffix "'_sqerr'"

# some special visualization options are available
python render.py experiments/example/config.py --keyfilter "['irgbrec']" --maxframes 256 --colorprims True
```

The mse script can be used similarly, e.g.: (note that --cam must be specified)
```
python mse.py experiments/example/config.py --segments "['segment_name']" --cam "'all'"

python mse.py experiments/example/config.py --segments "'all'" --cam "'holdout'"

python mse.py experiments/example/config.py --segments "'all'" --cam "'400264'"
```

## Dataset

The Dataset instantiation in the config file usually looks like this:

```
def get_dataset(
        camerafilter=lambda x: x.startswith("40") and x not in holdoutcams,
        segmentfilter=lambda x: x not in holdoutseg,
        keyfilter=[],
        maxframes=-1,
        subsampletype=None,
        downsample=2,
        **kwargs):
    """
    Parameters
    -----
    camerafilter : Callable[[str], bool]
        Function to determine cameras to use in dataset (camerafilter(cam) ->
        True to use cam, False to not use cam).
    segmentfilter : Callable[[str], bool]
        Function to determine segments to use in dataset (segmentfilter(seg) ->
        True to use seg, False to not use seg).
    keyfilter : list
        List of data to load from dataset. See Dataset class (e.g.,
        data.multiviewvideo) for a list of valid keys.
    maxframes : int
        Maximum number of frames to load.
    subsampletype : Optional[str]
        Type of subsampling to perform on camera images. Note the PyTorch
        module does the actual subsampling, the dataset class returns an array
        of pixel coordinates.
    downsample : int
        Downsampling factor of input images.
    """
    return datamodel.Dataset(
        krtpath=krtpath,
        geomdir=geomdir,
        imagedir=imagedir,
        bgpath=bgpath,
        returnbg=False,
        avgtexsize=256,
        baseposepath=baseposepath,
        camerafilter=camerafilter,
        segmentfilter=segmentfilter,
        keyfilter=["bg", "camera", "modelmatrix", "modelmatrixinv", "pixelcoords", "image", "avgtex", "verts"] + keyfilter,
        maxframes=maxframes,
        subsampletype=subsampletype,
        subsamplesize=384,
        downsample=downsample,
        blacklevel=[3.8, 2.5, 4.0],
        maskbright=True,
        maskbrightbg=True)
```

The main dataset file is `data.multiviewvideo`. It loads a multi-view video
dataset and returns values according to keyfilter. See the Dataset class for
details.

## Autoencoder

The main autoencoder for MVP is `models.volumetric`. It takes as input the
dataset object, an encoder object, a decoder object, a raymarcher, an image
color calibration model, and a background model. Encoders are located in
`models.encoders.*`, decoders in `models.decoders.*`, etc. The volumetric
autoencoder sends input data through the encoder and volumetric decoder, ray
marches the volume, applies the color calibration, alpha mattes the background,
and outputs the reconstructed image and optionally any losses. Each of the sub-
modules (encoder, decoder, etc.) behaves similarly: they accept a list of
losses to compute and return outputs and losses.

## Encoder

Encoders are located in `models.encoders.*`. Encoders take input data (e.g.,
fixedcamimage, avgtex, verts) and produce an encoding (and optionally a loss,
e.g., VAE KL divergence).

## Decoders

Decoders are located in `models.decoders.*`. Each decoder takes as input the
encoding and viewing position to produce an output volume. That volume can be
parameterized in arbitrary ways. In MVP, it is represented as a set of
4-channel voxel grids and their rigid transforms. In Neural Volumes, the volume
is represented as a large 3D voxel grid containing RGBA values and a smaller
warp field voxel grid. In NeRF, it is the weights of the MLPs.

## Raymarchers

Raymarchers are located in `models.raymarcher.*`. The raymarcher takes the
volume representation from the decoder and raymarches through to produce an
image. Not all decoders and raymarchers can be mixed and matched; the
raymarcher will expect certain outputs from the decoder.

## Color calibration

In some cases, multi-view camera rigs may not be color calibrated accurately
(i.e., a picture from one camera may return slightly brighter or darker pixel
values than another, even when imaging the same scene from the same pose). To
address this the color calibration network learns a per-channel gain and bias
for each camera during training. This helps prevent the network from baking
differences in color owing to camera calibration into the view-dependent
volumetric representation.

## Background model

Volumetric models, being very flexible, will try to model every pixel in the
training images as best it can. Sometimes these pixels belong to objects far in
the background, well past the object of interest. To address this, we use a
background model.  The background models typically consist of two parts. One is
a static background image taken without placing the object in the camera rig.
The second is a learnable residual.  The second part is important with humans
because the lower body is outside the bounding volume of the scene but can be
seen in some cameras, causing the system to try to model those parts near the
cameras, causing haze in the reconstructed volume.

