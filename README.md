# BCFind

BCfind (acronym for Brain Cell Finder) is a tool for brain cell localization from 3D microscopy images. A U-Net architecture transform the images into ideal ones where only the cells are visible and mostly in the form of gaussian spheres. Then Difference of Gaussian (DoG) algorithm is applied as blob detector to detect the cell coordinates from the U-Net predictions.

## Installation

First build the wheel:

```console
python setup.py bdist_wheel
```

Then install the wheel with pip:

```console
pip install dist/bcfind-2.1.0-py3-none-any.whl
```

## Build docker image

Once the wheel has been built (no need for complete installation):

```console
docker build -t bcfind:your_tag .
```

Run the docker container in interactive mode:

```console
docker run -it --rm --gpus all -v /volume/to/mount/:/home/ bcfind:your_tag
```

## (LENS researchers only) Push the Docker image to the ATLANTE LENS registry

Retag the image to the registry on `atlante.lens.unifi.it`:

```console
docker image tag bcfind:your_tag atlante.lens.unifi.it:5000/bcfind:your_tag
```

To push the image a VPN connection with atlante is needed, so on a new terminal run the following:

```console
sshuttle -r user@atlante.lens.unifi.it atlante.lens.unifi.it:5000
```

Then you can push the image to the atlante registry:

```console
docker push atlante.lens.unifi.it:5000/bcfind:your_tag
```

## Training

A file .yaml sets the experiment configurations: as the data path, the resolution of the images, some preprocessing steps, the experiment name and path where the results will be saved, the number of iterations to perform, etc.
Here the layout:

```yaml
Dataset:
    name: "my_data"
    basepath: "/path/to/data/folders"
    shape: [160, 480, 480]
    dim_resolution: [2., .65, .65]
    cell_radius: [3, 9, 9] # in pixels, for border exclusion

Experiment:
    name: "my_exp"
    basepath: "/path/to/experiment/outputs"

UNet:
    model: "unet" # can be one of ['unet', 'res-unet', 'se-unet', 'eca-unet', 'attention-unet', 'moe-unet']
    input_shape: [80, 240, 240]
    epochs: 4000
    batch_size: 10
    learning_rate: 0.001
    n_blocks: 4
    n_filters: 32
    k_size: !!python/tuple [3, 5, 5]
    k_stride: !!python/tuple [2, 2, 2]
    dropout: null
    regularizer: null # { "l2": 0.0001 }

    squeeze_factor: 2 # only used if model == 'se-unet'
    moe_n_experts: 5 # only used if model == 'moe-unet'
    moe_top_k_experts: null # only used if model == 'moe-unet'
    moe_noise: true # only used if model == 'moe-unet'
    moe_balance_loss: "load" # only used if model == 'moe-unet'; can be 'load' or 'importance'

DoG:
    iterations: 40
    max_match_dist: 10 # same scale as given by dim_resolution
    n_cpu: 5

PreProcessing:
    clip: "bit" # can be one of ['constant', 'bit', 'quantile', 'auto', null]
    clip_value: 15
    center: null # can be one of ['constant', 'min', 'mean', null]
    center_value: null
    scale: "bit" # can be one of ['constant', 'bit', 'max', 'std', null]
    scale_value: 15

DataAugmentation:
    augment: false

    gamma:
        param_range: [0.9, 1.1]
        p: 0.3
    # contrast:
    #     param_range     : [1., 3.]
    brightness:
        param_range: [-0.06, 0.06]
        p: 0.3
    # zoom:
    #     param_range     : [1.0, 1.1]
    #     order           : 1
    #     p               : 0.3
    blur:
        param_range: [0., 0.3]
        p: 0.3
    noise:
        param_range: [0., 0.03]
        p: 0.3
    # rotation:
    #     param_range     : [0., 270.]
    #     axes            : [-2, -1]
    #     p               : 0.3
    flip:
        axes: [-2]
        p: 0.3
```

`Dataset.basepath` must have the following sub-directories:

-   `GT_files/Train`
-   `GT_files/Test`
-   `Tiff_files/Train`
-   `Tiff_files/Test`

While `Experiment.basepath` does not need any particular structure, it is the path to the main folder where all the outputs of the experiment will be saved: the U-Net and DoG checkpoints, the tensorboard folders and the final predictions on the train and test sets.
It will contain the following folders:

-   `UNet_checkpoints`
-   `UNet_tensorboard`
-   `DoG_checkpoints`
-   `Train_pred_lmdb`
-   `Test_pred_lmdb`

### Train BCFind

Once inside the docker container or directly on the machine where the bcfind package has been installed, you can start the training with:

```console
bcfind-train /path/to/train_config.yaml
```

The above command takes one mandatory argument that is the configuration file and other four optional arguments:

-   `--gpu`: (int) select the gpu to use. Default to -1;
-   `--lmdb`: (store true) if your dataset do not fit into memory an lmdb database is created before training to save memory usage. Default to False;
-   `--only-dog`: (store true) train only the Difference of Gaussian from latest UNet checkpoint. Skip the UNet training. Default to False;
-   `--only-test`: (store true) skip the whole training and evaluate only on test-set. Default to False;
-   `--val-from-test`: (store true) split the test-set to obtain a validation-set (1/3). UNet weights will be saved only if the validation loss improves. The DoG will be trained on the validation-set. Default to False.

### Evaluate BCFind on test-set

(Deprecated) The above command (bcfind-train) also returns test-set results.

```console
bcfind-test /path/to/train_config.yaml
```

## Tensorboard

TensorBoard directories can be visualized in any browser by linking your local machine to the remote server with a bridge

```console
user@local:~$ ssh -L 1111:localhost:2222 user@remote.server.it
```

Once your are logged into the remote server, type the following:

```console
user@remote:~$ tensorboard --logdir=/path/to/experiment/outputs/UNet_tensorboard --port=2222
```

Then open a window on your browser at the address `localhost:1111`, it will show the time-series of the training iterations.

## VirtualFusedVolume predictions

If the volume is too large to fit into memory, only volumes stitched with zetastitcher can be handled.

The following .yaml file sets the configurations for prediction:

```yaml
Experiment:
    name: "my_exp"
    basepath: "/path/to/experiment/outputs"

VirtualFusedVolume:
    name: "vfv_name"
    config_file: "/path/to/vfv/stitch.yml" # can be also a .tiff file (if it fits into memory...)
    dim_resolution: [2., .65, .65]
    patch_shape: [160, 480, 480]
    patch_overlap: [12, 36, 36] # NOTE: it should be at least two times the diameter of the largest cell
    mask_path: "/path/to/mask.tiff" # can be any resolution, any shape difference with the VFV will be accounted for
    outdir: "/path/to/vfv/predictions/folder"

# Attn! Should be identical to the training phase! Be aware of any differences: check training config file!
PreProcessing:
    clip: "bit" # can be one of ['constant', 'bit', 'quantile', 'auto', null]
    clip_value: 14
    center: null # can be one of ['constant', 'min', 'mean', null]
    center_value: null
    scale: "bit" # can be one of ['constant', 'bit', 'max', 'std', 'none', null]
    scale_value: 14
```

To start the prediction:

```console
bcfind-vfv-pred /path/to/vfv_config.yaml
```
