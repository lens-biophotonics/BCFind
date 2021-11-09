# BCFind
BCfind (acronym for Brain Cell Finder) is a tool for brain cell localization from 3D microscopy images. A U-Net architecture transform the images into ideal ones where only the cells are visible and mostly in the form of gaussian spheres. Then Difference of Gaussian (DoG) algorithm for blob detection identify locates the spheres predicted by the U-Net.<br>

## Installation
#### If you pull the github repository
Build the wheel first with:<br>
```console
foo@bar:~$ cd /path/to/BCFind2.1
foo@bar:/path/to/BCFind2.1$ python setup.py bdist_wheel
foo@bar:/path/to/BCFind2.1$ pip install dist/bcfind-2.1.0-py3-none-any.whl
```
#### From NASone (recommended, more often updated)
Just pip install the wheel in `/mnt/NASone/curzio/Python/dist/bcfind-2.1.0-py3-none-any.whl`:
```console
user@lens_remote:~$ pip install /mnt/NASone/curzio/Python/dist/bcfind-2.1.0-py3-none-any.whl
```
You may need to install also the zetastitcher (other standard python packages are automatically installed):
```console
user@lens_remote:~$ pip install /mnt/NASone/curzio/Python/dist/zetastitcher-0.5.0-py3-none-any.whl
```
## Training  (for LENS users)
Training includes 4 steps:<br>

1. creation of artificial targets for the U-Net from files .marker of manually annotated coordinates of the cells;<br>
2. training the U-Net parameters;<br>
3. training the DOG parameters;<br>
4. evaluation of predictions on train and test sets.<br>
<br>

A file .yaml sets the experiment specifications: as the data path, the resolution of the images, some preprocessing steps, the experiment name and path where the results will be saved, the number of iterations to perform, etc.<br>
Here the layout:
```yaml
 Dataset:
    name           : 'my_data'
    basepath       : '/path/to/my/data/folders'
    data_shape     : !!python/tuple [160, 480, 480]
    dim_resolution : !!python/tuple [2., .65, .65]
    marker_columns : !!python/tuple [' z', ' y', '#x']

Experiment:
    name           : 'my_exp'
    basepath       : '/path/to/my/experiment/outputs'

PreProcessing:
    transpose         : !!python/tuple [2, 1, 0]
    flip_axis         : 1
    clip_threshold    : 8192
    gamma_correction  : 1.2
    downscale         : null
    normalization     : 'input'
    standardization   : null

DataAugmentation:
    augment      : true
    gamma        : !!python/tuple [0.5, 1.8]
    contrast     : null
    brightness   : !!python/tuple [-0.1, 0.1]
    zoom         : !!python/tuple [1.0, 1.3]
    gauss_filter : !!python/tuple [0.5, 2]
    noise        : !!python/tuple [0.05, 0.15]

UNet:
    val_fold       : '1/4'
    val_seed       : 123
    input_shape    : !!python/tuple [240, 240, 80]
    exclude_border : !!python/tuple [10, 10, 3]
    epochs         : 3000
    batch_size     : 10
    learning_rate  : 0.001
    n_filters      : 32
    k_size         : !!python/tuple [5, 5, 3]
    k_stride       : !!python/tuple [2, 2, 2]

DoG:
    iterations     : 40
    exclude_border : !!python/tuple [10, 10, 3]
    max_match_dist : 10
    n_cpu          : 1

VirtualFusedVolume:
    name           : 'vfv_name'
    config_file    : '/path/to/vfv/stitch.yml'
    sub_shape      : !!python/tuple [160, 480, 480]
    sub_overlap    : !!python/tuple [8, 20, 20]
    mask_path      : '/path/to/mask.tiff'
    mask_downscale : !!python/tuple [3, 3, 3]
    outdir         : '/path/to/vfv/predictions/folder'
```
<br>

`Dataset.basepath` must have the following sub-directories:<br>

- `GT_files/Train`
- `GT_files/Test`
- `Tiff_files/Train`
- `Tiff_files/Test`

While `Experiment.basepath` does not need any particular structure, it is the path to the main folder where all the outputs of the experiment will be saved: the .h5 files of preprocessed inputs and artificial targets, the U-Net and DoG checkpoints, the tensorboard folders for training steps visualization and the final predictions on the train and test sets.<br>
At the end it will contain the following folders:<br>

- `H5_files`
- `UNet_checkpoints`
- `UNet_tensorboard`
- `DoG_logs`
- `DoG_checkpoints`
- `{Dataset.name}_predictions/Pred_centers`
- `{Dataset.name}_predictions/train_eval.csv`
- `{Dataset.name}_predictions/test_eval.csv`
<br>
<br>
<br>

**TensorBoard** directories can be visualized in any browser by linking your machine with the remote server with a bridge. Type the following command on your local machine to open the bridge:
```console
user@local:~$ ssh -L 1111:localhost:2222 user@remote.server.it
```
Once your are logged in the remote server, type the following:
```console
user@remote:~$ tensorboard --logdir=/path/to/experiment/outputs/UNet_tensorboard --port=2222
```
Then open a window on your browser at the address `localhost:1111`, it will show you the time-series of the training iterations.<br>
<br>

#### Step 1 - Creation of artificial targets for U-Net
Run the following:<br>
```console
foo@bar:~$ bcfind-make-data /path/to/my/config.yaml
```
<br>
It will create an additional folder in `Experiment.basepath` named `H5_files` where the preprocessed inputs and artificial targets for the U-Net are stored.
<br>
<br>

#### Steps 2 and 3 - Traning both the U-Net and the DoG
Run the following:<br>
```console
foo@bar:~$ bcfind-train /path/to/my/config.yaml
```

The **main function** of `train` module will build the U-Net architecture, compile it with Adam as optimizer and start the training with ModelCheckpoint and TensorBoard callbacks. The callbacks will create the folders `{Experiment.basepath}/{Experiment.name}/UNet_tensorboard` and `{Experiment.basepath}/{Experiment.name}/UNet_checkpoints` to monitor the loss and other metrics and to periodically save the weights of the model.<br>
Once the U-Net training is completed, the DoG training will start. This firstly needs the U-Net prediction on the training-set, then a Tree-Parzen Estimator will search for the best parameters based on the F1 computed on the full data. During the DoG training a folders will be created: `{Experiment.basepath}/{Experiment.name}/DoG_checkpoints` for model parameters saving.<br>
<br>
<br>

#### Step 4 - Evaluate BCFind predictions
Run the following:
```console
foo@bar:~$ bcfind-evaluate /path/to/my/config.yaml
```
<br>

The **main function** of `evaluate_train` module iterates over the train and test data folders producing the predicted locations of each volume and two .csv files, `train_eval.csv` and `test_eval.csv`, with the counts of TP, FN and FP of each volume as found by the bipartite match algorithm. <br>
The predicted locations will be saved in `{Experiment.basepath}/{Experiment.name}/{Dataset.name}_predictions/Pred_centers` as .marker files, while the two .csv will be in `{Experiment.basepath}/{Experiment.name}/{Dataset.name}_predictions`.
