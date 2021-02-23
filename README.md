# BCFind
BCfind (acronym for Brain Cell Finder) is a tool for brain cell localization from 3D microscopy images. A U-Net architecture transform the images into ideal ones where only the cells appear and mostly in the form of gaussian spheres. Then the Difference of Gaussian (DoG) algorithm fro blob detection is used to identify the location of the spheres enhanced by the U-Net.<br>

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
    name           : 'my_data_name'
    basepath       : '/my/path/to/data/folders'
    data_shape     : !!python/tuple [480, 480, 160]
    dim_resolution : !!python/tuple [.65, .65, 2.]

Preprocessing:
    transpose         : !!python/tuple [2, 1, 0]
    flip_axis         : null
    clip_threshold    : 4096
    gamma_correction  : 1.2
    downscale_factors : null

Experiment:
    name             : 'my_exp_name'
    basepath         : '/my/path/for/saving/experiment/outputs'
    input_shape      : !!python/tuple [160, 160, 80]
    unet_epochs      : 3000
    dog_iterations   : 30
    batch_size       : 10
    check_every      : 1
    learning_rate    : 0.001
    n_filters        : 32
    e_size           : !!python/tuple [5, 5, 3]
    e_stride         : !!python/tuple [2, 2, 2]
    d_size           : !!python/tuple [5, 5, 3]
    d_stride         : !!python/tuple [2, 2, 2]

```
<br>

`Dataset.basepath` must have the following sub-directories:<br>

- `GT_files/Train`
- `GT_files/Test`
- `Tiff_files/Train`
- `Tiff_files/Test`

While `Experiment.basepath` does not need any particular structure, it is the path to the main folder where all the outputs of the experiment will be saved: as the U-Net and DoG checkpoints, the tensorboard folders for training steps visualization and the final predictions on the train and test sets.<br>
At the end it will contain the following folders:<br>

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

**TensorBoard** directories can be visualized in any browser by linking your machine with the remote server you are using with a bridge. Type the following command on your terminal to open the bridge:
```console
user@local:~$ ssh -L 1111:localhost:2222 user@remote.server.it
```
Once your are logged in the remote server you only have to type the following:
```console
user@remote:~$ tensorboard --logdir=/my/path/for/saving/experiment/outputs/UNet_tensorboard --port=2222
```
Opening a window on your browser at the link `localhost:1111` will show you the time-series of the training iterations.<br>
<br>

If the **Liquid server** will be used, first activate `/home/checcucci/tf20` environment:
```console
user@liquid:~ $ source /home/checcucci/tf20/bin/activate
```
then add libcupti.so.10.1 library to the LD_LIBRARY_PATH:
```console
user@liquid:~ $ export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64/
```
<br>

#### Step 1 - Creation of artificial targets for U-Net
**Requirements**:<br>
`os, h5py, argparse, numpy, pandas, scikit-image, scipy, colorama`
<br>
<br>

Running the following command will create an additional folder in `Dataset.basepath` named `H5_files` where the preprocessed inputs and artificial targets for the U-Net will be stored:
```console
user@remote:/path/to/BCFind2.1$ python make_training_data.py /path/to/my/config/my_config.yaml
```
<br>

The **main function** of the module `make_training_data` iterates over `{Dataset.basepath}/Tiff_files/Train` and `{Dataset.basepath}/GT_files/Train` to fill two .h5 files, `X_train.h5` and `Y_train.h5`, with the preprocessed substacks and the generated targets respectively.<br>
<br>
<br>

#### Steps 2 and 3 - Traning both the U-Net and the DoG
**Requirements**:<br>
`os, json, shutil, argparse, numpy, pandas, multiprocessing, scikit-image, tensorflow==2.3, ray[tune], hyperopt`
<br>
<br>

Run the following to start the training on the remote server you are logged in:<br>
```console
user@remote:/path/to/BCFind2.1$ python train.py /path/to/my/config/my_config.yaml
```

To run the training on the Cluster, log into atlante.lens.unifi.it and copy the folder `/mnt/NASone/curzio/Cluster_jobs/` into your NASone user space. Go to your `Cluster_jobs/Train_bcfind` directory and remove all .yaml, err., out. and .log files in there. Copy your `my_config.yaml` into that folder and run the following:
```console
user@atlante:/mnt/NASone/user/Cluster_jobs/Train_bcfind $ condor_submit train.job
```
all files .yaml in `Train_bcfind` will start a training process with that configurations on separate machines of the cluster.<br>
<br>

The **main function** of `train` module will build the U-Net architecture, compile it with Adam as optimizer and start the training with ModelCheckpoint and TensorBoard callbacks. The callbacks will create the folders `{Experiment.basepath}/{Experiment.name}/UNet_tensorboard` and `{Experiment.basepath}/{Experiment.name}/UNet_checkpoints` to monitor the loss and other metrics and to periodically save the weights of the model.<br>
Once the U-Net training is completed, the DoG training will start. This firstly needs the train-set predictions of the U-Net, then a Tree-Parzen Estimator will search for the best parameters based on the F1 computed on the full train-set. During the DoG training two folders will be created: `{Experiment.basepath}/{Experiment.name}/DoG_logs` and `{Experiment.basepath}/{Experiment.name}/DoG_checkpoints` to monitor the training steps and save the model parameters, respectively.<br>
<br>
<br>

#### Step 4 - Evaluate BCFind predictions
**Requirements**:<br>
`os, json, argparse, numpy, pandas, tensorflow==2.3`
<br>
<br>

Run the following:
```console
user@remote:/path/to/BCFind2.1$ python evaluate_train.py /path/to/my/config/my_config.yaml
```
<br>

The **main function** of `evaluate_train` module iterates over the train and test data folders producing the predicted locations of each volume and two .csv files, `train_eval.csv` and `test_eval.csv`, with the counts of TP, FN and FP of each volume as found by the bipartite match algorithm. <br>
The predicted locations will be saved in `{Experiment.basepath}/{Experiment.name}/{Dataset.name}_predictions/Pred_centers` as .npy files, while the two .csv will be in `{Experiment.basepath}/{Experiment.name}/{Dataset.name}_predictions`.
