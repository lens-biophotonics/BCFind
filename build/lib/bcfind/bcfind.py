import os
import inspect
import numpy as np
import pandas as pd
import functools as ft

from skimage import io
from zetastitcher import VirtualFusedVolume

from bcfind.config_manager import Configuration
from bcfind.utils import preprocessing, sigmoid
from bcfind.make_training_data import make_train_data
from bcfind.train import build_unet, unet_fit, dog_fit
from bcfind.bipartite_match import bipartite_match
from bcfind.vfv_pred import predict_vfv, make_cloud


def ARGUMENT_ERROR(arg_name):
    err = (
        f"{arg_name} is not defined: provide the {arg_name} argument "
        "or initialize the BCFind class with a configuration file."
    )
    return err


def selfy(func):
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        self = args[0]

        sig = inspect.signature(func)
        arguments = [param for param in sig.parameters if param != "self"]
        conf_attrs = [
            self,
            self.conf,
            self.conf.data,
            self.conf.vfv,
            self.conf.preproc,
            self.conf.exp,
        ]

        selfied_kwargs = {}
        for arg in arguments:
            for attr in conf_attrs:
                try:
                    selfied_kwargs[arg] = kwargs.get(arg, getattr(attr, arg))
                except AttributeError:
                    continue
        return func(*args, **selfied_kwargs)

    return wrapper


class BCFind:
    """
    Object class to access the main functionalities of the bcfind package.
    It can be initialized with a .yaml configuration file (recommended) or not.
    """

    def __init__(self, config_file=None):
        if config_file is not None:
            self.conf = Configuration(config_file)
            self.conf.data.x_file = f"{self.conf.exp.train_data_dir}/X_train.h5"
            self.conf.data.y_file = f"{self.conf.exp.train_data_dir}/Y_train.h5"
            sub_shape = self.conf.vfv.sub_shape
            if self.conf.preproc.transpose is not None:
                sub_shape = [
                    self.conf.vfv.sub_shape[i] for i in self.conf.preproc.transpose
                ]
            if self.conf.vfv.mask_path is not None:
                self.vfv_mask = io.imread(self.conf.vfv.mask_path)

            self.train_preprocessing_fun = ft.partial(
                preprocessing,
                transpose=self.conf.preproc.transpose,
                flip_axis=self.conf.preproc.flip_axis,
                clip_threshold=self.conf.preproc.clip_threshold,
                gamma_correction=self.conf.preproc.gamma_correction,
                downscale_factors=self.conf.preproc.downscale_factors,
                pad_output_shape=self.conf.data.data_shape,
            )
            self.vfv_preprocessing_fun = ft.partial(
                preprocessing,
                transpose=self.conf.preproc.transpose,
                flip_axis=None,
                clip_threshold=self.conf.preproc.clip_threshold,
                gamma_correction=self.conf.preproc.gamma_correction,
                downscale_factors=self.conf.preproc.downscale_factors,
                pad_output_shape=sub_shape,
            )
        else:
            self.conf = None

    @selfy
    def make_unet_train_data(
        self,
        train_tif_dir=None,
        train_gt_dir=None,
        h5_dir=None,
        train_preprocessing_fun=None,
        data_shape=None,
        dim_resolution=None,
        downscale_factors=None,
        x_file=None,
        y_file=None,
    ):
        """Method to prepare and store the training-set for fitting the UNet.
        It generates artificial targets from each .marker file and stores both the inputs and the targets in two .h5 files.
        The generated targets are images with gaussian spheres located at the coordinates given by the .marker files.
        If BCFind has been initialized with a configuration file all parameters are set as configuration, if no configuration file
        has been provided all parameters must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.


        :param train_tif_dir: path to the directory where the input images are stored. The images must be with .tif or .tiff extension.
        :param train_gt_dir: path to the directory where the .marker files of the true coordinates of the cells are stored.
            The file names must be equal to the corresponding tiff image (including the extension) with the additional .marker extension.
            (e.g. sub_0000_1450_0230.tif.marker)
        :param train_preprocessing_fun: function that takes a single image as input and apply the desired preprocessing.
        :param data_shape:
        :param dim_resolution: list or numpy.ndarray containing the resolution of the images for each dimension.
        :param downscale_factors: (optional) list or numpy.ndarray containing the possible downscaling factors to apply at the coordinates.
        :param x_file: path to the file where the input images will be stored. Must have the .h5 extension.
        :param y_file: path to the file where the target images will be stored. Must have the .h5 extension.

        :returns: the h5 dataset of input images, the h5 dataset of target images, a numpy.ndarray of the file names contained in the datasets.
            All of them are however already saved on disk at the location given.

        """
        make_train_data(
            train_tif_dir,
            train_gt_dir,
            h5_dir,
            data_shape,
            train_preprocessing_fun,
            dim_resolution,
            downscale_factors,
        )

    @selfy
    def build_default_unet(
        self,
        input_shape,
        n_filters=None,
        k_size=None,
        k_stride=None,
        learning_rate=None,
    ):
        """Method to build and compile U-Net model. Non-configurable defaults are the loss function (binary cross-entropy), the optimizer (Adam)
        and the metrics (mse and accuracy).
        input_shape argument has no default and must be given by the user.
        If BCFind has been initialized with a configuration file all others parameters are set as configuration, if no configuration file
        has been provided they must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param input_shape: shape of the input data.
        :param n_filters: (int) number of filters for first layer. Following layers will increase this number exponentially.
        :param k_size: (list or int) size of the convolutional kernel.
        :param k_stride: (list or int) stride of the convolution
        :param learning_rate: (float) learning rate for Adam optimizer.
        :returns: a tensorflow.keras.Model instance of the builded and compiled U-Net.

        """
        unet = build_unet(n_filters, k_size, k_stride, input_shape, learning_rate)
        return unet

    @selfy
    def unet_fit(
        self,
        x_file=None,
        y_file=None,
        batch_size=None,
        input_shape=None,
        unet_epochs=None,
        learning_rate=None,
        n_filters=None,
        k_size=None,
        k_stride=None,
        unet_checkpoint_dir=None,
        unet_tensorboard_dir=None,
        check_every=None,
    ):
        """Method for fitting U-Net model. The U-Net will be builded and compiled as default (see build_default_unet method).
        If BCFind has been initialized with a configuration file all parameters are set as configuration, if no configuration file
        has been provided all parameters must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param x_file: (string or path) path to the .h5 file where training inputs are stored.
        :param y_file: (string or path) path to the .h5 file where training targets are stored.
        :param batch_size: (int) size of the training batch.
        :param input_shape: (list) shape of the input data.
        :param unet_epochs: (int) number of epochs to perform.
        :param learning_rate: (float) learning rate for Adam optimizer.
        :param n_filters: (int) number of filters for first layer. Following layers will increase this number exponentially.
        :param k_size: (list or int) size of the convolutional kernel.
        :param k_stride: (list or int) stride of the convolution
        :param unet_checkpoint_dir: (string or path) path to the directory where a model.h5 file of final and intermediate model weights will be saved.
        :param unet_tensorboard_dir: (string or path) path to the tensorboard directory. See tensorboard documentation to learn how to visualize tensorboard callback.
        :param check_every: (int) epochs interval for sending updates to tensorboard callback.
        :returns: a tensorflow.keras.Model instance of the trained U-Net.

        """
        unet = unet_fit(
            x_file,
            y_file,
            batch_size,
            input_shape,
            unet_epochs,
            learning_rate,
            n_filters,
            k_size,
            k_stride,
            unet_checkpoint_dir,
            unet_tensorboard_dir,
            check_every,
        )
        return unet

    @selfy
    def dog_fit(
        self,
        unet,
        train_data_dir=None,
        train_gt_dir=None,
        dim_resolution=None,
        train_preprocessing_fun=None,
        dog_iterations=None,
        dog_logs_dir=None,
        dog_checkpoint_dir=None,
    ):
        """Method for fitting DoG blob detector.
        This method uses the ray library with TreeParzenEstimator (TPE) algorithm for best parameters searching.
        unet argument has no default and must be given by the user.
        If BCFind has been initialized with a configuration file all parameters are set as configuration, if no configuration file
        has been provided all parameters must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param unet: (tensorflow.keras.Model) fitted U-Net model.
        :param X: (iterable) iterable of preprocessed training images. They will be fistly processed with the U-Net model and then used
            for DoG training.
        :param Y: (iterable) iterable of arrays containg the true cell coordinates for each input image in X.
        :param dim_resolution: (list) resolution of the input images for each dimension.
        :param dog_iterations: (int) number of iteration that the TPE will perform.
        :param dog_logs_dir: (string or path) path to the directory where the logs will be saved. This folder can be visualized with tensorboard.
        :param dog_checkpoint_dir: (string or path) path to the directory where the optimal estimated parameters will be saved.
            Since the parameters are a dictionary, they will be saved as .json.
        :returns: a bcfind.BlobDoG instance with parameters set to the estimated ones.

        """
        dog = dog_fit(
            unet,
            train_data_dir,
            train_gt_dir,
            dim_resolution,
            dog_iterations,
            dog_logs_dir,
            dog_checkpoint_dir,
        )
        return dog

    @staticmethod
    def predict(unet, dog, img, outfile=None):
        """Method for predicting the soma locations from a single image.
        This is a static method, no init configurations are used. Hence the unet model, the dog blob detector and the image to
        predict must be provided.

        :param unet: (tensorflow.keras.Model) fitted U-Net model.
        :param dog: (bcfind.BlobDoG) the blob detector to be used.
        :param img: (numpy.ndarray) the image to use for prediction. It must be in [0, 1].
        :param outfile: (string or path) path to the file where the array of coordinates will
        be saved. If not specified results won't be saved.
        :returns: a numpy.ndarray of coordinates.

        """
        x = img[np.newaxis, ..., np.newaxis]
        emb = unet.predict(x)
        emb = sigmoid(np.reshape(emb, img.shape)) * 255
        centers = dog.predict(emb)
        if outfile is not None:
            np.save(outfile, centers)
        return centers

    @staticmethod
    def evaluate(pred, true, max_dist, dim_resolution=[1.0, 1.0, 1.0]):
        """Method for evaluating the predictions with respect to the true locations.
        This is a static method, no init configurations are used. Hence all arguments must be provided.

        :param pred: (numpy.ndarray) array of predicted coordinates (x, y, z)
        :param true: (numpy.ndarray) array of true coordinates (x, y, z)
        :param max_dist: (float) max distance between true and predicted center for considering the latter a possible true positive.
            It should be on the same scale of the image resolution.
        :param dim_resolution: (list of floats) resolution of the image for each dimension. Default to [1., 1., 1.].

        :returns: a pandas.DataFrame of cell coordinates each one labelled as TP, FP or FN

        """
        centers_eval = bipartite_match(true, pred, max_dist, dim_resolution)
        return centers_eval

    @selfy
    def evaluate_on_train_set(
        self,
        unet,
        dog,
        train_tif_dir=None,
        train_gt_dir=None,
        train_preprocessing_fun=None,
        max_dist=10,
        dim_resolution=None,
        save_res=True,
        predictions_dir=None,
    ):
        """Method for evaluating the fitted model on the whole training set.
        It returns a pandas.DataFrame with the counts of TP, FP and FN for each file in the given training directory.
        unet and dog arguments have no default and must be given by the user.
        If BCFind has been initialized with a configuration file all others parameters are set as configuration, if no configuration file
        has been provided they must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param unet: (tensorflow.keras.Model) fitted U-Net model.
        :param dog: (bcfind.BlobDoG) fitted DoG blob detector.
        :param train_tif_dir: (string or path) path to the directory where the .tiff files of training-set are stored.
        :param train_gt_dir: (string or path) path to the directory where the .marker files of the training-set are stored.
        :param train_preprocessing_fun: (function) function that takes a single image as argument and applies the desired preprocessing.
        :param max_dist: (float) max distance between true and predicted center for considering the latter a possible true positive.
            It should be on the same scale of the image resolution.
        :param dim_resolution: (list of floats) resolution of the image for each dimension. Default to [1., 1., 1.].
        :param save_res: (bool) wheter to save the predicted centers of each training file or not. Default to True.
        :param predictions_dir: (string or path) path to the directory where the predicted centers will be saved.

        :returns: a pandas.DataFrame with the counts of TP, FP and FN of each file.

        """
        f_names = [
            f
            for f in os.listdir(train_tif_dir)
            if f.endswith(".tif") or f.endswith(".tiff")
        ]

        train_res = []
        for f_name in f_names:
            img = io.imread(f"{train_tif_dir}/{f_name}")
            img = train_preprocessing_fun(img)
            out_f_name = f_name.split(".")[0]

            gt = pd.read_csv(open(f"{train_gt_dir}/{f_name}.marker", "r"))
            gt = gt[["#x", " y", " z"]].dropna(0)

            print(f"Predicting file {f_name}")
            pred = self.predict(
                unet,
                dog,
                img,
                save_res,
                f"{predictions_dir}/Pred_centers/{out_f_name}.npy",
            )

            print(f"Evaluating predictions on file {f_name}")
            res = self.evaluate(pred, gt, max_dist, dim_resolution)
            train_res.append(res)

        train_res = pd.concat(train_res)
        train_res.index = f_names
        return train_res

    @selfy
    def evaluate_on_test_set(
        self,
        unet,
        dog,
        test_tif_dir=None,
        test_gt_dir=None,
        train_preprocessing_fun=None,
        max_dist=10,
        dim_resolution=None,
        save_res=True,
        predictions_dir=None,
    ):
        """Method for evaluating the fitted model on the whole test-set.
        It returns a pandas.DataFrame with the counts of TP, FP and FN for each file in the given test-set directory.
        unet and dog arguments have no default and must be given by the user.
        If BCFind has been initialized with a configuration file all others parameters are set as configuration, if no configuration file
        has been provided they must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param unet: (tensorflow.keras.Model) fitted U-Net model.
        :param dog: (bcfind.BlobDoG) fitted DoG blob detector.
        :param train_tif_dir: (string or path) path to the directory where the .tiff files of test-set are stored.
        :param train_gt_dir: (string or path) path to the directory where the .marker files of the test-set are stored.
        :param train_preprocessing_fun: (function) function that takes a single image as argument and applies the desired preprocessing.
        :param max_dist: (float) max distance between true and predicted center for considering the latter a possible true positive.
            It should be on the same scale of the image resolution.
        :param dim_resolution: (list of floats) resolution of the image for each dimension. Default to [1., 1., 1.].
        :param save_res: (bool) wheter to save the predicted centers of each training file or not. Default to True.
        :param predictions_dir: (string or path) path to the directory where the predicted centers will be saved.

        :returns: a pandas.DataFrame with the counts of TP, FP and FN of each file.

        """
        f_names = [
            f
            for f in os.listdir(test_tif_dir)
            if f.endswith(".tif") or f.endswith(".tiff")
        ]

        test_res = []
        for f_name in f_names:
            img = io.imread(f"{test_tif_dir}/{f_name}")
            img = train_preprocessing_fun(img)
            out_f_name = f_name.split(".")[0]

            gt = pd.read_csv(open(f"{test_gt_dir}/{f_name}.marker", "r"))
            gt = gt[["#x", " y", " z"]].dropna(0)

            print(f"Predicting file {f_name}")
            pred = self.predict(
                unet,
                dog,
                img,
                save_res,
                f"{predictions_dir}/Pred_centers/{out_f_name}.npy",
            )

            print(f"Evaluating predictions on file {f_name}")
            res = self.evaluate(pred, gt, max_dist, dim_resolution)
            test_res.append(res)

        test_res = pd.concat(test_res)
        test_res.index = f_names
        return test_res

    @selfy
    def predict_vfv(
        self,
        unet,
        dog,
        vfv=None,
        pred_outdir=None,
        sub_shape=None,
        sub_overlap=None,
        vfv_preprocessing_fun=None,
        vfv_mask=None,
        sub_queue_size=5,
        emb_queue_size=5,
        dog_threads=10,
    ):
        """Method to predict cell locations of a whole zetastitcher.VirtualFusedVolume. The volume will be divided in substacks of dimensions sub_shape
        with an overlap between each other of sub_overlap.
        unet and dog arguments have no default and must be given by the user.
        If BCFind has been initialized with a configuration file all others parameters are set as configuration, if no configuration file
        has been provided they must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param unet:
        :param dog:
        :param vfv:
        :param pred_outdir:
        :param sub_shape:
        :param sub_overlap:
        :param vfv_preprocessing_fun:
        :param vfv_mask:
        :param sub_queue_size:
        :param emb_queue_size:
        :param dog_threads:
        :returns:

        """
        if vfv is None and self.conf is not None:
            vfv = VirtualFusedVolume(self.conf.vfv.config_file)
        if vfv is None and self.conf is None:
            err = ARGUMENT_ERROR("vfv")
            raise ValueError(err)

        predict_vfv(
            unet,
            dog,
            vfv,
            sub_shape,
            sub_overlap,
            pred_outdir,
            vfv_preprocessing_fun,
            vfv_mask,
            sub_queue_size,
            emb_queue_size,
            dog_threads,
        )

    @selfy
    def make_cloud(self, vfv_pred_dir=None, dim_resolution=None):
        """TODO describe function
        If BCFind has been initialized with a configuration file all parameters are set as configuration, if no configuration file
        has been provided all parameters must be given. Even if BCFind has been initialized with a configuration file you can
        overwrite any parameter specification by setting the argument as desired, no changes to the init configuration will be made.

        :param vfv_pred_dir:
        :param dim_resolution:
        :returns:

        """
        if vfv_pred_dir is None and self.conf is not None:
            vfv_pred_dir = self.conf.vfv.pred_outdir
        if vfv_pred_dir is None and self.conf is None:
            err = ARGUMENT_ERROR("vfv_pred_dir")
            raise ValueError(err)

        cloud_df = make_cloud(vfv_pred_dir, dim_resolution)
        return cloud_df
