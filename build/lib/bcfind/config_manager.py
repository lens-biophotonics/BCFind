import yaml
import numpy as np


# Class to acces dictionary keys as they were attributes
# (some properties and methods of dictionary class are lost)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Configuration:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        self.data = AttrDict(conf["Dataset"])
        self.exp = AttrDict(conf["Experiment"])
        self.preproc = AttrDict(conf["PreProcessing"])
        self.data_aug = AttrDict(conf["DataAugmentation"])
        self.unet = AttrDict(conf["UNet"])
        self.dog = AttrDict(conf["DoG"])
        self.vfv = AttrDict(conf["VirtualFusedVolume"])

        # Data configuration
        self.data.data_shape = np.array(self.data.data_shape)
        self.data.dim_resolution = np.array(self.data.dim_resolution)

        self.data.train_tif_dir = f"{self.data.basepath}/Tiff_files/Train"
        self.data.train_gt_dir = f"{self.data.basepath}/GT_files/Train"
        self.data.test_tif_dir = f"{self.data.basepath}/Tiff_files/Test"
        self.data.test_gt_dir = f"{self.data.basepath}/GT_files/Test"

        if self.preproc.transpose is not None:
            self.data.data_shape = np.array(
                [self.data.data_shape[i] for i in self.preproc.transpose]
            )
            self.data.dim_resolution = np.array(
                [self.data.dim_resolution[i] for i in self.preproc.transpose]
            )
            self.data.marker_columns = [
                self.data.marker_columns[i] for i in self.preproc.transpose
            ]

        if self.preproc.downscale is not None:
            self.data.data_shape = np.ceil(
                self.data.data_shape * self.preproc.downscale
            ).astype(int)

            self.data.dim_resolution = self.data.dim_resolution / self.preproc.downscale

        # Experiment configuration
        self.exp.h5_dir = f"{self.exp.basepath}/{self.exp.name}/H5_files"

        # PreProcessing configuration        
        if self.preproc.normalization is None:
            self.preproc.normalization = "none"

        if self.preproc.standardization is None:
            self.preproc.standardization = "none"

        # UNet configuration
        self.unet.input_shape = np.array(self.unet.input_shape)

        self.unet.checkpoint_dir = (
            f"{self.exp.basepath}/{self.exp.name}/UNet_checkpoint"
        )
        self.unet.tensorboard_dir = (
            f"{self.exp.basepath}/{self.exp.name}/UNet_tensorboard"
        )

        # DoG configuration
        self.dog.exclude_border = np.array(self.unet.exclude_border) + self.dog.exclude_border

        self.dog.logs_dir = f"{self.exp.basepath}/{self.exp.name}/DoG_logs"
        self.dog.checkpoint_dir = f"{self.exp.basepath}/{self.exp.name}/DoG_checkpoint"
        self.dog.predictions_dir = (
            f"{self.exp.basepath}/{self.exp.name}/{self.data.name}_predictions"
        )

        # VirtualFusedVolume folders
        self.vfv.sub_shape = np.array(self.vfv.sub_shape)
        self.vfv.sub_overlap = np.array(self.vfv.sub_overlap)

        self.vfv.outdir = f"{self.vfv.outdir}/{self.vfv.name}"
        self.vfv.pred_outdir = f"{self.vfv.outdir}/stack_pred"


if __name__ == "__main__":
    print(Configuration("config.yaml").data)
