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
        self.preproc = AttrDict(conf["PreProcessing"])
        self.exp = AttrDict(conf["Experiment"])
        self.vfv = AttrDict(conf["VirtualFusedVolume"])

        # Data folders
        self.data.train_tif_dir = f"{self.data.basepath}/Tiff_files/Train"
        self.data.train_gt_dir = f"{self.data.basepath}/GT_files/Train"
        self.data.test_tif_dir = f"{self.data.basepath}/Tiff_files/Test"
        self.data.test_gt_dir = f"{self.data.basepath}/GT_files/Test"

        # Experiments folders
        self.exp.train_data_dir = f"{self.exp.basepath}/{self.exp.name}/H5_files"
        self.exp.unet_checkpoint_dir = (
            f"{self.exp.basepath}/{self.exp.name}/UNet_checkpoints"
        )
        self.exp.unet_tensorboard_dir = (
            f"{self.exp.basepath}/{self.exp.name}/UNet_tensorboard"
        )
        self.exp.dog_logs_dir = f"{self.exp.basepath}/{self.exp.name}/DoG_logs"
        self.exp.dog_checkpoint_dir = (
            f"{self.exp.basepath}/{self.exp.name}/DoG_checkpoint"
        )
        self.exp.predictions_dir = (
            f"{self.exp.basepath}/{self.exp.name}/{self.data.name}_predictions"
        )

        # VirtualFusedVolume folders
        self.vfv.outdir = f"{self.vfv.outdir}/{self.vfv.name}"
        self.vfv.pred_outdir = f"{self.vfv.outdir}/stack_pred"

        # Additional data handling
        self.data.data_shape = np.array(self.data.data_shape)
        self.data.dim_resolution = np.array(self.data.dim_resolution)
        self.vfv.sub_shape = np.array(self.vfv.sub_shape)
        self.vfv.sub_overlap = np.array(self.vfv.sub_overlap)
        self.exp.input_shape = np.array(self.exp.input_shape)

        if self.preproc.transpose is not None:
            self.data.data_shape = np.array(
                [self.data.data_shape[i] for i in self.preproc.transpose]
            )
            self.data.dim_resolution = np.array(
                [self.data.dim_resolution[i] for i in self.preproc.transpose]
            )

        if self.preproc.downscale_factors is not None:
            self.data.data_shape = np.ceil(
                self.data.data_shape * self.preproc.downscale_factors
            ).astype(int)

            self.data.dim_resolution = (
                self.data.dim_resolution / self.preproc.downscale_factors
            )


if __name__ == "__main__":
    print(Configuration("config.yaml").data)
