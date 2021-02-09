import yaml
import numpy as np


# Class to acces dictionary keys as they were attributes
# (many properties and methods of dictionary class are lost)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Configuration:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        self.data = AttrDict(conf["Dataset"])
        self.preproc = AttrDict(conf["Preprocessing"])
        self.exp = AttrDict(conf["Experiment"])

        self.data.train_tif_dir = f"{self.data.basepath}/Tiff_files/Train"
        self.data.train_gt_dir = f"{self.data.basepath}/GT_files/Train"

        self.data.test_tif_dir = f"{self.data.basepath}/Tiff_files/Test"
        self.data.test_gt_dir = f"{self.data.basepath}/GT_files/Test"

        self.data.files_h5_dir = f"{self.data.basepath}/H5_files"

        self.data.data_shape = np.array(self.data.data_shape)
        self.data.dim_resolution = np.array(self.data.dim_resolution)
        self.exp.input_shape = np.array(self.exp.input_shape)

        if self.preproc.downscale_factors is not None:
            self.data.data_shape = np.ceil(
                self.data.data_shape * self.preproc.downscale_factors
            ).astype(int)

            self.data.dim_resolution = (
                self.data.dim_resoltion / self.preproc.downscale_factors
            )


if __name__ == "__main__":
    print(Configuration("config.yaml").data)
