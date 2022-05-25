import yaml
import numpy as np

# Class to acces dictionary keys as if they were attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Configuration:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        yaml_key_to_attr = {
            'Dataset': 'data',
            'Experiment': 'exp',
            'PreProcessing': 'preproc',
            'DataAugmentation': 'data_aug',
            'UNet': 'unet',
            'DoG': 'dog',
            'VirtualFusedVolume': 'vfv',
        }
        for k, v in yaml_key_to_attr.items():
            try:
                setattr(self, v, AttrDict(conf[k]))
            except KeyError:
                pass
        
        # Dataset
        if self.data:
            self.data.shape = np.array(self.data.shape)
            self.data.dim_resolution = np.array(self.data.dim_resolution)

            self.data.train_tif_dir = f"{self.data.basepath}/Tiff_files/Train"
            self.data.train_gt_dir = f"{self.data.basepath}/GT_files/Train"
            self.data.test_tif_dir = f"{self.data.basepath}/Tiff_files/Test"
            self.data.test_gt_dir = f"{self.data.basepath}/GT_files/Test"

        if self.preproc.downscale is not None:
            self.data.shape = np.ceil(
                self.data.shape * self.preproc.downscale
            ).astype(int)

            self.data.dim_resolution = self.data.dim_resolution / self.preproc.downscale

        # Experiment
        self.exp.basepath = f"{self.exp.basepath}/{self.exp.name}"

        # PreProcessing
        if self.preproc.normalization is None:
            self.preproc.normalization = "none"

        if self.preproc.standardization is None:
            self.preproc.standardization = "none"

        # DataAugmentation
        if self.data_aug.augment:
            self.data_aug.op_args = {}
            self.data_aug.op_probs = []
            
            for k, v in self.data_aug.items():
                if k not in ['augment', 'op_args', 'op_probs']:
                    self.data_aug.op_args[k] = v
                    self.data_aug.op_probs.append(v['p'])
                    del self.data_aug.op_args[k]['p']
        else:
            self.data_aug.op_args = None
            self.data_aug.op_probs = None

        # UNet
        self.unet.input_shape = self.unet.input_shape

        self.unet.checkpoint_dir = (
            f"{self.exp.basepath}/UNet_checkpoint"
        )
        self.unet.tensorboard_dir = (
            f"{self.exp.basepath}/UNet_tensorboard"
        )

        # DoG
        self.dog.exclude_border = np.array(self.unet.exclude_border) + self.dog.exclude_border

        self.dog.logs_dir = f"{self.exp.basepath}/{self.exp.name}/DoG_logs"
        self.dog.checkpoint_dir = f"{self.exp.basepath}/{self.exp.name}/DoG_checkpoint"
        self.dog.predictions_dir = (
            f"{self.exp.basepath}/{self.exp.name}/{self.data.name}_predictions"
        )

        # VirtualFusedVolume
        self.vfv.sub_shape = np.array(self.vfv.sub_shape)
        self.vfv.sub_overlap = np.array(self.vfv.sub_overlap)

        self.vfv.outdir = f"{self.vfv.outdir}/{self.vfv.name}"
        self.vfv.pred_outdir = f"{self.vfv.outdir}/stack_pred"


if __name__ == "__main__":
    args = Configuration("/home/checcucci/Python/BCFind/config.yaml")
    print(args.data_aug.op_args)
