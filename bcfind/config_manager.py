import yaml
import numpy as np
import tensorflow as tf

# Class to acces dictionary keys as if they were attributes
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class TrainConfiguration:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        yaml_key_to_attr = {
            'Dataset': 'data',
            'Experiment': 'exp',
            'DataAugmentation': 'data_aug',
            'PreProcessing': 'preproc',
            'UNet': 'unet',
            'DoG': 'dog',
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

        # Experiment
        self.exp.basepath = f"{self.exp.basepath}/{self.exp.name}"

        # DataAugmentation
        if self.data_aug.augment:
            self.data_aug.op_args = {}
            self.data_aug.op_probs = []
            
            for key, value in self.data_aug.items():
                if key not in ['augment', 'op_args', 'op_probs']:
                    self.data_aug.op_args[key] = value
                    self.data_aug.op_probs.append(value['p'])
                    del self.data_aug.op_args[key]['p']
        else:
            self.data_aug.op_args = None
            self.data_aug.op_probs = None

        # UNet
        self.unet.checkpoint_dir = f"{self.exp.basepath}/UNet_checkpoints"
        self.unet.tensorboard_dir = f"{self.exp.basepath}/UNet_tensorboard"
        self.unet.exclude_border = self.data.cell_radius

        if isinstance(self.unet.regularizer, dict):
            if len(self.unet.regularizer) > 1:
                raise ValueError(f'''
                Dictionary specification for UNet regularizer must have max length = 1. 
                Got {self.unet.regularizer} of length {len(self.unet.regularizer)}
                ''')

            key, value = list(self.unet.regularizer.items())[0]
            if key == 'l1':
                self.unet.regularizer = tf.keras.regularizers.L1(value)
            elif key == 'l2':
                self.unet.regularizer = tf.keras.regularizers.L2(value)
            else:
                raise ValueError(f'''
                Dictionary keys for UNet regularizer must be one of ['l1', 'l2'].
                Got instead {key}
                ''')

        # DoG
        self.dog.exclude_border = np.array(self.data.cell_radius) * 2

        self.dog.checkpoint_dir = f"{self.exp.basepath}/DoG_checkpoints"
        self.dog.predictions_dir = f"{self.exp.basepath}/{self.data.name}_predictions"


class VFVConfiguration:
    def __init__(self, yaml_file):
        with open(yaml_file) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)

        yaml_key_to_attr = {
            'Experiment': 'exp',
            'VirtualFusedVolume': 'vfv',
            'PreProcessing': 'preproc'
        }
        for k, v in yaml_key_to_attr.items():
            try:
                setattr(self, v, AttrDict(conf[k]))
            except KeyError:
                pass

        # Experiment
        self.exp.basepath = f"{self.exp.basepath}/{self.exp.name}"

        # UNet
        self.unet = AttrDict()
        self.unet.checkpoint_dir = f"{self.exp.basepath}/UNet_checkpoints"
        self.unet.tensorboard_dir = f"{self.exp.basepath}/UNet_tensorboard"

        # DoG
        self.dog = AttrDict()
        self.dog.checkpoint_dir = f"{self.exp.basepath}/DoG_checkpoints"

        # VirtualFusedVolume
        self.vfv.patch_shape = np.array(self.vfv.patch_shape)
        self.vfv.patch_overlap = np.array(self.vfv.patch_overlap)

        self.vfv.outdir = f"{self.vfv.outdir}/{self.vfv.name}"
        self.vfv.pred_outdir = f"{self.vfv.outdir}/stack_pred"


if __name__ == "__main__":
    args = VFVConfiguration("/home/curzio/Python/Projects/BCFind/vfv_config.yaml")
    print(args.preproc.center not in ['none', None])
