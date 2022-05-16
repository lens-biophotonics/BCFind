import tensorflow as tf
import functools as ft
import numpy as np


def get_mask_fn(target_shape, border_size):
    framing_mask = np.zeros(target_shape)
    framing_mask[
        border_size[0] : target_shape[0] - border_size[0],
        border_size[1] : target_shape[1] - border_size[1],
        border_size[2] : target_shape[2] - border_size[2],
    ] = 1

    framing_mask = tf.convert_to_tensor(
        framing_mask.astype("bool"), dtype=tf.bool
    )
    return ft.partial(tf.boolean_mask, mask=framing_mask)


tf.function
def dice_loss(y_true, y_pred, from_logits=False):
    y_true = tf.cast(y_true, y_pred.dtype)

    if from_logits:
        y_pred = tf.sigmoid(y_pred)
    
    numerator = 2 * tf.reduce_sum(y_true * y_pred, tf.range(1, tf.rank(y_pred)))
    denominator = tf.reduce_sum(y_true + y_pred, tf.range(1, tf.rank(y_pred)))
    return 1 - numerator / denominator


class FramedFocalCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary focal crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """

    def __init__(self, border_size, target_shape, from_logits=False, gamma=2.0, alpha=None, add_dice=False):
        super(FramedFocalCrossentropy3D, self).__init__()

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.alpha = alpha
        self.gamma = gamma
        self.add_dice = add_dice

        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)

        if self.from_logits:
            y_probs = tf.sigmoid(y_pred)
        
        # class imbalance smoothing
        if self.alpha is not None:
            alpha = tf.cast(self.alpha, y_true.dtype)
            balance_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        else:
            balance_factor = 1
        
        # focal smoothing
        p_t = y_true * y_probs + (1 - y_true) * (1 - y_probs)

        gamma = tf.cast(self.gamma, y_true.dtype)
        focal_factor = tf.pow((1.0 - p_t), gamma)

        # weighted loss
        focal_loss = ce * focal_factor * balance_factor
        
        if self.add_dice:
            dice = dice_loss(y_true, y_pred, from_logits=self.from_logits)
            # broadcast dice to crossentropy (dice loss is computed per image)
            for _ in tf.range(tf.rank(ce) - 1):
                dice = tf.expand_dims(dice, 1)
            
            focal_loss = focal_loss + dice

        # framed loss
        loss = tf.map_fn(self.mask_fn, focal_loss)

        # loss reduction
        loss = tf.reduce_mean(loss)
        return loss
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'gamma': self.gamma,
            'alpha': self.alpha,
            'add_dice': self.add_dice,
        }
        return config


class FramedCrossentropy3D(tf.keras.losses.Loss):
    """
    Implementation of binary crossentropy loss for 3D images where the predictions
    at the borders are not included in the computation.
    """

    def __init__(self, border_size, target_shape, from_logits=False, alpha=None, add_dice=False):
        super(FramedCrossentropy3D, self).__init__()

        self.border_size = border_size
        self.target_shape = target_shape
        self.from_logits = from_logits
        self.alpha = alpha
        self.add_dice = add_dice

        self.mask_fn = get_mask_fn(target_shape, border_size)

    def call(self, y_true, y_pred):
        ce = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
        
        # class imbalance smoothing
        if self.alpha is not None:
            alpha = tf.cast(self.alpha, y_true.dtype)
            balance_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        else:
            balance_factor = 1

        loss = ce * balance_factor

        if self.add_dice:
            dice = dice_loss(y_true, y_pred, from_logits=self.from_logits)
            # broadcast dice to ce (dice loss is computed per image)
            for _ in tf.range(tf.rank(loss) - 1):
                dice = tf.expand_dims(dice, 1)

            loss = loss + dice

        # framed loss
        loss = tf.map_fn(self.mask_fn, loss)

        # loss reduction
        loss = tf.reduce_mean(loss)
        return loss
    
    def get_config(self):
        config = {
            'border_size': self.border_size,
            'target_shape': self.target_shape,
            'from_logits': self.from_logits,
            'add_dice': self.add_dice,
        }
        return config


if __name__ == '__main__':
    import os
    from pathlib import Path

    from bcfind.training_dataset import TrainingDataset


    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    data_dir = '/mnt/NASone/curzio/Data/I48/Gray_matter'
    exp_dir = '/mnt/NASone/curzio/Experiments/I48gray_AttentionUNet'

    data_shape = [50, 100, 100]
    dim_resolution = 3.3
    input_shape = [48, 100, 100]
    batch_size = 4
    augmentations = None # {
    #     'brightness': [-0.1, 0.1],
    #     'gamma': [0.5, 1.8],
    #     # 'zoom': [1, 1.5],
    #     'noise': [0.001, 0.05],
    #     'blur': [0.01, 1.5],
    #     'rotation': [-180, 180],
    # }
    augmentations_prob = 0.3

    n_filters = 32
    k_size = [3, 5, 5]
    k_stride = [2, 2, 2]

    lr = 0.001
    exclude_border = [2, 2, 2]
    epochs = 1500

    dog_iterations = 30
    max_match_dist = 10

    unet_checkpoints = f'{exp_dir}/UNet_checkpoints'
    tensorboard_dir = f'{exp_dir}/UNet_tensorboard'
    dog_checkpoints = f'{exp_dir}/DoG_checkpoints'
    pred_test_dir = f'{exp_dir}/Pred_dir'

    print('LOADING UNET DATA')
    marker_list = [f'{data_dir}/GT_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/GT_files/Train')]
    tiff_list = [f'{data_dir}/Tiff_files/Train/{fname}' for fname in os.listdir(f'{data_dir}/Tiff_files/Train')]

    ordered_tiff_list = []
    for f in marker_list:
        fname = Path(f).with_suffix('').name
        tiff_file = [f for f in map(lambda f: Path(f), tiff_list) if f.name == fname]
        ordered_tiff_list.append(str(tiff_file[0]))

    data = TrainingDataset(
        tiff_list=ordered_tiff_list[:], 
        marker_list=marker_list[:], 
        batch_size=batch_size, 
        dim_resolution=dim_resolution, 
        output_shape=input_shape, 
        augmentations=augmentations, 
        augmentations_prob=augmentations_prob)
    
    loss = FramedCrossentropy3D(exclude_border, input_shape, True)
    for x, y in data:
        print(loss(y, x, True))