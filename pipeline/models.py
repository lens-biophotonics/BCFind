import numpy as np
import os
import tensorflow as tf
import json

from bcfind.localizers import BlobDoG



def get_unet(unet_dir: str) -> tf.keras.Model:
    try:
      """
      Load a BCFind trained U-Net model.

      Parameters:
        unet_dir: Directory containing 'model.tf' saved model.

      Returns:
        A BCFind TensorFlow U-Net model ready for inference.
      """
      # GPU setup
      gpus = tf.config.list_physical_devices('GPU')
      if not gpus:
          raise RuntimeError("$ No GPU devices found for U-Net inference.")
      # Use the first GPU and allow memory growth
      tf.config.set_visible_devices(gpus[0], 'GPU')
      tf.config.experimental.set_memory_growth(gpus[0], True)

      # Load and build model
      model_path = os.path.join(unet_dir, 'model.tf')
      unet = tf.keras.models.load_model(model_path, compile=False)
      # Build with dynamic spatial dimensions and channel=1
      unet.build(input_shape=(None, None, None, None, 1))

      return unet
    except Exception as exc:
        raise Exception(
            f"& Error occured when getting the U-Net model.\n{exc}"
        )


def get_dog(
        dog_dir,
        patch_shape,
        patch_overlap,
        dim_resolution
    ) -> BlobDoG:
    """
    Initialize the DoG blob detector.

    Parameters:
      dog_dir       : Directory containing 'BlobDoG_parameters.json'
      path_shape    : Spatial dimensions of the input volume
      dim_resolution: Physical size per voxel
      patch_overlap : Overlap size for patch-based processing

    Returns:
      Configured BlobDoG instance for neuron detection.
    """
    try:
      # Instantiate with half-overlap to ensure continuity
      half_overlap = np.array(patch_overlap) // 2
      dog = BlobDoG(len(patch_shape), dim_resolution, half_overlap)

      # Load and set parameters
      par_path = os.path.join(dog_dir, 'BlobDoG_parameters.json')
      with open(par_path, 'r') as f:
          params = json.load(f)
      dog.set_parameters(params)

      return dog
    except Exception as exc:
      raise Exception(
         f"& Error occured when getting the DoG modol.\n{exc}"
      )

