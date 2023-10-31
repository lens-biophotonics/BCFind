from .dice_loss import DiceLoss
from .framed_crossentropy import FramedCrossentropy3D
from .framed_focal_crossentropy import FramedFocalCrossentropy3D
from .moe_losses import ImportanceLoss, LoadLoss, MUMLRegularizer

# import tensorflow as tf

# tf.keras.utils.get_custom_objects().update(
#     {
#         "DiceLoss": DiceLoss,
#         "FramedCrossentropy3D": FramedCrossentropy3D,
#         "FramedFocalCrossentropy3D": FramedFocalCrossentropy3D,
#         "ImportanceLoss": ImportanceLoss,
#         "LoadLoss": LoadLoss,
#         "MUMLRegularizer": MUMLRegularizer,
#     }
# )

# print(tf.keras.utils.get_custom_objects())
