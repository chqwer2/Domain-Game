# pip install -U git+https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp

model = smp.DeepLabV3(
    encoder_name="mit_b2",    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    # encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)

print(model)

# mit_b2
# Below is a table of suitable encoders (for DeepLabV3, DeepLabV3+,
# and PAN dilation support is needed also)
#  'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5',

# https://github.com/PRLAB21/MaxViT-UNet

