import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import modules as md
from typing import Optional, Union, List

from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    ClassificationHead,
)



class DualUnet(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type="scse",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
        sam = False,
        norm_type = "bn", act_type = "relu"):

        super().__init__()
        out_channels = 1
        if sam:
            in_channels += 1

        self.encoder = get_encoder(
            encoder_name, in_channels=in_channels,
            depth=encoder_depth, weights=encoder_weights,
        )
        self.encoder_dual = get_encoder(
            encoder_name, in_channels=in_channels,
            depth=encoder_depth, weights=encoder_weights,
        )
        
        # ------------------
        # Decoder
        # ------------------
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            k=3,
            norm_type=norm_type,
            act_type=act_type,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.decoder_dual = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            norm_type=norm_type,
            act_type=act_type,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.stem_head = md.extra_stem(decoder_channels[-1], decoder_channels[-1])


        self.segmentation_head = md.SegmentationHead(
            in_channels=decoder_channels[-1], out_channels=classes,
            activation=activation, kernel_size=3
        )

        self.reconstruction_head = md.ReconstructionHead(
            in_channels=decoder_channels[-1],  out_channels=out_channels,
            activation=activation,  kernel_size=3
        )

        self.name = "u-{}".format(encoder_name)
        self.classification_head = None
        self.initialize()



# ------------------------------------
# Decoder Modules
# ------------------------------------

class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels, skip_channels, out_channels,  kernel=3,
        norm=True,  act="leaky_relu",  attention=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dCustom(
            in_channels + skip_channels,
            out_channels,
            kernel_size=kernel,
            padding=(kernel-1)//2, norm=norm, act=act
        )
        self.attention1 = md.SEAttention(attention, in_channels=in_channels + skip_channels)
        
        self.conv2 = md.Conv2dCustom(
            out_channels,
            out_channels,
            kernel_size=kernel,
            padding=(kernel-1)//2, norm=norm, act=act
        )

        self.attention2 = md.SEAttention(attention, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_type="bn", act_type="leaky_relu"):
        conv1 = md.Conv2dCustom(
            in_channels, out_channels,
            kernel_size=3, padding=1, norm=norm_type, act=act_type
        )
        conv2 = md.Conv2dCustom(
            out_channels, out_channels,
            kernel_size=3, padding=1, norm=norm_type, act=act_type
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels, decoder_channels, n_blocks=5, k = 3,
        norm_type="bn", act_type="leaky_relu", attention_type=None, center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels,
                                      norm_type=norm_type, act_type=act_type)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(norm=norm_type, attention=attention_type, act=act_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, kernel=k, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x



class UnetCenterAttention(nn.Module):
    def __init__(
        self,
        encoder_channels, decoder_channels, n_blocks=5, k = 3,
        norm_type="bn", act_type="leaky_relu", attention_type=None, center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])

        # combine decoder keyword arguments

        blocks = [
            md.SEAttention(attention_type, in_channels=in_ch) for in_ch in zip(in_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x
