import torch
from torch import nn
from .norm import get_normalization
from .act  import Activation


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# -------------------------
# Unet Support
# -------------------------
class Conv2dCustom(nn.Sequential):
    def __init__(
        self,
        in_channels, out_channels, kernel_size,
        padding=0, stride=1, norm="bn", act="leaky_relu"
    ):

        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=not (norm),
        )

        act = Activation(act)
        norm = get_normalization(norm, num_features=out_channels)

        if norm == "inplace":
            act = nn.Identity()

        super(Conv2dCustom, self).__init__(conv, norm, act)

class Conv2dAdaLN(nn.Module):
    """
    A Conv block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size,
        padding=0, stride=1, norm="bn", act="leaky_relu"
    ):
        super(Conv2dCustom).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=not (norm),
        )

        self.act = Activation(act)
        norm = get_normalization(norm, out_channels)

        if norm == "inplace":
            self.act = nn.Identity()


        self.attn = SEAttention("scse", in_channels=out_channels)
        self.norm1 = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(out_channels, elementwise_affine=False, eps=1e-6)

        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_channels, 6 * out_channels, bias=True)
        )


    def forward(self, x, c = None):
        if not isinstance(c, None):
            x = x + self.conv(x)
            x = self.attn(x + self.act(self.norm1(x)))
            return x

        shift_msa, scale_msa, gate_msa, shift_con, scale_con, gate_con = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_con.unsqueeze(1) * self.conv(modulate(self.norm2(x), shift_con, scale_con))
        return x




# For SEAttention
class SEAttention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)




# -------------------------
# DiT Support
# -------------------------
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


class Patchify(nn.Module):
    def __init__(self, in_channels, out_channels, input_size,  patch_size, hidden_size):
        super().__init__()
        self.out_channels = out_channels

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        # Will use fixed sin-cos embedding:
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

    def patchify(self, x):
        x = self.x_embedder(x) + self.pos_embed
        return x

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs




class DiTBlockPatch(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, in_channels, input_size=32, hidden_size=384,
                       num_heads=12, mlp_ratio=4.0, patch_size=4, **block_kwargs):
        super().__init__()

        self.patch = Patchify(in_channels, in_channels, input_size, patch_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )


    def forward(self, x, c):
        x = self.patch.patchify(x)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = self.patch.unpatchify(x)
        return x


