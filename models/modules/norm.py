from torch import nn

# "To install see: https://github.com/mapillary/inplace_abn"




def get_normalization(norm_type, **kwargs):

    if norm_type == "inplace":
        from inplace_abn import InPlaceABN
        return InPlaceABN(out_channels=36,
                       activation="leaky_relu",
                       activation_param=0.0)

    if norm_type == "bn":
        return nn.BatchNorm2d(**kwargs)   # num_features

    elif norm_type == "gn":
        return nn.GroupNorm(**kwargs)     # num_groups, num_channels

    elif norm_type == "ln":
        return nn.LayerNorm(**kwargs)     # normalized_shape

    else:
        return nn.Identity()


from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
