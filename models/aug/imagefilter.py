# GIN
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import torch.nn.init as init


class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, kernel_pool = [1, 3], scale_ratio = 0.1, layer_id = 0, use_act = True, requires_grad = False, use_gpu=True, **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.kernel_pool     = kernel_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        self.use_gpu        = use_gpu
        self.silu           = nn.SiLU()
        assert requires_grad == False
        self.kernel_init    = "gaussian"    # "xavier" | "gaussian" | "kaiming"

    def forward(self, x_in, requires_grad = False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        # idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k     = int(np.random.choice(self.kernel_pool, size=(1,)))    #[idx_k[0]]
        nb, nc, nx, ny = x_in.shape

        # Xavier initialization

        # Gaussian initialization
        # RandomNormal(mean=0.0, stddev=0.05)
        if self.kernel_init == "gaussian":
            stddev = 1 # 0.05
            kernel = torch.randn([self.out_channel * nb, self.in_channel, k, k], requires_grad = self.requires_grad ) * stddev
        elif self.kernel_init == "xavier":
            kernel = torch.empty([self.out_channel * nb, self.in_channel, k, k])
            init.xavier_uniform_(kernel)
        elif self.kernel_init == "kaiming":
            kernel = torch.empty([self.out_channel * nb, self.in_channel, k, k])
            init.kaiming_uniform_(kernel, mode='fan_in', nonlinearity='leaky_relu')


        bias   = torch.randn([self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad) * 1.0
        # bias   = torch.randn([self.out_channel * nb, nx, ny], requires_grad=self.requires_grad) * 1.0

        if self.use_gpu:
            kernel = kernel.cuda()
            bias = bias.cuda()
        else:
            kernel = kernel.double()
            bias = bias.double()

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, kernel, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_conv = x_conv + bias
        if self.use_act:
            scale = np.random.random() * 0.2 + 0.9
            negative_slope_scale = 0.1 * scale
            # x_conv = torch.sigmoid(x_conv)   # More Smooth
            x_conv = F.leaky_relu(x_conv)  # 0.01  negative_slope=negative_slope_scale
            # x_conv = self.silu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel = 3, in_channel = 3, interm_channel = 2,
                 scale_pool = [1, 3 ], n_layer = 4, out_norm = 'frob', use_gpu=True, **kwargs):
        '''
        GIN
        '''
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool # don't make it tool large as we have multiple layers
        self.n_layer = n_layer
        self.layers = []
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.use_gpu = use_gpu

        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = in_channel, scale_pool = scale_pool, layer_id = 0, use_gpu=use_gpu)
                )
        for ii in range(n_layer - 2):
            self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = interm_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = ii + 1, use_gpu=use_gpu)
                )
        self.layers.append(
            GradlessGCReplayNonlinBlock(out_channel = out_channel, in_channel = interm_channel, scale_pool = scale_pool, layer_id = n_layer - 1, use_act = False, use_gpu=use_gpu)
                )

        self.layers = nn.ModuleList(self.layers)
        if self.use_gpu:
            self.layers = self.layers.cuda()

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim = 0)

        nb, nc, nx, ny = x_in.shape

        alphas = torch.rand(nb)[:, None, None, None] # nb, 1, 1, 1
        alphas = alphas.repeat(1, nc, 1, 1) # nb, nc, 1, 1
        if self.use_gpu:
            alphas = alphas.cuda()

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)
        mixed = alphas * x + (1.0 - alphas) * x_in

        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim = (-1, -2), p = 'fro', keepdim = False)
            _in_frob = _in_frob[:, None, None, None].repeat(1, nc, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim = (-1,-2), p = 'fro', keepdim = False)
            _self_frob = _self_frob[:, None, None, None].repeat(1, self.out_channel, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5 ) ) * _in_frob

        return mixed


