import torch
from torch import nn


class tumor_model(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # ------------------------------------------ Bio Model
        k = 3
        self.rho = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
            torch.nn.LeakyReLU(),  # SiLU
            torch.nn.Conv2d(out_channels, 1, kernel_size=k, padding=k // 2),
        )

        self.c = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
            torch.nn.LeakyReLU(),  # SiLU
            torch.nn.Conv2d(out_channels, 1, kernel_size=k, padding=k // 2),
        )

        D_length = 2
        self.D_softmax = nn.Softmax(D_length)
        self.D_parameter = nn.Parameter((torch.arange(D_length, dtype=torch.float)))
        self.D = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
            torch.nn.LeakyReLU(),  # SiLU
            torch.nn.Conv2d(out_channels, D_length, kernel_size=k, padding=k // 2),
        )

        # c = np.random.randn(64, 64, 64, 3)
        # D_parameter = [1, 2]  # Classes
        # D = np.random.randn(64, 64, 64, 3)  # Classes relates? D classes?, softmax?
        #
        # rho = np.random.randn(64, 64, 64, 3)  # Should be a constant, GAP? Should work
        # K = np.max(c)  # some thing Max, but it scales? or a Constant? Maybe 100?

        # upsampling = torch.nn.UpsamplingBilinear2d(scale_factor=1) if 1 > 1 else torch.nn.Identity()
        # activation = Activation(activation)
        # super().__init__(conv2d, upsampling, activation)

    def forward(self, feature):
        outputs = {}

        c = self.c(feature)
        rho = self.rho(feature)
        D_map = self.D(feature)
        softmax_D = self.D_softmax(D_map)
        D = torch.sum(softmax_D * self.D_parameter, dim=-1, keepdim=True)  # torch.argmax(D_map, axis=-1)

        outputs['c'] = c
        outputs['rho'] = rho
        outputs['D'] = D
        return outputs
