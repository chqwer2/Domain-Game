from torch.nn import init
from .network_swin_v2 import Swin_v2



def weights_init():
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__

            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def get_model_mac_score(net, inp_shape=(3, 256, 256)):
    # pip install ptflops
    from ptflops import get_model_complexity_info
    FLOPS = 0
    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=True)

    # params = float(params[:-4])
    # MACs (G) in log scale
    print(params)
    macs = float(macs[:-4]) + FLOPS / 10 ** 9

    print('mac', macs, params)



def get_swinnet(**kwargs):
    net = Swin_v2(**kwargs)
    net.apply(weights_init())
    return net

