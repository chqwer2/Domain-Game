"""
Basic model modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
"""
import os
import torch, numpy as np
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict

def median_filter(input, kernel_size=20):
    # Define a median filter kernel
    kernel = torch.ones(1, 3, kernel_size, kernel_size) / (kernel_size * kernel_size)
    kernel = kernel.cuda()

    # Apply median filtering
    output = F.conv2d(input, kernel, padding=kernel_size // 2, groups=1)
    return output


# helper functions
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


class BasicTrainer(object):
    def name(self):
        return 'BasicTrainer'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.snapshot_dir = opt.snapshot_dir
        self.pred_dir = opt.pred_dir

    def set_input(self, input):
        self.input = input


    def lr_update(self):
        for _scheduler in self.schedulers:
            _scheduler.step()

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.snapshot_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # load a specific network directly
    def load_network_by_fid(self, network, fid):
        network.load_state_dict(torch.load(fid))
        print(f'Load: network {fid} as been loaded')

    # copy paste things, copying from save_network and load_networl
    # helper saving function that can be used by subclasses
    def save_optimizer(self,optimizer, optimizer_label, epoch_label, gpu_ids):
        save_filename = '%s_optim_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.snapshot_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = '%s_optim_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self.load_dir, save_filename)
        optimizer.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def as_np(self, data):
        return data.cpu().data.numpy()

    # added from new cycleGAN code
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # Get Error Stat
    def get_current_errors_tr(self):
        """
        Nothing
        """
        ret_errors = [ ('Dice', self.loss_dice),
                ('WCE', self.loss_wce),
                ('Consist', self.loss_consist_tr)]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors


    def get_current_errors_val(self):
        ret_errors = [('loss_wce_val', self.loss_wce_val.mean()),\
                ('loss_dice_val', self.loss_dice_val.mean())]

        ret_errors = OrderedDict(ret_errors)
        return ret_errors

    def get_current_visuals_val(self):
        img_val    = t2n(self.input_img.data)
        gth_val    = t2n(self.gth_val.data)
        pred_val   = t2n( torch.argmax(self.pred_val.data, dim =1, keepdim = True ))

        ret_visuals = OrderedDict([\
                ('img_seen_val', img_val),\
                ('pred_val', pred_val * 1.0 / self.n_cls),\
                ('gth_val', gth_val * 1.0 / self.n_cls)
                ])
        return ret_visuals

    def get_current_visuals_tr(self):
        img_tr  = t2n( to01(self.input_img_3copy[: self._nb].data, True))
        pred_tr = t2n( torch.argmax(self.seg_tr.data, dim =1, keepdim = True )  )
        gth_tr  = t2n(self.input_mask.data )

        ret_visuals = OrderedDict([\
                ('img_seen_tr', img_tr),\
                ('seg_tr', (pred_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )),\
                ('gth_seen_tr', (gth_tr + 0.01) * 1.0 / (self.n_cls + 0.01 )), \
                ])

        if hasattr(self, 'blend_mask'):
            blend_tr  = t2n(self.blend_mask[:,0:1, ...] )
            ret_visuals['blendmask'] = (blend_tr + 0.01) * 1.0 / (1 + 0.01 )

        return ret_visuals

    def plot_image_in_tb(self, writer, result_dict):
        for key, img in result_dict.items():
            print("plot img:", img.shape)
            writer.add_image(key, img)

    def track_scalar_in_tb(self, writer, result_dict, which_iter):
        for key, val in result_dict.items():
            writer.add_scalar(key, val, which_iter)


def t2n(x):
    if isinstance(x, np.ndarray):
        return x
    if x.is_cuda:
        x = x.data.cpu()
    else:
        x = x.data

    return np.float32(x.numpy())

def to01(x, by_channel = False):
    if not by_channel:
        out = (x - x.min()) / (x.max() - x.min())
    else:
        nb, nc, nh, nw = x.shape
        xmin = x.view(nb, nc, -1).min(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        xmax = x.view(nb, nc, -1).max(dim = -1)[0].unsqueeze(-1).unsqueeze(-1).repeat(1,1,nh, nw)
        out = (x - xmin + 1e-5) / (xmax - xmin + 1e-5)
    return out
