"""
Utilities for image transforms, part of the code base credits to Dr. Jo Schlemper
"""
from os.path import join
import torch
import numpy as np
import torchvision.transforms as deftfx
from . import image_transforms as myit
import copy
import math
from torchvision.transforms.functional import rotate as torchrotate
from torchvision.transforms.functional import InterpolationMode

my_augv = {
'flip'      : { 'v':False, 'h':False, 't': False, 'p':0.25 },
'affine'    : {
  'rotate':20,
  'shift':(15,15),
  'shear': 20,
  'scale':(0.5, 1.5),
},
'elastic'   : {'alpha':20,'sigma':5}, # medium
'reduce_2d': True,
'gamma_range': (0.2, 1.8),
'noise' : {
    'noise_std': 0.15,
    'clip_pm1': False
    },
'bright_contrast': {
    'contrast': (0.60, 1.5),
    'bright': (-10,  10)
    }
}

tr_aug = {
    'aug': my_augv
}


def get_contrast_example(image, random_angle=0, flip=0):
    if flip == [3]:
        flip = [1, 2]
    # [..., H, W]
    image_rotate = torchrotate(image, random_angle,
                                interpolation=InterpolationMode.BILINEAR)  # Bilinear
    image_rotate = torch.flip(image_rotate, flip)

    return image_rotate
    
    

def get_geometric_transformer(aug, order=3):
    affine     = aug['aug'].get('affine', 0)
    alpha      = aug['aug'].get('elastic',{'alpha': 0})['alpha']
    sigma      = aug['aug'].get('elastic',{'sigma': 0})['sigma']
    flip       = aug['aug'].get('flip', {'v': True, 'h': True, 't': True, 'p':0.125})

    tfx = []
    if 'flip' in aug['aug']:
        tfx.append(myit.RandomFlip3D(**flip))

    if 'affine' in aug['aug']:
        tfx.append(myit.RandomAffine(affine.get('rotate'),
                                     affine.get('shift'),
                                     affine.get('shear'),
                                     affine.get('scale'),
                                     affine.get('scale_iso',True),
                                     order=order))

    if 'elastic' in aug['aug']:
        tfx.append(myit.ElasticTransform(alpha, sigma))

    input_transform = deftfx.Compose(tfx)
    return input_transform

def get_intensity_transformer(aug):

    def gamma_tansform(img):
        gamma_range = aug['aug']['gamma_range']
        if isinstance(gamma_range, tuple):
            gamma = np.random.rand() * (gamma_range[1] - gamma_range[0]) + gamma_range[0]
            cmin = img.min()
            irange = (img.max() - cmin + 1e-5)

            img = img - cmin + 1e-5
            img = irange * np.power(img * 1.0 / irange,  gamma)
            img = img + cmin

        elif gamma_range == False:
            pass
        else:
            raise ValueError("Cannot identify gamma transform range {}".format(gamma_range))
        return img

    def brightness_contrast(img):
        '''
        Chaitanya,K. et al. Semi-Supervised and Task-Driven data augmentation,864in: International Conference on Information Processing in Medical Imaging,865Springer. pp. 29–41.
        '''
        cmin, cmax = aug['aug']['bright_contrast']['contrast']
        bmin, bmax = aug['aug']['bright_contrast']['bright']
        c = np.random.rand() * (cmax - cmin) + cmin
        b = np.random.rand() * (bmax - bmin) + bmin
        img_mean = img.mean()
        img = (img - img_mean) * c + img_mean + b
        return img

    def zm_gaussian_noise(img):
        """
        zero-mean gaussian noise
        """
        noise_sigma = aug['aug']['noise']['noise_std']
        noise_vol = np.random.randn(*img.shape) * noise_sigma
        img = img + noise_vol

        if aug['aug']['noise']['clip_pm1']: # if clip to plus-minus 1
            img = np.clip(img, -1.0, 1.0)
        return img

    def compile_transform(img):
        # bright contrast
        if 'bright_contrast' in aug['aug'].keys():
            img = brightness_contrast(img)

        # gamma
        if 'gamma_range' in aug['aug'].keys():
            img = gamma_tansform(img)

        # additive noise
        if 'noise' in aug['aug'].keys():
            img = zm_gaussian_noise(img)

        return img

    return compile_transform


def transform_with_label(aug, add_pseudolabel = False):
    """
    Doing image geometric transform
    Proposed image to have the following configurations
    [H x W x C + CL]
    Where CL is the number of channels for the label. It is NOT a one-hot thing
    """

    geometric_tfx = get_geometric_transformer(aug)
    intensity_tfx = get_intensity_transformer(aug)

    def transform(comp, c_label, c_img, c_sam, nclass, is_train, use_onehot = False):
        """
        Args
            comp:               a numpy array with shape [H x W x C + c_label]
            c_label:            number of channels for a compact label. Note that the current version only supports 1 slice (H x W x 1)
            nc_onehot:          -1 for not using one-hot representation of mask. otherwise, specify number of classes in the label
            is_train:           whether this is the training set or not. If not, do not perform the geometric transform
        """
        comp = copy.deepcopy(comp)
        if (use_onehot is True) and (c_label != 1):
            raise NotImplementedError("Only allow compact label, also the label can only be 2d")
        assert c_img + c_sam + c_label == comp.shape[-1], "only allow single slice 2D label"

        if is_train is True:
            _label = comp[..., c_img ]
            _sam   = np.expand_dims(comp[..., c_img+c_label], axis=-1)
            # compact to onehot
            _h_label = np.float32(np.arange( nclass ) == (_label[..., None]) )
            # print("h_label=", _h_label.shape)
            # print("_sam=", _sam.shape)

            comp = np.concatenate( [comp[...,  :c_img ], _h_label, _sam], -1 )
            comp = geometric_tfx(comp)
            # round one_hot labels to 0 or 1
            t_label_h = comp[..., c_img : -c_sam]
            t_label_h = np.rint(t_label_h)
            t_img = comp[..., 0 : c_img ]
            t_sam = np.rint(comp[..., -c_sam:])

        # intensity transform
        t_img = intensity_tfx(t_img)

        if use_onehot is True:
            t_label = t_label_h
        else:
            t_label = np.expand_dims(np.argmax(t_label_h, axis = -1), -1)
        return t_img, t_label, t_sam

    return transform




def gamma_concern(img, gamma):
    mean = np.mean(img)

    img = (img - mean) * gamma
    img = img + mean
    img = np.clip(img, 0, 1)

    return img

def gamma_power(img, gamma, direction=0):
    if direction == 1:
        img = 1 - img
    img = np.power(img, gamma)

    img = img / np.max(img)
    if direction == 1:
        img = 1 - img

    return img

def gamma_exp(img, gamma, direction=0):
    if direction == 1:
        img = 1 - img

    img = np.exp(img * gamma)
    img = img / np.max(img)

    if direction == 1:
        img = 1 - img
    return img


def GammaInterference(img):
    # Shape Span
    gamma = np.random.random() * 1.5 + 0.25  # 0.25 ~ 1.75
    # gamma = np.random.random() * 1.75 + 0.25  # 0.25 ~ 1.75
    img = gamma_concern(img, gamma)  # concerntrate

    # Shape Tilt
    choose = np.random.randint(0, 2)
    direction = np.random.randint(0, 2)

    if choose == 0:
        gamma = 0.2 + np.random.random() * 2.3  # 2.5
        img = gamma_power(img, gamma, direction)
    else:
        gamma = np.random.random() * 2.3 + 0.6  # 1.5 center
        img = gamma_exp(img, gamma, direction)

    return img




