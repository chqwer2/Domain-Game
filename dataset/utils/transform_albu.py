# -*- encoding: utf-8 -*-
#Time        :2022/02/24 18:14:15
#Author      :Hao Chen
#FileName    :trans_lib.py
#Version     :2.0

import cv2
import torch
import numpy as np
import albumentations as A
def gaussian_noise(img, mean, sigma):
    return img + torch.FloatTensor(img.shape).normal_(mean=mean, std=sigma)
# from albumentations.pytorch import ShiftScaleRotate

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



def get_resize_transforms(img_size = (192, 192)):
    # if type == 'train':
    return A.Compose([
        A.Resize(img_size[0], img_size[1])
    ], p=1.0, additional_targets={'image2': 'image', "mask2": "mask"})


def get_albu_transforms(img_size = (192, 192)):
    # if type == 'train':
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=(-0.2, 0.2), 
                            rotate_limit=30, p=0.5),
        
        # A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=0.5),
        A.GaussNoise(var_limit=(10.0, 25.0), p=0.5),
        
        # A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        # A.Emboss(alpha=(0.5, 1.0), strength=(0.5, 1.0), p=0.5),  # Added
        
        # A.FDA([target_image], p=1, read_fn=lambda x: x)
        # A.PixelDistributionAdaptation( reference_images=[reference_image],
        
        # A.Defocus(radius=(4, 8), alias_blur=(0.2, 0.4), p=0.5)
        
        # Randomly posterize between 2 and 5 bits
        # A.Posterize(num_bits=(4, 6), p=0.5),
        
        # A.OneOf([
        #     A.RandomShadow(p=1.0),
        #     A.Solarize(p=1.0),
        #     A.RandomSunFlare(p=1.0),
        # ], p=0.5),
        
        # A.Saturation
        A.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=0, val_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), 
                                    contrast_limit=(-0.1, 0.1), p=0.5),
        A.MaskDropout(p=0.5),
        # A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.3),  
        
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0), 
            A.ElasticTransform(alpha=1, sigma=5, alpha_affine=5, p=1.0)
        ], p=0.5),

        A.CoarseDropout(max_holes=5, max_height=img_size[0] // 20, max_width=img_size[1] // 20,
                        min_holes=1, fill_value=0, mask_fill_value=0, p=0.5),

    ], p=1.0, additional_targets={'image2': 'image', "mask2": "mask"})




# Beta function
def gamma_concern(img, gamma):
    mean = torch.mean(img)

    img = (img - mean) * gamma
    img = img + mean
    img = torch.clip(img, 0, 1)

    return img

def gamma_power(img, gamma, direction=0):
    if direction == 1:
        img = 1 - img
    img = torch.pow(img, gamma)

    img = img / torch.max(img)
    if direction == 1:
        img = 1 - img

    return img

def gamma_exp(img, gamma, direction=0):
    if direction == 1:
        img = 1 - img

    img = torch.exp(img * gamma)
    img = img / torch.max(img)

    if direction == 1:
        img = 1 - img
    return img




