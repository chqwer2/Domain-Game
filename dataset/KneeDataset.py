# Dataloader for abdominal images
import glob
import numpy as np
from .utils import niftiio as nio
from .utils import transform_utils as trans
from .utils.abd_dataset_utils import get_normalize_op
from .utils.transform_albu import get_albu_transforms, get_resize_transforms

import torch
import os
from pdb import set_trace
from multiprocessing import Process
from .BasicDataset import BasicDataset

LABEL_NAME = ["bg", "femur bone", "femur cartilage", "tibia bone", "tibia cartilage"]



class KneeDataset(BasicDataset):
    def __init__(self, opt, mode, transforms, base_dir, domains: list, **kwargs):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        self.dataset_key = "knee"
        super(KneeDataset, self).__init__(opt, mode, transforms, base_dir, domains,
                                          LABEL_NAME=LABEL_NAME, filter_non_labeled=True, **kwargs)

        self.original_size = [384, 384]  # down sample
        self.use_size = [192, 192]  # down scale x2
        self.crop_size = [160, 160]  # crop size

    def hwc_to_chw(self, img):
        img = np.float32(img)
        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        img = torch.from_numpy(img.copy())
        return img

    def perform_trans(self, img, mask, sam):

        T = self.albu_transform if self.is_train else self.test_resizer
        buffer = T(image=img, mask=mask, mask2=sam)  # [0 - 255]
        img, mask, sam = buffer['image'], buffer['mask'], buffer['mask2']

        if self.is_train:
            comp = np.concatenate([img, mask, sam], axis=-1)
            if self.transforms:
                img, mask, sam = self.transforms(comp, c_img=self.input_window, c_label=1, c_sam=1,
                                                 nclass=self.nclass, is_train=self.is_train, use_onehot=False)

            img, mask, sam = self.get_patch_from_img(img, mask, sam, crop_size=self.crop_size)  # 192

        return img, mask, sam

    def mask_sam(self, sam):
        sam_mask = np.ones_like(sam)
        if self.is_train is True:
            num_patch = np.random.randint(1, 4)
            for i in range(num_patch):
                patch_size = np.random.randint(0.1 * sam.shape[-1], 0.5 * sam.shape[-1])
                rnd_h = np.random.randint(0, np.maximum(0, sam.shape[-2] - patch_size))
                rnd_w = np.random.randint(0, np.maximum(0, sam.shape[-1] - patch_size))
                sam_mask[:, rnd_h: rnd_h + patch_size, rnd_w:rnd_w + patch_size] = 0  # Mask out
        return sam_mask

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]  # numpy

        # ----------------------- Extract Slice -----------------------
        img, mask, sam = curr_dict["img"], curr_dict["lb"], curr_dict["sam"]  # H, W, C, [0 - 255]
        domain, pid = curr_dict["domain"], curr_dict["pid"]
        mean, std = curr_dict['mean'], curr_dict['std']

        # max, min = img.max(), img.min()
        std = 1 if std < 1e-3 else std

        img = (img - mean) / std
        img, mask, sam = self.perform_trans(img, mask, sam)

        img, mask, sam = self.hwc_to_chw(img), self.hwc_to_chw(mask), self.hwc_to_chw(sam)
        img_minmax = (img - img.min()) / (img.max() - img.min())  # ((img * std) + mean - min) / minmax    # [0 - 1]

        # sam_mask_patch
        sam_mask = self.mask_sam(sam)

        if self.tile_z_dim > 1 and self.input_window == 1:
            img = img.repeat([self.tile_z_dim, 1, 1])
            assert img.ndimension() == 3

        data = {"img": img, "img_minmax": img_minmax, "lb": mask, "sam": sam, "sam_mask": sam_mask,
                "is_start": curr_dict["is_start"], "is_end": curr_dict["is_end"],
                "nframe": np.int32(curr_dict["nframe"]),
                "scan_id": curr_dict["scan_id"], "z_id": curr_dict["z_id"], "file_id": curr_dict["file_id"]
                }

        return data



