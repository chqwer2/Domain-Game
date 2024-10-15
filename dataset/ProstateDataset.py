# Dataloader for abdominal images
import glob
import numpy as np
from .utils import niftiio as nio
from .utils import transform_utils as trans
from .utils.abd_dataset_utils import get_normalize_op
from .utils.transform_albu import get_albu_transforms, get_resize_transforms

import time
import torch
import random
import os
import copy
import platform
import json
import torch.utils.data as torch_data
import math
import itertools
from pdb import set_trace

hostname = platform.node()
# folder for datasets
# BASEDIR = './data/abdominal/'
# print(f'Running on machine {hostname}, using dataset from {BASEDIR}')


LABEL_NAME = ["bg", "prostate"]



def get_basedir(opt):
    return os.path.join(opt.data_dir, "Prostate")


class ProstateDataset(torch_data.Dataset):
    def __init__(self, opt, mode, transforms, base_dir, domains: list,
                 pseudo=False, idx_pct=[0.7, 0.1, 0.2],
                 tile_z_dim=3, extern_norm_fn=None):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        super(ProstateDataset, self).__init__()
        self.opt = opt
        self.transforms = transforms
        self.is_train = True if mode == 'train' else False
        self.phase = mode
        self.domains = domains
        self.pseudo = pseudo
        self.all_label_names = LABEL_NAME
        self.nclass = len(LABEL_NAME)
        self.tile_z_dim = tile_z_dim
        self._base_dir = base_dir
        self.idx_pct = idx_pct
        self.albu_transform = get_albu_transforms()
        self.test_resizer = get_resize_transforms((opt.fineSize, opt.fineSize))

        self.fake_interpolate = True  # True

        self.input_window = opt.input_window

        self.img_pids = {}
        for _domain in self.domains:  # load file names
            self.img_pids[_domain] = sorted([fid.split("_")[-1].split(".nii.gz")[0] for fid in
                                             glob.glob(self._base_dir + "/" + _domain + "/img/img_*.nii.gz")],
                                            key=lambda x: int(x))

        self.scan_ids = self.__get_scanids(mode, idx_pct)  # train val test split in terms of patient ids

        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids)  # image files names according to self.scan_ids

        self.pid_curr_load = self.scan_ids

        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [itm['img_fid'] for _, itm in
                                                                   self.sample_list[self.domains[0]].items()])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
            assert len(self.domains) == 1, 'for now we only support one normalization function for the entire set'
            self.normalize_op = extern_norm_fn

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using domain length =  {len(self.pid_curr_load)}')

        # load to memory
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset)  # 2D

    def __get_scanids(self, mode, idx_pct):
        """
        index by domains given that we might need to load multi-domain data
        idx_pct: [0.7 0.1 0.2] for train val test. with order te val tr
        """
        tr_ids = {}
        val_ids = {}
        te_ids = {}
        te_all_ids = {}

        for _domain in self.domains:
            dset_size = len(self.img_pids[_domain])
            tr_size = round(dset_size * idx_pct[0])
            val_size = math.floor(dset_size * idx_pct[1])
            te_size = dset_size - tr_size - val_size

            te_ids[_domain] = self.img_pids[_domain][: te_size]
            val_ids[_domain] = self.img_pids[_domain][te_size: te_size + val_size]
            tr_ids[_domain] = self.img_pids[_domain][te_size + val_size:]
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]))

        if self.phase == 'train':
            return tr_ids
        elif self.phase == 'val':
            return val_ids
        elif self.phase == 'test':
            return te_ids
        elif self.phase == 'test_all':
            return te_all_ids

    def __search_samples(self, scan_ids):
        """search for filenames for images and masks
        """
        out_list = {}
        self.all_img_list = {}
        self.all_seg_list = {}
        self.zip_domain_id = {}

        for _domain, id_list in scan_ids.items():
            domain_dir = os.path.join(self._base_dir, _domain)
            print("=== reading domains from:", domain_dir)
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(domain_dir, 'img', f'img_{curr_id}.nii.gz')
                if not self.pseudo:
                    _lb_fid = os.path.join(domain_dir, 'seg', f'seg_{curr_id}.nii.gz')
                else:
                    _lb_fid = os.path.join(domain_dir, 'seg', f'pseudo_{curr_id}.nii.gz.npy')  # npy
                _sam_fid = os.path.join(domain_dir, 'seg', f'sam_{curr_id}.npy')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                curr_dict["sam_fid"] = _sam_fid
                out_list[_domain][str(curr_id)] = curr_dict

                self.all_img_list[_img_fid] = ""
                self.all_seg_list[_lb_fid] = ""
                self.zip_domain_id[_img_fid] = [_domain, curr_id]
                self.zip_domain_id[_lb_fid] = [_domain, curr_id]

        print("=== search sample num:", len(out_list))
        return out_list

    # @ti.kernel

    def load_nii_img(self, imgpath):
        _domain, scan_id = self.zip_domain_id[imgpath]

        if scan_id not in self.pid_curr_load[_domain]:
            return

        img, _info = nio.read_nii_bysitk(imgpath, peel_info=True)  # get the meta information out
        img = np.float32(img)
        self.info_by_scan[imgpath] = _info
        self.all_img_list[imgpath] = img

    def load_label(self, labpath):
        _domain, scan_id = self.zip_domain_id[labpath]

        if scan_id not in self.pid_curr_load[_domain]:
            return

        lb = nio.read_nii_bysitk(labpath)
        lb = np.float32(lb)
        self.all_seg_list[labpath] = lb

        # if not self.pseudo:
        #     lb = nio.read_nii_bysitk(itm["lbs_fid"])
        # else:
        #     uncertainty_thr = 0.05  # 0.05
        #     lb_cache = np.load(itm["lbs_fid"], allow_pickle=True).item()
        #     lb = lb_cache['pseudo'].cpu().numpy()  # "pseudo": curr_pred, "score":curr_score , "uncertainty"
        #     uncertainty = lb_cache['uncertainty'].cpu().numpy()  # Z, C, H, W
        #     uncertainty = np.float32(uncertainty)
        #
        #     new_lb = np.zeros_like(lb)
        #     for cls in range(self.opt.nclass - 1):
        #         un_mask = (uncertainty[:, cls + 1] < uncertainty_thr) * (cls + 1)
        #
        #         # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
        #         new_lb[lb == (cls + 1)] = un_mask[lb == (cls + 1)]
        #         # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
        #     # print("=== new_lb stat=", new_lb.shape, new_lb.max(), new_lb.min())
        #     lb = new_lb



    def __read_dataset(self):
        """
        Read the dataset into memory
        """
        import time
        start_time = time.time()
        import multiprocessing
        self.info_by_scan = {}

        # img_cache = []
        from multiprocessing.dummy import Pool

        # with Pool() as p:
        #     arr = p.map(self.load_nii_img, list(self.all_img_list))

        # print("finish loading time=", time.time()-start_time)

        # self.all_img_list.append(_img_fid)
        # self.all_seg_list.append(_lb_fid)
        # read all img/seg in

        out_list = []
        for _domain, _sample_list in self.sample_list.items():
            max_id = 200
            _sample_list = list(_sample_list.items())[:max_id]  #
            self._domain = _domain

            # Time = 89.3
            # for idx, (scan_id, itm) in enumerate(_sample_list):
            #     results = self.parallel_read_dataset(_domain, scan_id, itm, idx, len(_sample_list))
            #     out_list.extend(results)

            tmp = []
            for idx, (scan_id, itm) in enumerate(_sample_list):
                tmp.append([_domain, scan_id, itm, idx, len(_sample_list)])  # cannot pickle

            with Pool() as p:
                results = p.map(self.parallel_read_dataset, tmp)  # [results]

            for i in results:
                out_list.extend(i)
        # print
        print("using time=", time.time() - start_time)
        return out_list

    # @ti.kernel
    def parallel_read_dataset(self, l):
        (_domain, scan_id, itm, idx, length) = l
        out_list = []
        if scan_id not in self.pid_curr_load[_domain]:
            return out_list  # Pass

        img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info=True)  # get the meta information out
        self.info_by_scan[_domain + '_' + scan_id] = _info
        img = np.float32(img)

        # QUANTILE = {0.02: -1024, 0.98: 3071}
        # if self._domain == "oai":
        #     QUANTILE = {0.02: 0.0, 0.98: 189.63}
        # elif self._domain == "ski":
        #     QUANTILE = {0.02: 4.0, 0.98: 2315.0}
        #
        # img[img > QUANTILE[0.98]] = QUANTILE[0.98]
        # img[img < QUANTILE[0.02]] = QUANTILE[00.02]

        if (idx + 1) % (max(length // 10, 5)) == 0:
            print(f" {idx + 1}/{length}  {_domain} stat:   ", img.shape, img.max(), img.min())

        # img, self.mean, self.std = self.normalize_op(img)
        _, mean, std = self.normalize_op(img)
        if not self.pseudo:
            lb = nio.read_nii_bysitk(itm["lbs_fid"])
        else:
            uncertainty_thr = 0.05  # 0.05
            lb_cache = np.load(itm["lbs_fid"], allow_pickle=True).item()
            lb = lb_cache['pseudo'].cpu().numpy()  # "pseudo": curr_pred, "score":curr_score , "uncertainty"
            uncertainty = lb_cache['uncertainty'].cpu().numpy()  # Z, C, H, W
            uncertainty = np.float32(uncertainty)

            new_lb = np.zeros_like(lb)
            for cls in range(self.opt.nclass - 1):
                un_mask = (uncertainty[:, cls + 1] < uncertainty_thr) * (cls + 1)

                # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
                new_lb[lb == (cls + 1)] = un_mask[lb == (cls + 1)]
                # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
            # print("=== new_lb stat=", new_lb.shape, new_lb.max(), new_lb.min())
            lb = new_lb

        lb = np.float32(lb)

        try:
            sam = np.load(itm["sam_fid"])
        except:
            sam = lb

        # H, W, C
        img = np.transpose(img, (1, 2, 0))
        lb = np.transpose(lb, (1, 2, 0))
        sam = np.transpose(sam, (1, 2, 0))

        # print("load img shape: ", img.shape, lb.shape)

        # filter zero
        if self.phase == "train":
            # C, H, W
            filter = np.any(np.any(img, axis=0), axis=0)
            img = img[..., filter]
            lb = lb[..., filter]
            sam = sam[..., filter]

        lb[lb > 1] = 1


        # print("after filter:", img.shape)
        H, W, C = img.shape

        assert img.shape[-1] == lb.shape[-1]

        out_list = self.add_to_list(out_list, img, lb,
                                    sam, mean, std, _domain, scan_id, itm["img_fid"])

        if self.phase == "train" and False:
            # WCH
            # continue
            img = np.transpose(img, (1, 2, 0))  # HCW
            lb = np.transpose(lb, (1, 2, 0))
            sam = np.transpose(sam, (1, 2, 0))

            img = np.resize(img, (H, W, img.shape[-1]))  # [192^3]
            lb = np.resize(lb, (H, W, lb.shape[-1]))
            sam = np.resize(sam, (H, W, sam.shape[-1]))

            filter = np.any(np.any(lb, axis=0), axis=0)
            img_flitered = img[..., filter].copy()
            lb_flitered = lb[..., filter].copy()
            sam_flitered = sam[..., filter].copy()
            # print("after step 1 filter:", img_flitered.shape)
            # filter lb where has label

            out_list, glb_idx = self.add_to_list(glb_idx, out_list,
                                                 img_flitered, lb_flitered, sam_flitered,
                                                 mean, std, _domain, scan_id)

            # CHW
        return out_list

    def add_to_list(self, out_list, img, lb, sam, mean, std, _domain, scan_id, file_id):
        # now start writing everthing in
        c = 3

        # write the beginning frame
        if self.input_window == 3:
            start_img = img[..., 0: c].copy()
            start_img[..., 1] = start_img[..., 0]
        elif self.input_window == 1:
            start_img = img[..., 0: 0 + 1].copy()

        out_list.append({"img": start_img,
                         "lb": lb[..., 0: 0 + 1],
                         "sam": sam[..., 0: 0 + 1],
                         "mean": mean,
                         "std": std,
                         "is_start": True,
                         "is_end": False,
                         "domain": _domain,
                         "nframe": img.shape[-1],
                         #    "n_cls"
                         "scan_id": _domain + "_" + scan_id,
                         "pid": scan_id,
                         "file_id": file_id,
                         "z_id": 0})
        # self.glb_idx += 1

        for ii in range(1, img.shape[-1] - 1):
            # print("slice minmax =", img[..., ii -1: ii + 2].max(),
            #       img[..., ii -1: ii + 2].min())
            if self.input_window == 3:
                middle_img = img[..., ii - 1: ii + 2].copy()
            elif self.input_window == 1:
                middle_img = img[..., ii: ii + 1].copy()

            out_list.append({"img": middle_img,
                             "lb": lb[..., ii: ii + 1],
                             "sam": sam[..., ii: ii + 1],
                             "mean": mean,
                             "std": std,
                             "is_start": False,
                             "is_end": False,
                             "nframe": -1,
                             "domain": _domain,
                             "scan_id": _domain + "_" + scan_id,
                             "pid": scan_id,
                             "file_id": file_id,
                             "z_id": ii
                             })
            # self.glb_idx += 1

            if self.fake_interpolate and _domain == "CHAOST2" and self.is_train:  #
                middle_img = img[..., ii - 1: ii].copy() / 2 + img[..., ii: ii + 1].copy() / 2
                middle_lb = np.maximum(lb[..., ii - 1: ii], lb[..., ii: ii + 1])

                middle_sam = sam[..., ii - 1: ii] + sam[..., ii: ii + 1]
                middle_sam = np.clip(middle_sam, 0, 1)

                out_list.append({
                    "img": middle_img,
                    "lb": middle_lb,
                    "sam": middle_sam,
                    "mean": mean,
                    "std": std,
                    "is_start": False,
                    "is_end": False,
                    "nframe": -1,
                    "domain": _domain,
                    "scan_id": _domain + "_" + scan_id,
                    "pid": scan_id,
                    "file_id": file_id,
                    "z_id": ii - 0.5
                })
                # self.glb_idx += 1

        ii += 1  # last frame, note the is_end flag

        if self.input_window == 3:
            end_image = img[..., ii - 2: ii + 1].copy()
            end_image[..., 0] = end_image[..., 1]
        elif self.input_window == 1:
            end_image = img[..., ii: ii + 1].copy()

        out_list.append({"img": end_image,
                         "lb": lb[..., ii: ii + 1],
                         "sam": sam[..., ii: ii + 1],
                         "mean": mean,
                         "std": std,
                         "is_start": False,
                         "is_end": True,
                         "nframe": -1,
                         "domain": _domain,
                         "scan_id": _domain + "_" + scan_id,
                         "pid": scan_id,
                         "file_id": file_id,
                         "z_id": ii
                         })
        # self.glb_idx += 1

        return out_list

    def get_patch_from_img(self, img_H, img_L, img_L2, crop_size=[160, 160], zslice_dim=2):
        # --------------------------------
        # randomly crop the patch
        # --------------------------------

        H, W, _ = img_H.shape
        rnd_h = random.randint(0, max(0, H - crop_size[0]))
        rnd_w = random.randint(0, max(0, W - crop_size[1]))

        # image = torch.index_select(image, 0, torch.tensor([1]))
        if zslice_dim == 2:
            patch_H = img_H[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
            patch_L = img_L[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
            patch_L2 = img_L2[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], :]
        elif zslice_dim == 0:
            patch_H = img_H[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]
            patch_L = img_L[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]
            patch_L2 = img_L2[:, rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1]]

        return patch_H, patch_L, patch_L2

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]  # numpy

        # ----------------------- Extract Slice -----------------------
        img = curr_dict["img"]  # H, W, C, [0 - 255]
        mask = curr_dict["lb"]  # H, W, C
        sam = curr_dict["sam"]
        mean, std = curr_dict['mean'], curr_dict['std']
        domain = curr_dict["domain"]
        pid = curr_dict["pid"]
        file_id = curr_dict["file_id"]
        max = img.max()
        min = img.min()
        minmax = np.maximum(max - min, 1)
        std = 1 if std < 1e-3 else std

        if not self.pseudo:
            if domain == "CHAOST2":
                mask = mask[::-1]  # H reverse
            elif domain == "SABSCT" and int(pid) == 13:
                mask = mask[::-1]

        # img = (img - min)/minmax     # [0 - 1.0]
        if self.is_train:
            T = self.albu_transform
        else:
            T = self.test_resizer

        # img = (img - min) / minmax * 255
        buffer = T(image=img, mask=mask, mask2=sam)  # [0 - 255]
        img = buffer['image']
        mask = buffer['mask']
        sam = buffer['mask2']

        # img = trans.GammaInterference(img)
        # img = np.clip(img, 0, 1)
        #

        img = (img.copy() - mean) / std

        if self.is_train:
            comp = np.concatenate([img, mask, sam], axis=-1)
            if self.transforms:
                img, mask, sam = self.transforms(comp, c_img=self.input_window,
                                                 c_label=1, c_sam=1,
                                                 nclass=self.nclass,
                                                 is_train=self.is_train,
                                                 use_onehot=False)

            img, mask, sam = self.get_patch_from_img(img, mask, sam)

        img = np.float32(img)
        mask = np.float32(mask)
        sam = np.float32(sam)

        img = np.transpose(img, (2, 0, 1))  # [C, H, W]
        mask = np.transpose(mask, (2, 0, 1))  # [C, H, W]
        sam = np.transpose(sam, (2, 0, 1))  # [C, H, W]

        img = torch.from_numpy(img.copy())
        mask = torch.from_numpy(mask.copy())
        sam = torch.from_numpy(sam.copy())  # * 10
        img_minmax = ((img * std) + mean - min) / minmax  # [0 - 1]

        # sam_mask_patch
        sam_mask = np.ones_like(sam)
        if self.is_train is True:
            num_patch = np.random.randint(1, 4)
            for i in range(num_patch):
                patch_size = np.random.randint(0.1 * sam.shape[-1], 0.5 * sam.shape[-1])
                rnd_h = np.random.randint(0, np.maximum(0, sam.shape[-2] - patch_size))
                rnd_w = np.random.randint(0, np.maximum(0, sam.shape[-1] - patch_size))
                sam_mask[:, rnd_h: rnd_h + patch_size, rnd_w:rnd_w + patch_size] = 0  # Mask out

        if self.tile_z_dim > 1 and self.input_window == 1:
            img = img.repeat([self.tile_z_dim, 1, 1])
            assert img.ndimension() == 3

        is_start = curr_dict["is_start"]
        is_end = curr_dict["is_end"]
        nframe = np.int32(curr_dict["nframe"])
        scan_id = curr_dict["scan_id"]
        z_id = curr_dict["z_id"]

        # smaller than -200??
        # img[img < -100] = -100

        # print("=== data prepare:", img.max(), img_minmax.max(), mask.max(), sam.max())

        data = {"img": img,
                "img_minmax": img_minmax,
                "lb": mask,
                "sam": sam,
                "sam_mask": sam_mask,
                "is_start": is_start,
                "is_end": is_end,
                "nframe": nframe,
                "scan_id": scan_id,
                "z_id": z_id,
                "file_id": file_id
                }

        if self.opt.contrast:
            random_angle = np.random.randint(-30, 30)  # 0~360
            flip = np.random.choice([1, 2, 3], replace=False, size=1).tolist()

            data['img_r'] = trans.get_contrast_example(img, random_angle, flip)
            data['lb_r'] = trans.get_contrast_example(mask, random_angle, flip)
            data['sam_r'] = trans.get_contrast_example(sam, random_angle, flip)
            data['img_minmax_r'] = trans.get_contrast_example(img_minmax, random_angle, flip)
            data['angle'] = random_angle
            data['flip'] = flip

        return data

    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)


tr_func = trans.transform_with_label(trans.tr_aug)


def get_training(opt, modality, idx_pct=[0.7, 0.1, 0.2], tile_z_dim=3, pseudo=False):
    return ProstateDataset(opt, idx_pct=idx_pct,
                       mode="train",  # pseudo_train
                       pseudo=pseudo,
                       domains=modality,
                       transforms=tr_func,
                       base_dir=get_basedir(opt),
                       extern_norm_fn=None,  # normalization function is decided by domain
                       tile_z_dim=tile_z_dim)


def get_validation(opt, modality, norm_func, idx_pct=[0.7, 0.1, 0.2], tile_z_dim=3):
    return ProstateDataset(opt, idx_pct=idx_pct,
                       mode='val',
                       transforms=None,
                       domains=modality,
                       base_dir=get_basedir(opt),
                       extern_norm_fn=norm_func,
                       tile_z_dim=tile_z_dim)


def get_test(opt, modality, norm_func, tile_z_dim=3, idx_pct=[0.7, 0.1, 0.2]):
    return ProstateDataset(opt, idx_pct=idx_pct,
                       mode='test',
                       transforms=None,
                       domains=modality,
                       extern_norm_fn=norm_func,
                       base_dir=get_basedir(opt),
                       tile_z_dim=tile_z_dim)


def get_test_all(opt, modality, norm_func, tile_z_dim=3, idx_pct=[0.7, 0.1, 0.2]):
    return ProstateDataset(opt, idx_pct=idx_pct,
                       mode='test_all',
                       transforms=None,
                       domains=modality,
                       extern_norm_fn=norm_func,
                       base_dir=get_basedir(opt),
                       tile_z_dim=tile_z_dim)




