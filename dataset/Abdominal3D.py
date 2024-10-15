# Dataloader for abdominal images
import glob
import numpy as np
from .utils import niftiio as nio
from .utils import transform_utils as trans
from .utils.abd_dataset_utils import get_normalize_op
from .utils.transform_albu import get_albu_transforms, get_resize_transforms

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
from multiprocessing import Process

hostname = platform.node()
LABEL_NAME = ["bg", "liver", "rk", "lk", "spleen"]


def get_basedir(opt):
    return os.path.join(opt.data_dir, "Abdominal")


class Abdominal3D(torch_data.Dataset):
    def __init__(self, opt, mode, transforms, base_dir, domains: list, pseudo = False, idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, extern_norm_fn = None):
        """
        Args:
            mode:               'train', 'val', 'test', 'test_all'
            transforms:         naive data augmentations used by default. Photometric transformations slightly better than those configured by Zhang et al. (bigaug)
            idx_pct:            train-val-test split for source domain
            extern_norm_fn:     feeding data normalization functions from external, only concerns CT-MR cross domain scenario
        """
        super(Abdominal3D, self).__init__()
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
        self.test_resizer   = get_resize_transforms((opt.fineSize, opt.fineSize))
        self.fake_interpolate    =  True # True
        
        self.input_window = opt.input_window

        self.img_pids = {}
        for _domain in self.domains: # load file names
            self.img_pids[_domain] = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in
                                     glob.glob(self._base_dir + "/" + _domain + "/img/img_*.nii.gz") ],
                                     key = lambda x: int(x))

        self.scan_ids = self.__get_scanids(mode, idx_pct) # train val test split in terms of patient ids

        self.info_by_scan = None
        self.sample_list = self.__search_samples(self.scan_ids) # image files names according to self.scan_ids
        if self.is_train:
            self.pid_curr_load = self.scan_ids
        elif mode == 'val':
            self.pid_curr_load = self.scan_ids
        elif mode == 'test': # Source domain test
            self.pid_curr_load = self.scan_ids
        elif mode == 'test_all':
            # Choose this when being used as a target domain testing set. Liu et al.
            self.pid_curr_load = self.scan_ids
        if extern_norm_fn is None:
            self.normalize_op = get_normalize_op(self.domains[0], [itm['img_fid'] for _, itm in
                                                                   self.sample_list[self.domains[0]].items() ])
            print(f'{self.phase}_{self.domains[0]}: Using fold data statistics for normalization')
        else:
            assert len(self.domains) == 1, 'for now we only support one normalization function for the entire set'
            self.normalize_op = extern_norm_fn

        print(f'For {self.phase} on {[_dm for _dm in self.domains]} using scan ids {self.pid_curr_load}')

        # load to memory
        self.actual_dataset = self.__read_dataset()
        self.size = len(self.actual_dataset) # 2D

    def __get_scanids(self, mode, idx_pct):
        """
        index by domains given that we might need to load multi-domain data
        idx_pct: [0.7 0.1 0.2] for train val test. with order te val tr
        """
        tr_ids      = {}
        val_ids     = {}
        te_ids      = {}
        te_all_ids  = {}

        for _domain in self.domains:
            dset_size   = len(self.img_pids[_domain])
            tr_size     = round(dset_size * idx_pct[0])
            val_size    = math.floor(dset_size * idx_pct[1])
            te_size     = dset_size - tr_size - val_size

            te_ids[_domain]     = self.img_pids[_domain][: te_size]
            val_ids[_domain]    = self.img_pids[_domain][te_size: te_size + val_size]
            tr_ids[_domain]     = self.img_pids[_domain][te_size + val_size: ]
            te_all_ids[_domain] = list(itertools.chain(tr_ids[_domain], te_ids[_domain], val_ids[_domain]   ))

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
        for _domain, id_list in scan_ids.items():
            domain_dir = os.path.join(self._base_dir, _domain)
            print("=== reading domains from:", domain_dir)
            out_list[_domain] = {}
            for curr_id in id_list:
                curr_dict = {}

                _img_fid = os.path.join(domain_dir, 'img', f'img_{curr_id}.nii.gz')
                if not self.pseudo:
                    _lb_fid  = os.path.join(domain_dir, 'seg', f'seg_{curr_id}.nii.gz')
                else:
                    _lb_fid  = os.path.join(domain_dir, 'seg', f'pseudo_{curr_id}.nii.gz.npy')  # npy
                _sam_fid = os.path.join(domain_dir, 'seg', f'sam_{curr_id}.npy')

                curr_dict["img_fid"] = _img_fid
                curr_dict["lbs_fid"] = _lb_fid
                curr_dict["sam_fid"] = _sam_fid
                out_list[_domain][str(curr_id)] = curr_dict

        print("=== search sample num:", len(out_list))
        return out_list


    def __read_dataset(self):
        """
        Read the dataset into memory
        """

        out_list = []
        self.info_by_scan = {} # meta data of each scan
        glb_idx = 0 # global index of a certain slice in a certain scan in entire dataset
        print_idx = 0
        for _domain, _sample_list in self.sample_list.items():
            for scan_id, itm in _sample_list.items():
                if scan_id not in self.pid_curr_load[_domain]:
                    continue  # Pass
                
                img, _info = nio.read_nii_bysitk(itm["img_fid"], peel_info = True) # get the meta information out
                self.info_by_scan[_domain + '_' + scan_id] = _info

                img = np.float32(img)
                
                if (print_idx + 1) % 10:
                    print(f" {_domain} stat: ", img.shape, img.max(), img.min())
                    
                print_idx += 1   
                
                # img, self.mean, self.std = self.normalize_op(img)
                _, mean, std = self.normalize_op(img)
                if not self.pseudo:
                    lb = nio.read_nii_bysitk(itm["lbs_fid"])
                else:
                    uncertainty_thr = 0.05  #  0.05
                    lb_cache = np.load(itm["lbs_fid"], allow_pickle=True).item()
                    lb = lb_cache['pseudo'].cpu().numpy()              # "pseudo": curr_pred, "score":curr_score , "uncertainty"
                    uncertainty = lb_cache['uncertainty'].cpu().numpy()   # Z, C, H, W
                    uncertainty = np.float32(uncertainty)
                    
                    new_lb = np.zeros_like(lb)
                    for cls in range(self.opt.nclass - 1):
                        un_mask = (uncertainty[:, cls+1] < uncertainty_thr ) * (cls+1)
                        
                        # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
                        new_lb[lb == (cls+1)] = un_mask[lb == (cls+1)]
                        # print("un_mask=", un_mask.shape, ", new_lb=",new_lb.shape)
                    # print("=== new_lb stat=", new_lb.shape, new_lb.max(), new_lb.min())
                    lb = new_lb
                    
         

                lb = np.float32(lb)
                sam = np.load(itm["sam_fid"])

                # H, W, C
                img     = np.transpose(img, (1, 2, 0))  
                lb      = np.transpose(lb, (1, 2, 0))
                sam     = np.transpose(sam, (1, 2, 0))
                

                # filter zero
                if self.phase == "train":
                    # C, H, W
                    filter = np.any(np.any(img, axis=0), axis=0)
                    img     = img[..., filter]
                    lb      = lb[..., filter]
                    sam     = sam[..., filter]

                # print("after filter:", img.shape)
                H, W, C = img.shape

                assert img.shape[-1] == lb.shape[-1]

                out_list, glb_idx = self.add_to_list(glb_idx, out_list, img, lb,
                                                     sam, mean, std, _domain, scan_id, itm["img_fid"])

                if self.phase == "train":
                    # WCH
                    continue
                    img = np.transpose(img, (1, 2, 0))  # HCW
                    lb = np.transpose(lb, (1, 2, 0))
                    sam = np.transpose(sam, (1, 2, 0))
                    
                    img = np.resize(img, (H, W, img.shape[-1])) # [192^3]
                    lb = np.resize(lb, (H, W, lb.shape[-1]))
                    sam = np.resize(sam, (H, W, sam.shape[-1]))
                    
                    
                    filter = np.any(np.any(lb, axis=0), axis=0)
                    img_flitered = img[..., filter].copy()
                    lb_flitered  = lb[..., filter].copy()
                    sam_flitered = sam[..., filter].copy()
                    # print("after step 1 filter:", img_flitered.shape)
                    # filter lb where has label
                    
                    out_list, glb_idx = self.add_to_list(glb_idx, out_list, 
                                                         img_flitered, lb_flitered, sam_flitered, 
                                                         mean, std, _domain, scan_id)

                    # CHW
                    # img = np.transpose(img, (1, 2, 0))
                    # lb = np.transpose(lb, (1, 2, 0))
                    # sam = np.transpose(sam, (1, 2, 0))
                    # filter = np.any(np.any(lb, axis=0), axis=0)
                    # img_flitered = img[..., filter].copy()
                    # lb_flitered  = lb[..., filter].copy()
                    # sam_flitered = sam[..., filter].copy()

                    # out_list, glb_idx = self.add_to_list(glb_idx, out_list, img_flitered, lb_flitered,
                    #                                      sam_flitered, mean, std, _domain, scan_id)

        return out_list

    # def parallel_read_dataset()

    def add_to_list(self, glb_idx, out_list, img, lb, sam, mean, std, _domain, scan_id, file_id):
        # now start writing everthing in
        c = 3
        # img = np.transpose(img, (1, 2, 0))
        
        # write the HWC frame
        out_list.append( {"img": img,
                       "lb":lb,
                       "sam":sam,
                       "mean":mean,
                        "std":std,
                       "domain": _domain,
                       "nframe": img.shape[-1],
                       "scan_id": _domain + "_" + scan_id,
                       "pid": scan_id,
                        "file_id": file_id})
        glb_idx += 1
        
        if self.phase == "train":
            H, W, C = img.shape
            # H C W
            img = np.transpose(img, (1, 2, 0))          
            lb = np.transpose(lb, (1, 2, 0))
            sam = np.transpose(sam, (1, 2, 0))
            img = np.resize(img, (H, W, img.shape[-1])) # [192^3]
            lb = np.resize(lb, (H, W, lb.shape[-1]))
            sam = np.resize(sam, (H, W, sam.shape[-1]))
            
            out_list.append( {"img": img,
                        "lb":lb,
                        "sam":sam,
                        "mean":mean,
                        "std":std,
                        "domain": _domain,
                        "nframe": img.shape[-1],
                        "scan_id": _domain + "_" + scan_id,
                        "pid": scan_id,
                        "file_id": file_id})
            glb_idx += 1
            
            # CHW
            img = np.transpose(img, (1, 2, 0))          # HCW
            lb = np.transpose(lb, (1, 2, 0))
            sam = np.transpose(sam, (1, 2, 0))

            out_list.append( {"img": img,
                        "lb":lb,
                        "sam":sam,
                        "mean":mean,
                        "std":std,
                        "domain": _domain,
                        "nframe": img.shape[-1],
                        "scan_id": _domain + "_" + scan_id,
                        "pid": scan_id,
                        "file_id": file_id})
            glb_idx += 1
            
        return out_list, glb_idx

    # 384 -> 320
    def get_patch_from_img_3D(self, img_H, img_L, img_L2, crop_size=[320, 320, 320], zslice_dim=2):
        # --------------------------------
        # randomly crop the patch
        # --------------------------------

        H, W, C = img_H.shape
        rnd_h = random.randint(0, max(0, H - crop_size[0]))
        rnd_w = random.randint(0, max(0, W - crop_size[1]))
        rnd_c = random.randint(0, max(0, C - crop_size[1]))

        # image = torch.index_select(image, 0, torch.tensor([1]))
        patch_H  = img_H[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], rnd_c:rnd_c + crop_size[2]]
        patch_L  = img_L[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], rnd_c:rnd_c + crop_size[2]]
        patch_L2 = img_L2[rnd_h:rnd_h + crop_size[0], rnd_w:rnd_w + crop_size[1], rnd_c:rnd_c + crop_size[2]]

        return patch_H, patch_L, patch_L2

    def hwc_to_chw(self, img):
        img = np.float32(img)
        img = np.transpose(img, (2, 0, 1))   # [C, H, W]
        img = torch.from_numpy( img.copy() )
        return img
    

    def __getitem__(self, index):
        index = index % len(self.actual_dataset)
        curr_dict = self.actual_dataset[index]  # numpy

        # ----------------------- Extract Slice -----------------------
        img = curr_dict["img"]     # H, W, C, [0 - 255]
        mask = curr_dict["lb"]     # H, W, C
        sam  = curr_dict["sam"]
        mean, std = curr_dict['mean'], curr_dict['std']
        domain = curr_dict["domain"]
        pid    = curr_dict["pid"]
        file_id = curr_dict["file_id"]

        max = img.max()
        min = img.min()
        minmax = np.maximum(max - min, 1)
        std    = 1 if std < 1e-3 else std
        
        if not self.pseudo:
            if domain == "CHAOST2":
                mask = mask[::-1]      # H reverse
            elif domain == "SABSCT" and int(pid) == 13:
                mask = mask[::-1]
            
        if self.is_train:
            T = self.albu_transform
        else:
            T = self.test_resizer
        
        img = (img - mean) / std
        
        buffer = T(image = img, mask=mask, mask2=sam)  # [0 - 255]
        img  = buffer['image']
        mask = buffer['mask']
        sam  = buffer['mask2']

        if self.is_train:
            windowsize = img.shape[-1]
            comp = np.concatenate( [ img, mask, sam ], axis = -1 )
            if self.transforms:
                img, mask, sam = self.transforms(comp, c_img = windowsize, 
                                          c_label = windowsize, c_sam = windowsize,
                                          nclass = self.nclass,
                                          is_train = self.is_train,
                                          use_onehot = False)
                
            img, mask, sam = self.get_patch_from_img_3D(img, mask, sam, crop_size=[160, 160, 160])  # 192

        img  = self.hwc_to_chw(img)
        mask = self.hwc_to_chw(mask)
        sam  = self.hwc_to_chw(sam)

        img_minmax = ((img * std) + mean - min) / minmax  

        # sam Mask out 
        sam_mask = np.ones_like(sam)
        if self.is_train is True:
            num_patch = np.random.randint(1, 4)
            for i in range(num_patch):
                patch_size = np.random.randint(0.1 * sam.shape[-1], 0.5 * sam.shape[-1])
                rnd_h = np.random.randint(0, np.maximum(0, sam.shape[-2] - patch_size))
                rnd_w = np.random.randint(0, np.maximum(0, sam.shape[-1] - patch_size))
                rnd_c = np.random.randint(0, np.maximum(0, sam.shape[0]  - patch_size))
                sam_mask[rnd_c:rnd_c+patch_size,  rnd_h: rnd_h+patch_size, rnd_w:rnd_w+patch_size] = 0  # Mask out


        is_start    = curr_dict["is_start"]
        is_end      = curr_dict["is_end"]
        nframe      = np.int32(curr_dict["nframe"])
        scan_id     = curr_dict["scan_id"]
        z_id        = curr_dict["z_id"]


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

        return data




    def __len__(self):
        """
        copy-paste from basic naive dataset configuration
        """
        return len(self.actual_dataset)




