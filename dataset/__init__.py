from torch.utils.data import DataLoader
import numpy as np
from . import AbdominalDataset as abdominal
from . import BrainDataset as brain
from . import ProstateDataset as prostate
from . import KneeDataset as knee
import os
from .utils import transform_utils as trans

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_dataset(opt, idx_pct = [0.7, 0.1, 0.2], chunksize=200):
    print("=== mission:", opt.mission)
    kwargs = dict()
    if not isinstance(opt.tr_domain, list):
        opt.tr_domain = [opt.tr_domain]
    if not isinstance(opt.te_domain, list):
        opt.te_domain = [opt.te_domain]
            
    if opt.mission.lower() == 'abdominal':
        datast_func  = abdominal.AbdominalDataset
        label_name   = abdominal.LABEL_NAME
        basedir      = os.path.join(opt.data_dir, "Abdominal")

    elif opt.mission.lower() == 'knee':
        datast_func  = knee.KneeDataset
        label_name   = knee.LABEL_NAME
        basedir      = os.path.join(opt.data_dir, "Knee")
        kwargs = dict(chunksize = 100 )  # use_diff_axis_view= True,

    elif opt.mission.lower() == 'prostate':
        datast_func  = prostate.ProstateDataset
        label_name   = prostate.LABEL_NAME
        basedir      = os.path.join(opt.data_dir, "Prostate")

    elif opt.mission.lower() == 'brain':
        datast_func  = brain.BrainDataset
        label_name   = brain.LABEL_NAME
        
        basedir      = os.path.join(opt.data_dir, "brats/Brain")
        if not os.path.exists(basedir):
            basedir  = os.path.join(opt.data_dir, "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
        
        if not os.path.exists(basedir):
            basedir = "/home/hao/data/medical/Brain"
            
        idx_pct      = [0.9, 0.05, 0.05]  # validation
        kwargs = dict( chunksize = opt.chunksize ) 

    else:
        raise NotImplementedError(opt.data_name)

    train_set       = set_dataset(opt, datast_func, "train", basedir, idx_pct = idx_pct,
                                  modality = opt.tr_domain, pseudo = opt.pseudo, **kwargs)
    val_source_set  = set_dataset(opt, datast_func, "val", basedir, idx_pct = idx_pct,
                                  modality = opt.tr_domain, norm_func = train_set.normalize_op)
   
    
    if opt.te_domain[0] == opt.tr_domain[0]:
        # if same domain, then use the normalize op from the source
        test_set        = set_dataset(opt, datast_func, "test", basedir, idx_pct = idx_pct,
                                      modality = opt.te_domain, norm_func = train_set.normalize_op)
        test_source_set = test_set
        
    else:
        test_set        = set_dataset(opt, datast_func, "test_all", basedir, idx_pct = idx_pct,
                                      modality = opt.te_domain, norm_func = None) # train_set.normalize_op) 
        # norm_func used to be None
        
        test_source_set = set_dataset(opt, datast_func, "test", basedir, idx_pct = idx_pct,
                                      modality = opt.tr_domain, norm_func = None) # train_set.normalize_op)


    print(f'Using TR domain {opt.tr_domain}; TE domain {opt.te_domain}')
    train_loader = DataLoader(dataset = train_set, num_workers = opt.nThreads, batch_size = opt.batchSize, 
                              shuffle = True, drop_last = True, worker_init_fn = worker_init_fn,
                              pin_memory = True)

    val_loader = iter(DataLoader(dataset = val_source_set, num_workers = 1, batch_size = 1, 
                                 shuffle = False, pin_memory = True))

    test_tgt_loader = DataLoader(dataset = test_set, num_workers = 1, batch_size = 1, 
                                 shuffle = False, pin_memory = True)

    test_src_loader = DataLoader(dataset = test_source_set, num_workers = 1, batch_size = 1, 
                                 shuffle = False, pin_memory = True)

    loaders = {"train": train_loader, "val": val_loader,
               "test_tgt": test_tgt_loader, "test_src": test_src_loader}

    datasets = {"train": train_set, "val": val_source_set,
                "test_tgt": test_set, "test_src": test_source_set, "label_name": label_name}

    return loaders, datasets


def update_dataset(opt, train_set, train_loader):
    del train_loader
    train_set.update_chunk()
    train_loader = DataLoader(dataset=train_set, num_workers=opt.nThreads,
                              batch_size=opt.batchSize, pin_memory=True,
                              shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
    return train_loader


def set_dataset(opt, datast_func, mode, basedir, modality, norm_func = None,
                idx_pct = [0.7, 0.1, 0.2], tile_z_dim = 3, pseudo = False, chunksize=200, **kwargs):

    norm_func = None if mode=="train" else norm_func
    tr_func = None if mode == "train" else trans.transform_with_label(trans.tr_aug)

    return datast_func(opt, idx_pct = idx_pct,
                            mode = mode,
                            pseudo = pseudo,
                            domains = modality,
                            transforms = tr_func,
                            base_dir = basedir,
                            extern_norm_fn = norm_func,
                            tile_z_dim = tile_z_dim,
                            chunksize  = chunksize, **kwargs)

