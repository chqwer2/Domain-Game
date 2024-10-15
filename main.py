import time
import numpy as np
import os, copy
# import dataloaders.niftiio as nio
import pickle as pkl
import wandb, torch

from models import create_forward

from pdb import set_trace
from tqdm import tqdm
# from configs_exp import ex # configuration files
from tensorboardX import SummaryWriter
from lib import test_one_epoch, eval_one_epoch, pseudo_one_epoch, printer
from config import get_config
from dataset import get_dataset, update_dataset

# torch.autograd.set_detect_anomaly(True)
import warnings, logging
warnings.filterwarnings('ignore')


def main():
    opt = get_config()
    opt.experiment = opt.experiment + "_" + opt.m.replace(" ", "")
    exp_folder = f"./result/{opt.mission}/{opt.experiment}"

    
    opt.snapshot_dir = os.path.join( exp_folder, 'snapshot' )
    opt.pred_dir = os.path.join( exp_folder, 'pred_dir' )
    opt.image_dir = os.path.join( exp_folder, 'image_dir' )
    tbfile_dir = os.path.join( exp_folder, 'tboard_file' )
    tb_writer = SummaryWriter( tbfile_dir )
    

    os.makedirs(tbfile_dir, exist_ok=True)
    os.makedirs(opt.image_dir, exist_ok=True)
    os.makedirs(opt.snapshot_dir, exist_ok=True)
    os.makedirs(opt.pred_dir, exist_ok=True)
    opt.log_dir = logging.basicConfig(filename=os.path.join( exp_folder, 'logger.txt' ), level=logging.INFO)

    loaders, datasets = get_dataset(opt)
    test_scan_info = datasets['test_tgt'].info_by_scan
    # new_val_loaders = copy.deepcopy(loaders['val'])
    val_loader_iter = iter(loaders['val'])

    if opt.exp_type == 'gin' or opt.exp_type == 'ginipa':
        model = create_forward(opt)
        
    elif opt.exp_type == 'erm':
        raise NotImplementedError # coming soon
    else:
        raise NotImplementedError(opt.exp_type)

    total_steps = 0
    total_iter = 0
    max_trainstep = 100
    
    if opt.phase == 'test' or opt.phase == 'pseudo':
        opt.epoch_count = 0
        opt.niter = 0
        opt.niter_decay = 0
        
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        # np.random.seed()
        update_interval = 10 if not opt.debug else 2
        
        if (epoch + 1) % update_interval == 0:   # TODO: 10
            print("=== UPDATE train dataset")
            loaders['train'] = update_dataset(opt, datasets['train'], loaders['train'])

        if opt.phase == 'train' or opt.phase == 'pseudo_train':
            for i, train_batch in tqdm(enumerate(loaders['train']), total = loaders['train'].dataset.size // opt.batchSize - 1):
                if i > max_trainstep:
                    break

                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time
                total_steps += opt.batchSize
                epoch_iter  += opt.batchSize

                # avoid batchsize issues caused by fetching last training batch
                if train_batch["img"].shape[0] != opt.batchSize:
                    continue

                if not opt.contrast:
                    model.set_input_aug_sup(train_batch)
                    suc = model.optimize_parameters(total_iter, opt)

                ## run a training step
                else:
                    model.set_input_contrast(train_batch)
                    
                    if (total_iter % opt.D_freq == 0) and opt.use_discriminator:
                        model.set_requires_grad(model.netD, True)
                        model.backward_D(total_iter)
                        model.set_requires_grad(model.netD, False)
                        
                    suc = model.optimize_parameters_contrast(total_iter, opt)

                if not suc:
                    model.load('latest')

                ## display training losses
                if total_steps % opt.display_freq == 0:
                    tr_viz = model.get_current_visuals_tr()
                    # model.plot_image_in_tb(tb_writer, tr_viz)  # TODO

                if total_steps % opt.print_freq == 0:
                    tr_error = model.get_current_errors_tr()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    model.track_scalar_in_tb(tb_writer, tr_error, total_steps)

                ## run and display validation losses
                if total_steps % opt.validation_freq == 0:
                    eval_one_epoch(model, val_loader_iter, datasets, total_steps, opt, tb_writer)

                iter_data_time = time.time()
                total_iter += 1


        if opt.phase == 'pseudo_gen':
            pseudo_one_epoch(model, loaders['test_tgt'], loaders['test_src'], test_scan_info,
                           opt, epoch, total_steps, datasets['label_name'])
            print("message = ", opt.m)
            return

        ## test
        if (epoch % opt.infer_epoch_freq == 0):
            test_one_epoch(model, loaders['test_tgt'], loaders['test_src'], test_scan_info,
                           opt, epoch, total_steps, datasets['label_name'])
            print("message = ", opt.m)

        if opt.phase == 'test':
            return

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        if epoch == opt.early_stop_epoch:
            return

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        model.update_learning_rate()


if __name__ == "__main__":
    main()
