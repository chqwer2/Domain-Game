import torch
import numpy as np
from tqdm import tqdm
import os, time
import SimpleITK as sitk
import copy
from torch.utils.data import DataLoader
import wandb, gc, logging
import matplotlib.pyplot as plt

DICE_Best = {"val":0, "target":0}
logger = logging.getLogger(__name__)

def prediction_wrapper_all(model, test_loader, opt, epoch, label_name, mode = 'base', save_prediction = False, tag="val"):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    with torch.no_grad():
        out_prediction_list = {} # a buffer for saving results
        recomp_img_list = []
        for idx, batch in tqdm(enumerate(test_loader), total = len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img'].shape
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  ))  #.cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  ))    #.cuda()
                curr_img = np.zeros( [nx, ny, nframe]  )

            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1

            model.set_input(batch)
            gth, pred = model.get_segmentation_gpu(raw_logits = False)
            curr_pred[slice_idx, ...]   = pred[0, ...].detach().cpu() # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...]    = gth[0, ...].detach().cpu()
            curr_img[:,:,slice_idx] = batch['img'][0, 1,...].numpy()
            slice_idx += 1
            # Exceed
            if batch['is_end']:
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                if opt.phase == 'test':
                    recomp_img_list.append(curr_img)

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper_all(out_prediction_list, len(label_name ), model, label_name, tag=tag)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()
        gc.collect()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper_all(vol_list, nclass, model, label_name, tag = "val"):
    """
    Evaluatation and arrange predictions
    """
    out_count = len(vol_list)
    tables_by_domain = {} # tables by domain
    conf_mat_list = [] # confusion matrices
    dsc_table = np.ones([ out_count, nclass ]  ) # rows and samples, columns are structures

    idx = 0
    for scan_id, comp in vol_list.items():
        domain, pid = scan_id.split("_")
        if domain not in tables_by_domain.keys():
            tables_by_domain[domain] = {'scores': [],
                    'scan_ids': []}
        pred_ = comp['pred']
        gth_  = comp['gth']
        dices = model.ScoreDiceEval(torch.unsqueeze(pred_, 1), gth_, dense_input = True).cpu().numpy() # this includes the background class
        tables_by_domain[domain]['scores'].append( [_sc for _sc in dices]  )
        tables_by_domain[domain]['scan_ids'].append( scan_id )
        dsc_table[idx, ...] = np.reshape(dices, (-1))
        del pred_
        del gth_
        idx += 1
        torch.cuda.empty_cache()

    # then output the result
    error_dict = {}
    for organ in range(nclass):
        mean_dc = np.mean( dsc_table[:, organ] )
        std_dc  = np.std(  dsc_table[:, organ] )
        info = "Organ {} with dice: mean: {:06.5f} \n, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc)
        print(info)
        logger.info(info)
        error_dict[label_name[organ]] = mean_dc

    d_mean = dsc_table[:,1:].mean()
    
    if DICE_Best[tag] <  d_mean and tag != "val":
        model.save(f'dice={d_mean:.4f}')
    
    DICE_Best[tag] = np.maximum(d_mean, DICE_Best[tag])
    
    info = "Overall mean dice by sample {:06.5f}, current best {:06.5f}".format( dsc_table[:,1:].mean(), DICE_Best[tag])
    print(info) # background is noted as class 0 and therefore not counted
    logger.info(info)
    error_dict['overall'] = dsc_table[:, 1:].mean()
    

    
    wandb.log({f"mean {tag} dice": d_mean})

    # then deal with table_by_domain issue
    overall_by_domain = []
    domain_names = []
    for domain_name, domain_dict in tables_by_domain.items():
        domain_scores = np.array( tables_by_domain[domain_name]['scores']  )
        domain_mean_score = np.mean(domain_scores[:, 1:])
        error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
        error_dict[f'domain_{domain_name}_table'] = domain_scores
        overall_by_domain.append(domain_mean_score)
        domain_names.append(domain_name)

    error_dict['overall_by_domain'] = np.mean(overall_by_domain)

    print("Overall mean dice by domain {:06.5f}".format( error_dict['overall_by_domain'] ) )
    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain
    # wandb.log({"mean val dice":dsc_table[:,1:].mean()})


    return error_dict, dsc_table, domain_names


color_map = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ])

def mask2color(mask):
    
    """

    Convert mask to color

    Args:

        mask: np.array The mask to convert

    Returns:

        np.array Color mask generated from the given mask

    """

    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    mask = mask.astype(int)

    for label in np.unique(mask):
        if label < color_map.shape[0]:  # Check if label is within the bounds of color_map
            color_mask[mask == label] = color_map[label]
        else:
            print(f"Warning: label {label} is out of bounds for the color map.")

    return color_mask



def visualization_wrapper(model, test_loader, opt, epoch, label_name, 
                          mode='base', save_prediction=False, tag="val", top=10):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """
    
    span = len(test_loader) // 12 + 1 #
    
    out_batch_image = []
    slice_idx = 0
    slice = 4
    start = np.random.randint(0, 100)
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            
            if slice_idx >= top:
                break
            
            if batch['is_start'] and slice_idx == 0:
                nb, nc, nx, ny = batch['img'].shape
                
                curr_pred = torch.Tensor(np.zeros([top, nx, ny])) 
                curr_gth = torch.Tensor(np.zeros([top, nx, ny]))  
                curr_img = np.zeros([top, nx, ny])
                curr_full = torch.Tensor(np.zeros([top, nx, ny]))  
                curr_anatomical = torch.Tensor(np.zeros([top, nx, ny]))
                curr_domain = torch.Tensor(np.zeros([top, nx, ny]))
                
                
                curr_full_comp = torch.Tensor(np.zeros([top, nx, slice * ny]))
                curr_anatomical_comp = torch.Tensor(np.zeros([top, nx, slice * ny]))
                curr_domain_comp = torch.Tensor(np.zeros([top,  nx, slice * ny]))
            
            if ((start + idx+1) % span) != 0:
                continue
            
            assert batch['lb'].shape[0] == 1      # enforce a batchsize of 1

            model.set_input(batch)
            gth, pred, full, anatomical, domain, full_comp, anatomical_comp, domain_comp = model.get_visualization_gpu(raw_logits=False)
            curr_pred[slice_idx, ...] = pred[0, ...].detach().cpu()  # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...] = gth[0, ...].detach().cpu()
            
            # Normalization
            full = (full - full.min())/ (full.max() - full.min())
            anatomical = (anatomical - anatomical.min())/ (anatomical.max() - anatomical.min())
            domain = (domain - domain.min())/ (domain.max() - domain.min())
            batch['img'] = (batch['img'] - batch['img'].min())/ (batch['img'].max() - batch['img'].min())
            
            buff = [(img - img.min())/(img.max() - img.min()) for img in full_comp[0, ...].detach().cpu()]
            curr_full_comp[slice_idx, ...] = torch.cat(buff, dim=1)
            
            buff = [(img - img.min())/(img.max() - img.min()) for img in anatomical_comp[0, ...].detach().cpu()]
            curr_anatomical_comp[slice_idx, ...] = torch.cat(buff, dim=1)
            
            buff = [(img - img.min())/(img.max() - img.min()) for img in domain_comp[0, ...].detach().cpu()]
            curr_domain_comp[slice_idx, ...] = torch.cat(buff, dim=1)
            
            curr_full[slice_idx, ...] = full[0, ...].detach().cpu()
            curr_anatomical[slice_idx, ...] = anatomical[0, ...].detach().cpu()
            curr_domain[slice_idx, ...] = domain[0, ...].detach().cpu()
            curr_img[slice_idx, ... ] = batch['img'][0, 1, ...].numpy()
            
            # curr_anatomical_comp[slice_idx, ...] = anatomical_comp[0, ...].detach().cpu()
            # curr_domain_comp[slice_idx, ...] = domain_comp[0, ...].detach().cpu()
            # curr_full_comp[slice_idx, ...] = full_comp[0, ...].detach().cpu()
            slice_idx += 1
            

        torch.cuda.empty_cache()
        gc.collect()

    # print("=== visualization_wrapper stats:")
    # print(f"curr_pred shape: {curr_pred.shape}, {curr_pred.min()}, {curr_pred.max()}")
    # print(f"curr_gth shape: {curr_gth.shape}, {curr_gth.min()}, {curr_gth.max()}")
    # print(f"curr_img shape: {curr_img.shape}, {curr_img.min()}, {curr_img.max()}")
    # print(f"curr_full shape: {curr_full.shape}, {curr_full.min()}, {curr_full.max()}")
    # print(f"curr_anatomical shape: {curr_anatomical.shape}, {curr_anatomical.min()}, {curr_anatomical.max()}")
    # print(f"curr_domain shape: {curr_domain.shape}, {curr_domain.min()}, {curr_domain.max()}")
    

    l = [curr_img, curr_full, curr_anatomical, curr_domain]
    mask = [curr_pred, curr_gth]
    
    comps = [curr_full_comp.numpy(), curr_anatomical_comp.numpy(), curr_domain_comp.numpy()]
    comps = np.concatenate(comps, axis=2).reshape(top * nx, len(comps) * slice * ny)
    comps = np.stack([comps,comps,comps],axis=-1)
    
    mask = np.concatenate(mask, axis=2).reshape(top * nx, len(mask) * ny)
    mask_color = mask2color(mask)
    output = np.concatenate(l, axis=2).reshape(top * nx, len(l) * ny)
    output = np.stack([output,output,output],axis=-1)
    
    # print("output, mask_color:", output.shape, mask_color.shape)
    output = np.concatenate([output, mask_color], axis=1)
    
    return output, comps




def prediction_wrapper(model, test_loader, opt, epoch, label_name, mode='base', save_prediction=False, tag="val"):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
        mode:           str Adding a note for the saved testing results
    """

    error_dict = {}
    tables_by_domain = {}  # tables by domain
    dsc_table = []
    domain_names = []
    test_case = 0
    
    with torch.no_grad():
        out_prediction_list = {}  # a buffer for saving results
        for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            if batch['is_start']:
                slice_idx = 0

                scan_id_full = str(batch['scan_id'][0])
                out_prediction_list[scan_id_full] = {}

                nframe = batch['nframe']
                nb, nc, nx, ny = batch['img'].shape
                curr_pred = torch.Tensor(np.zeros([nframe, nx, ny]))  # .cuda() # nb/nz, nc, nx, ny
                curr_gth = torch.Tensor(np.zeros([nframe, nx, ny]))  # .cuda()
                curr_img = np.zeros([nx, ny, nframe])

            assert batch['lb'].shape[0] == 1  # enforce a batchsize of 1

            model.set_input(batch)
            gth, pred = model.get_segmentation_gpu(raw_logits=False)
            curr_pred[slice_idx, ...] = pred[0, ...].detach().cpu()  # nb (1), nc, nx, ny
            curr_gth[slice_idx, ...] = gth[0, ...].detach().cpu()
            curr_img[:, :, slice_idx] = batch['img'][0, 1, ...].numpy()
            slice_idx += 1
            
            # Exceed
            if batch['is_end']:
                domain, pid = scan_id_full.split("_")
                if domain not in tables_by_domain.keys():
                    tables_by_domain[domain] = {'scores': [],
                                                'scan_ids': []}

                dsc = eval_list_wrapper(curr_pred, curr_gth, model)
                dsc_table.append(dsc)
                
                tables_by_domain[domain]['scores'].append([_sc for _sc in dsc])
                tables_by_domain[domain]['scan_ids'].append(scan_id_full)
                test_case += 1
                
                if opt.phase == "train" and test_case > 25:
                    break

        dsc_table    = np.asarray(dsc_table)
        # print("dsc_table shape = ", dsc_table.shape)
        info = "Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode)
        print(info); logger.info(info)
        
        try:
            for organ in range(len(label_name)):
                mean_dc = np.mean(dsc_table[:, organ])
                std_dc = np.std(dsc_table[:, organ])
                print("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
                logger.info("Organ {} with dice: mean: {:06.5f}, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
                
                error_dict[label_name[organ]] = mean_dc
        except:
            print("Error in calculating dice score, dataset could be vacant!")
            
        d_mean = dsc_table[:, 1:].mean()

        if DICE_Best[tag] < d_mean and tag != "val":
            model.save(f'dice={d_mean:.4f}')

        DICE_Best[tag] = np.maximum(d_mean, DICE_Best[tag])

        info = "Overall mean dice by sample {:06.5f}, current best {:06.5f}".format(
            dsc_table[:, 1:].mean(), DICE_Best[ tag])
        print(info)  # background is noted as class 0 and therefore not counted
        logger.info(info)
        
        error_dict['overall'] = dsc_table[:, 1:].mean()

        wandb.log({f"mean {tag} dice": d_mean})

        # then deal with table_by_domain issue
        overall_by_domain = []
        for domain_name, domain_dict in tables_by_domain.items():
            domain_scores = np.array(tables_by_domain[domain_name]['scores'])
            domain_mean_score = np.mean(domain_scores[:, 1:])
            error_dict[f'domain_{domain_name}_overall'] = domain_mean_score
            error_dict[f'domain_{domain_name}_table'] = domain_scores
            overall_by_domain.append(domain_mean_score)
            domain_names.append(domain_name)
            
            logger.info(f"Domain {domain_name} with dice: mean: {domain_mean_score:06.5f}")

        error_dict['overall_by_domain'] = np.mean(overall_by_domain)

        info = "Overall mean dice by domain ({}) {:06.5f}\n".format(domain_names, error_dict['overall_by_domain'])
        print(info); logger.info(info)


        # error_dict, dsc_table, domain_names =
        error_dict["mode"] = mode
        if not save_prediction:  # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()
        gc.collect()

    return out_prediction_list, dsc_table, error_dict, domain_names


def eval_list_wrapper(pred, gth, model):
    """
    Evaluatation and arrange predictions
    """
    # this includes the background class
    dices = model.ScoreDiceEval(torch.unsqueeze(pred, 1).cuda(),
                                gth.cuda(), dense_input=True).cpu().numpy()
    dice_ones = np.reshape(dices, (-1))
    del pred
    del gth

    torch.cuda.empty_cache()
    gc.collect()


    # for prostate dataset, we use by-domain results to mitigate the differences in number of samples for each target domain
    # wandb.log({"mean val dice":dsc_table[:,1:].mean()})

    return dice_ones


# test_scan_info = test_set.info_by_scan  # test_scan_info
def eval_one_epoch(model, val_loader_iter, datasets, total_steps, opt, tb_writer):
    iteration = 0
    snapshot_span = 25
    
    with torch.no_grad():
        iteration += 1
        try:
            val_batch = next(val_loader_iter)
            
        except StopIteration:
            # Reset the iterator to the beginning
            new_val_loaders = DataLoader(dataset = datasets['val'], num_workers = 1,
                batch_size = 1, shuffle = False, pin_memory = True)

            val_loader_iter = iter(new_val_loaders)
            val_batch = next(val_loader_iter)


        model.set_input(val_batch)
                
        if (iteration + 1) % snapshot_span == 0:
            # opt.image_dir
            model.validate(save=True)
            results = model.get_intermidiate_result_val()
            
            
        else:      
            model.validate()

        val_errors = model.get_current_errors_val()

    if total_steps % opt.display_freq == 0:
        val_viz = model.get_current_visuals_val()
        # model.plot_image_in_tb(tb_writer, val_viz)   # TODO

        val_errors = model.get_current_errors_val()
        model.track_scalar_in_tb(tb_writer, val_errors, total_steps)


def log_image_table(images, predicted, labels, probs):
    "Log a wandb.Table with (img, pred, target, scores)"
    # 🐝 Create a wandb Table to log images, labels and predictions to
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(10)])
    for img, pred, targ, prob in zip(images.to("cpu"), predicted.to("cpu"), labels.to("cpu"), probs.to("cpu")):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *prob.numpy())

    wandb.log({"predictions_table":table}, commit=False)

def test_one_epoch(model, test_tgt_loader, test_src_loader, test_scan_info,
                   opt, epoch, total_steps, label_name):
    t0 = time.time()
    print('infering the model at the end of epoch %d, iters %d' %
          (epoch, total_steps))

    with torch.no_grad():
        print(f'Starting inferring ... ')
        preds, dsc_table, error_dict, domain_list = prediction_wrapper(model, test_tgt_loader, 
                                                                       opt, epoch, label_name,
                                                                       save_prediction=True, tag="target")
        
        # wandb.summary
        wandb.log({
            "rawDiceTarget": dsc_table.tolist(),
            'meanDiceTarget': error_dict['overall'],
            'meanDiceAvgTargetDomains': error_dict['overall_by_domain']
        })


        for _dm in domain_list:
            wandb.log({f'meanDice_{_dm}': error_dict[f'domain_{_dm}_overall'],
                       f'rawDice_{_dm}': error_dict[f'domain_{_dm}_table'].tolist()
                       })


        print('test for source domain as a reference')
        _, dsc_table, error_dict, _ = prediction_wrapper(model, test_src_loader, opt, epoch, label_name,
                                                         save_prediction=True, tag="val")
        wandb.log({'source_rawDice': dsc_table.tolist(),
                   'source_meanDice': error_dict['overall']
                   })

        # pred_dir = os.path.join("result", , opt.experiment, 'pred_dir')
        # os.makedirs(pred_dir, exist_ok=True)

        image, comp = visualization_wrapper(model, test_tgt_loader, opt, epoch, label_name,
                                      save_prediction=True, tag="target")
        image_name = os.path.join(opt.image_dir, f"epoch_{epoch}_target.png")
        plt.imsave(image_name, image)
        
        image_name = os.path.join(opt.image_dir, f"epoch_{epoch}_target_comp.png")
        plt.imsave(image_name, comp)
        
        print("=== vis image save:", image_name)
        
        image, comp = visualization_wrapper(model, test_src_loader, opt, epoch, label_name,
                                      save_prediction=True, tag="target")
        image_name = os.path.join(opt.image_dir, f"epoch_{epoch}_source.png")
        plt.imsave(image_name, image)
        image_name = os.path.join(opt.image_dir, f"epoch_{epoch}_source_comp.png")
        plt.imsave(image_name, comp)
        print("=== vis image save:", image_name)
        
        
        t1 = time.time()
        print("End of model inference, which takes {} seconds".format(t1 - t0))

