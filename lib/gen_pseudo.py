import torch
import numpy as np
from tqdm import tqdm
import os, time, wandb
import SimpleITK as sitk
import copy
from torch.utils.data import DataLoader
import wandb
from torch.autograd import Variable

DICE_Best = {"val":0, "target":0}


def prediction_wrapper(model, test_loader, opt, epoch, label_name, mode = 'base', save_prediction = False, tag="val"):
    """
    A wrapper for the ease of evaluation
    Args:
        model:          Module The network to evalute on
        test_loader:    DataLoader Dataloader for the dataset to test
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
                curr_pred = torch.Tensor(np.zeros( [ nframe,  nx, ny]  ))#.cuda() # nb/nz, nc, nx, ny
                curr_score = torch.Tensor(np.zeros( [ nframe, opt.nclass, nx, ny]  ))#.cuda()
                curr_gth = torch.Tensor(np.zeros( [nframe,  nx, ny]  ))# .cuda()
                curr_uncertainty = torch.Tensor(np.zeros( [ nframe, opt.nclass, nx, ny]  ))#.cuda() # nb/nz, nc, nx, ny
                
                curr_img = np.zeros( [nx, ny, nframe]  )
                
                # Save

            assert batch['lb'].shape[0] == 1 # enforce a batchsize of 1

            # Old:
            model.set_input(batch)
            # gth, pred = model.get_segmentation_gpu(raw_logits = False)
            gth, pred_score, pred, uncertainty = model.forward_with_uncertainty(batch) # 1b per set
            # [1, 1, 192, 192], [1, 5, 192, 192], [1, 192, 192], [1, 5, 192, 192]
            
            # new
            # print(f"forward_with_un: gth={gth.shape}, pred={pred.shape}, uncertainty={uncertainty.shape}")
            curr_pred[slice_idx, ...]   = pred[0, ...] # nb (1), nc, nx, ny
            curr_score[slice_idx, ...]  = pred_score[0, ...] 
            curr_gth[slice_idx, ...]    = gth[0, ...]
            curr_uncertainty[slice_idx, ...]    = uncertainty[0, ...]
            
            curr_img[:,:,slice_idx] = batch['img'][0, 1,...].numpy()
            slice_idx += 1
            if batch['is_end']:  # End of the image slice
                out_prediction_list[scan_id_full]['pred'] = curr_pred
                out_prediction_list[scan_id_full]['gth'] = curr_gth
                if opt.phase == 'test':
                    recomp_img_list.append(curr_img)
                    
                file_id = str(batch['file_id'][0])  # img/img_10.nii.gz
                save_file = file_id.replace("/img/", "/seg/").replace("/img_", '/pseudo_')
                print("=== pseudo save to:", save_file)
                print("=== curr_pred stat:", curr_pred.shape, curr_pred.max(), curr_pred.min())
                print("=== curr_score stat:", curr_score.shape, curr_score.max(), curr_score.min())
                print("=== curr_uncertainty stat:", curr_uncertainty.shape, curr_uncertainty.max(), curr_uncertainty.min())
                
                np.save(save_file, {"pseudo": curr_pred.cpu().numpy(), 
                                    "score":curr_score.cpu().numpy() , 
                                    "uncertainty": curr_uncertainty.cpu().numpy()})
                

        print("Epoch {} test result on mode {} segmentation are shown as follows:".format(epoch, mode))
        error_dict, dsc_table, domain_names = eval_list_wrapper(out_prediction_list, len(label_name ), model, label_name, tag=tag)
        error_dict["mode"] = mode
        if not save_prediction: # to save memory
            del out_prediction_list
            out_prediction_list = []
        torch.cuda.empty_cache()

    return out_prediction_list, dsc_table, error_dict, domain_names

def eval_list_wrapper(vol_list, nclass, model, label_name, tag = "val"):
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
            tables_by_domain[domain] = {'scores': [], 'scan_ids': []}
            
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
        print("Organ {} with dice: mean: {:06.5f} \n, std: {:06.5f}".format(label_name[organ], mean_dc, std_dc))
        error_dict[label_name[organ]] = mean_dc

    d_mean = dsc_table[:,1:].mean()
    DICE_Best[tag] = np.maximum(d_mean, DICE_Best[tag])
    
    print("Overall mean dice by sample {:06.5f}, current best {:06.5f}".format( dsc_table[:,1:].mean(), DICE_Best[tag])) # background is noted as class 0 and therefore not counted
    error_dict['overall'] = dsc_table[:,1:].mean()
    

    
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





def pseudo_one_epoch(model, test_tgt_loader, test_src_loader, test_scan_info,
                   opt, epoch, total_steps, label_name):
    t0 = time.time()
    print('infering the model at the end of epoch %d, iters %d' %
          (epoch, total_steps))
    with torch.no_grad():
        print(f'Starting inferring ... ')
        preds, dsc_table, error_dict, domain_list = prediction_wrapper(model, test_tgt_loader, opt, epoch, label_name,
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

        for scan_id, comp in preds.items():
            _pred = comp['pred']

            itk_pred = sitk.GetImageFromArray(_pred.cpu().numpy())
            itk_pred.SetSpacing(test_scan_info[scan_id]["spacing"])
            itk_pred.SetOrigin(test_scan_info[scan_id]["origin"])
            itk_pred.SetDirection(test_scan_info[scan_id]["direction"])

            fid = os.path.join(opt.pred_dir, f'pred_{scan_id}_epoch_{epoch}.nii.gz')
            sitk.WriteImage(itk_pred, fid, True)
            # print('# {fid} has been saved #')

        t1 = time.time()
        print("End of model inference, which takes {} seconds".format(t1 - t0))

