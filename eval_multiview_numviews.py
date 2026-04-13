'''
Evaluate the model with different number of views in multi-view setting
Example usage: python eval_multiview_numviews.py --device 1 --test_scene {scene_name} --num_pairs {num_pairs} --init_weights {ckpt_path} --image_size 512 384
You can add --save_info to save the prediction results to avoid re-evaluating the same scene.
'''


import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import pdb
import pickle
from data.dataset_multiview import Gaze_Dataset_Multiview_CrossScene, Gaze_Dataset_Multiview_RandomSample
from model.model_transformer import Transformer_fov_cat
from utils.file_utils import load_pretrained_weights
from utils.evaluation import auc, euclid_dist, ap, compute_angular_error, metric_multipairs
from utils.log_utils import get_logger
from tqdm import tqdm


def train(args):
    
    print(args)
    base_dir = args.base_dir
    img_out_size = args.image_size                                                          
    head_out_shape = (224, 224)
    hm_size = (64, 64)
    
    
    val_dataset = Gaze_Dataset_Multiview_CrossScene(base_dir, img_out_size, head_out_shape, hm_size, eval_scene=args.test_scene, test=True, adapt=False)
    
    #val_dataset = Subset(val_dataset, range(0, 30)) 
    val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=False, num_workers=10)

    setting_name = f'_pairs{args.num_pairs}_model_{args.model}_test{args.test_scene}'
    if len(args.remark)>0:
        setting_name = args.remark + '_' + setting_name
    
    os.makedirs(os.path.join('./logs', args.project_name), exist_ok=True)
    log_path = os.path.join('./logs', args.project_name, setting_name+'.log')
    
     
    model = Transformer_fov_cat(args.lr, image_size=img_out_size, alpha=args.alpha, beta=args.beta, dir_weight=args.dir_weight,
                            fov_thres=args.fov_thres, use_var=args.use_var, hm_size=hm_size, 
                            sample_num=args.epipolar_sample, use_epi_attn=args.use_epi_attn, sim_type=args.simtype, use_select=args.use_select)
    load_pretrained_weights(model, weight_path = args.init_weights)
         
    
    auc_all, dist_all, ang_err_list, vis_pred_all, vis_gt_all, head_valid_other, main_info_inout, main_info_gf = [], [], [], [], [], [], [], []
    vis_gt_other = []

    model.cuda()
    model.eval()
    logger = get_logger(log_path)
    logger.info("Num images total: {} ".format(len(val_dataset)//5))
    eps = 1e-9
    logger.info(f"Evaluating {args.init_weights}")
    num_pairs_use = args.num_pairs
    gaze_var_all = []
    save_dir = os.path.join(base_dir, 'results', args.project_name)
    if args.save_info:  
        os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, f'{args.model}_{args.test_scene}.pkl')):
        with open(os.path.join(save_dir, f'{args.model}_{args.test_scene}.pkl'), 'rb') as file:
            info = pickle.load(file)
        auc_all, dist_all, ang_err_list, vis_pred_all, vis_gt_all, head_valid_other, main_info_inout, main_info_gf, gaze_var_all, vis_gt_other = info['auc'], info['dist'], info['ang'], info['vis_pred'], info['vis_gt'], info['head_valid_other'], info['main_info_inout'], info['main_info_gf'], info['gaze_var'], info['vis_gt_other']
        
    else:
        with torch.no_grad(): 
            for idx, batch in enumerate(tqdm(val_loader)):
                data = batch['data']
                img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT  = data
                fund_mat = batch['fund_mat']
                main_view = batch['main_id']
                bs, num_views = img.size()[:2]
                loss_mask = torch.logical_and(visib==1, head_valid==1)
                img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc,head_coords, intri, RT, fund_mat =\
                    img.cuda(), head_img.cuda(), head_mask.cuda(), depth.cuda(), gaze_heatmap.cuda(), visib.cuda(), gaze_coords.cuda(), gtvec_3d.cuda(), head_valid.cuda(), eye_loc.cuda(), head_coords.cuda(), intri.cuda(), RT.cuda(), fund_mat.cuda()
                
                hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = model((img, head_img, head_mask, depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
                    
                
                
                gaze_heatmap, gaze_coords = gaze_heatmap[:,0], gaze_coords[:,0] 
                num_samples = img.size(0)
                #visib, head_valid = visib.view(-1), head_valid.view(-1)
                
                head_other, vis_other = head_valid[:,1], visib[:,1]
                visib, head_valid, loss_mask = visib[:,0], head_valid[:,0], loss_mask[:,0]
                valid_mask = torch.logical_and(visib==1, head_valid==1) 
                gaze_coords = gaze_coords.cpu().numpy() 
            
                hm_pred = hm_pred.squeeze(1)
                hm_pred, vis_pred = hm_pred.view(bs, num_views, *hm_pred.size()[1:]), vis_pred.view(bs, num_views)
                hm_pred, vis_pred = hm_pred[:,0], vis_pred[:,0]
                occ_mask = visib==2
                visib[occ_mask] = 1  
                vis_pred = vis_pred.view(-1)
                #visib_loss = F.cross_entropy(vis_pred, visib) * self.beta
                head_valid_mask = head_valid==1 
                gaze_var = gaze_var.view(bs, num_views)       
                gaze_var = gaze_var.min(dim=1)[0]  # compute the minimum variance across views for all pairs
                gaze_var = gaze_var.cpu().tolist()
                gaze_var_all += gaze_var
                            
                vis_pred = torch.sigmoid(vis_pred)
                vis_pred[~head_valid_mask] = -1
                hm_pred, vis_pred = hm_pred.cpu().numpy(), vis_pred.cpu().numpy().tolist()
                try:
                    auc_list = auc(gaze_coords, hm_pred, hm_size, valid_mask, with_holder=True)
                except:
                    pdb.set_trace()
                dist_list = euclid_dist(hm_pred, gaze_coords, valid_mask, with_holder=True)
                
                # angular error
                
                gtvec_3d = gtvec_3d[:,0]
                gaze_vec = gaze_vec.view(-1, num_views, 3)
                gaze_vec = gaze_vec[:,0]
                vec_valid_mask = torch.logical_and(gtvec_3d[:,0]==0, gtvec_3d[:,1]==0)
                vec_valid_mask = ~vec_valid_mask
                vec_valid_mask = torch.logical_and(vec_valid_mask, head_valid.bool())
                ang_err_val = torch.ones(num_samples).to(gaze_vec) * -1
                ang_err = compute_angular_error(gaze_vec, gtvec_3d, spherical=False)
                ang_err_val[vec_valid_mask] = ang_err[vec_valid_mask]
                ang_err_list += ang_err_val.tolist()
                
                vis_gt= visib.cpu().numpy().tolist()
                auc_all += auc_list
                dist_all += dist_list
                vis_pred_all += vis_pred 
                vis_gt_all += vis_gt
                head_valid_other += head_other.tolist()
                vis_gt_other += vis_other.tolist()
                
                # record which images are used, for evaluation based on the same main view
                img_idx_inout = main_view.cpu().numpy().tolist()
                img_idx_gf = main_view.cpu().numpy().tolist()  
                main_info_inout += img_idx_inout
                main_info_gf += img_idx_gf 

            if args.save_info:
                info_all = {'auc':auc_all, 'dist':dist_all, 'ang':ang_err_list, 'vis_pred':vis_pred_all, 'vis_gt':vis_gt_all, 'head_valid_other':head_valid_other, 'main_info_inout':main_info_inout, 'main_info_gf':main_info_gf, 'gaze_var':gaze_var_all, 'vis_gt_other':vis_gt_other}
                with open(os.path.join(save_dir, f'{args.model}_{args.test_scene}.pkl'), 'wb') as f: 
                    pickle.dump(info_all, f)
        
        
        
    # evaluate
    auc_all, dist_all = np.array(auc_all), np.array(dist_all)
    vis_pred_all, vis_gt_all = np.array(vis_pred_all), np.array(vis_gt_all)
    img_info_inout, img_info_gf = np.array(main_info_inout), np.array(main_info_gf)
    ang_all, img_info_ang = np.array(ang_err_list), np.array(main_info_gf)
    head_valid_other, vis_gt_other = np.array(head_valid_other), np.array(vis_gt_other)
    gf_mask = auc_all!=-1
    inout_mask = vis_pred_all!=-1
    auc_all, dist_all, vis_pred_all, vis_gt_all = auc_all[gf_mask], dist_all[gf_mask], vis_pred_all[inout_mask], vis_gt_all[inout_mask]
    img_info_inout, img_info_gf = img_info_inout[inout_mask], img_info_gf[gf_mask]
    
    gaze_var_all = np.array(gaze_var_all)
    gaze_var_gf, gaze_var_inout = gaze_var_all[gf_mask], gaze_var_all[inout_mask]
    
    
    logger.info("Total evaluation samples: {}, for inout: {}".format(gf_mask.sum()/5, inout_mask.sum()/5))
    
    ang_mask = ang_all!=-1
    ang_all, img_info_ang = ang_all[ang_mask], img_info_ang[ang_mask]
    gaze_var_ang = gaze_var_all[ang_mask]
    
    auc_avg_all, auc_best_all, _ = metric_multipairs(auc_all, img_info_gf, gaze_var_gf, num_pairs_use, gt_list=None, is_auc=True, is_dist=False, is_ap=False, print_view=False)
    dist_avg_all, dist_best_all, _ = metric_multipairs(dist_all, img_info_gf, gaze_var_gf, num_pairs_use, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
    vis_avg_all, vis_best_all, vis_gt_unique = metric_multipairs(vis_pred_all, img_info_inout, gaze_var_inout, num_pairs_use, gt_list=vis_gt_all, is_auc=False, is_dist=False, is_ap=True)
    ang_avg_all, ang_best_all, _ = metric_multipairs(ang_all, img_info_ang, gaze_var_ang, num_pairs_use, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
    #print(vis_gt_unique)
    
    dist_avg, dist_best = np.mean(dist_avg_all), np.mean(dist_best_all)
    vis_gt_unique = np.array(vis_gt_unique)
    ang_avg, ang_best = np.mean(ang_avg_all), np.mean(ang_best_all)
    vis_gt_unique = vis_gt_unique.astype(np.int32)
    ap_avg = ap(vis_avg_all, vis_gt_unique)
    logger.info("Full Evaluation: Dist {:.3f}, AP {:.3f} Ang {:.3f}".format(dist_avg, ap_avg, ang_avg))
    
    
            
        
        
    
    
    
    

    






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=20, help='in the unit of dual views, so the real batch size is 40')
    parser.add_argument("--image_size", type=int, nargs='+', required=True, help='input size of image to the model')
    parser.add_argument("--sample_num", type=int, default=64)
    parser.add_argument("--remark", default='')
    parser.add_argument("--model", default='transformer')
    parser.add_argument('--save_info', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for AdamW')
    parser.add_argument('--not_use_var', dest='use_var', action='store_false')  
    parser.add_argument("--test_scene", type=str, default='', help='scene for validation')
    parser.add_argument("--gaze_estimator_weights", default='/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/gaze360/gaze360_res18_backbone.pt')
    parser.add_argument("--init_weights", required=True)
    parser.add_argument("--project_name", default='Eval_Multiview')
    parser.add_argument('--fov_thres', type=float, default=0.9, help='threshold for fov heatmap in ChildPlay')
    parser.add_argument('--alpha', type=float, default=10.0, help='weight for heatmap loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for visibility loss')
    parser.add_argument('--dir_weight', type=float, default=0.1, help='weight for gaze direction loss')
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--num_pairs", type=int, default=3)
    parser.add_argument("--split_file", type=str, default='subj_splits_1.csv')
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--no_epi_attn", action='store_false', dest='use_epi_attn')
    parser.add_argument("--no_select", action='store_false', dest='use_select') 
    parser.add_argument("--simtype", type=str, default='softmax') 
    parser.add_argument('--base_dir', default='/data/add_disk0/qiaomu/datasets/gaze/Multiview_Gaze')
    parser.add_argument("--ckpt_dir", type=str,
                        default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze")
    parser.add_argument('--epipolar_sample', type=int, default=48)
    parser.add_argument("--device", type=str, default='0')
    opt = parser.parse_args()
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    train(opt)