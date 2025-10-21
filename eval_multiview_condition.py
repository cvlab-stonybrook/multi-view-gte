import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from data.dataset_multiview import Gaze_Dataset_Multiview_CrossScene
from model.model_transformer import Transformer_fov_cat
from utils.file_utils import load_pretrained_weights
from utils.evaluation import euclid_dist, ap, metric_avg_best, compute_angular_error
from utils.log_utils import get_logger
from tqdm import tqdm


def evaluate(args):
    """
    Evaluate multi-view gaze estimation under different conditions.
    """ 
    print(args)
    
    # Configuration constants
    base_dir = args.base_dir
    img_out_size = args.image_size                                                          
    head_out_shape = (224, 224)
    hm_size = (64, 64)
   
    # TODO: Change to the no-pdp version
    val_dataset = Gaze_Dataset_Multiview_CrossScene(base_dir, img_out_size, head_out_shape, hm_size, eval_scene=args.test_scene, test=True, adapt=False)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=14)

    # Setup logging
    setting_name = f'model_{args.model}_test{args.test_scene}'
    if args.remark:
        setting_name = f'{args.remark}_{setting_name}'
    
    log_dir = os.path.join('./logs', args.project_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{setting_name}.log')
    
    # Load multiview-gte model
    model = Transformer_fov_cat(
        args.lr, 
        image_size=img_out_size, 
        alpha=args.alpha, 
        beta=args.beta, 
        dir_weight=args.dir_weight,
        fov_thres=args.fov_thres, 
        use_var=args.use_var, 
        hm_size=hm_size, 
        sample_num=args.epipolar_sample, 
        use_epi_attn=args.use_epi_attn, 
        sim_type=args.simtype, 
        use_select=args.use_select
    )
    
    load_pretrained_weights(model, weight_path=args.init_weights)


    dist_all = []        
    ang_err_list = []     
    vis_pred_all = []    
    vis_gt_all = []       
    
    head_valid_other = []  
    vis_gt_other = []      
    main_info_inout = []   
    main_info_gf = []      
    
    # Setup model for evaluation
    model.cuda()
    model.eval()
    logger = get_logger(log_path)
    logger.info("Num images total: {} ".format(len(val_dataset) // 5))
    eps = 1e-9
    logger.info(f"Evaluating {args.init_weights}")
    
    with torch.no_grad(): 
        for idx, batch in enumerate(tqdm(val_loader)):
            data = batch['data']
            img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT = data
            fund_mat = batch['fund_mat']
            main_view = batch['main_id']
            
            bs, num_views = img.size()[:2]
            loss_mask = torch.logical_and(visib==1, head_valid==1)
            
            # Move all tensors to GPU
            img = img.cuda()
            head_img = head_img.cuda()
            head_mask = head_mask.cuda()
            depth = depth.cuda()
            gaze_heatmap = gaze_heatmap.cuda()
            visib = visib.cuda()
            gaze_coords = gaze_coords.cuda()
            gtvec_3d = gtvec_3d.cuda()
            head_valid = head_valid.cuda()
            eye_loc = eye_loc.cuda()
            head_coords = head_coords.cuda()
            intri = intri.cuda()
            RT = RT.cuda()
            fund_mat = fund_mat.cuda()
            
            # Transformer forward pass
            model_outputs = model((img, head_img, head_mask, depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
            hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = model_outputs
                
            gaze_heatmap, gaze_coords = gaze_heatmap[:, 0], gaze_coords[:, 0] 
            num_samples = img.size(0)
            
            head_other, vis_other = head_valid[:, 1], visib[:, 1]
            visib, head_valid, loss_mask = visib[:, 0], head_valid[:, 0], loss_mask[:, 0]  # only evaluate on the primary view
            valid_mask = torch.logical_and(visib == 1, head_valid == 1) 
            gaze_coords = gaze_coords.cpu().numpy() 
        
            hm_pred = hm_pred.squeeze(1)
            hm_pred, vis_pred = hm_pred.view(bs, num_views, *hm_pred.size()[1:]), vis_pred.view(bs, num_views)
            hm_pred, vis_pred = hm_pred[:, 0], vis_pred[:, 0]
            occ_mask = visib == 2
            visib[occ_mask] = 1  
            vis_pred = vis_pred.view(-1)
            head_valid_mask = head_valid == 1        
                        
            vis_pred = torch.sigmoid(vis_pred)
            vis_pred[~head_valid_mask] = -1
            hm_pred, vis_pred = hm_pred.cpu().numpy(), vis_pred.cpu().numpy().tolist()
            
            dist_list = euclid_dist(hm_pred, gaze_coords, valid_mask, with_holder=True)
            
            # Angular error
            gtvec_3d = gtvec_3d[:, 0]
            gaze_vec = gaze_vec.view(-1, num_views, 3)
            gaze_vec = gaze_vec[:, 0]
            vec_valid_mask = torch.logical_and(gtvec_3d[:, 0] == 0, gtvec_3d[:, 1] == 0)
            vec_valid_mask = ~vec_valid_mask
            vec_valid_mask = torch.logical_and(vec_valid_mask, head_valid.bool())
            ang_err_val = torch.ones(num_samples).to(gaze_vec) * -1
            ang_err = compute_angular_error(gaze_vec, gtvec_3d, spherical=False)
            ang_err_val[vec_valid_mask] = ang_err[vec_valid_mask]
            ang_err_list += ang_err_val.tolist()
            
            vis_gt = visib.cpu().numpy().tolist()
            dist_all += dist_list
            vis_pred_all += vis_pred 
            vis_gt_all += vis_gt
            head_valid_other += head_other.tolist()
            vis_gt_other += vis_other.tolist()
            
            # Record which images are used, for evaluation based on the same main view
            img_idx_inout = main_view.cpu().numpy().tolist()
            img_idx_gf = main_view.cpu().numpy().tolist()  
            main_info_inout += img_idx_inout
            main_info_gf += img_idx_gf 

    # Evaluate
    dist_all = np.array(dist_all)
    vis_pred_all, vis_gt_all = np.array(vis_pred_all), np.array(vis_gt_all)
    img_info_inout, img_info_gf = np.array(main_info_inout), np.array(main_info_gf)
    ang_all, img_info_ang = np.array(ang_err_list), np.array(main_info_gf)
    head_valid_other, vis_gt_other = np.array(head_valid_other), np.array(vis_gt_other)
    
    gf_mask = dist_all != -1
    inout_mask = vis_pred_all != -1
    dist_all, vis_pred_all, vis_gt_all = dist_all[gf_mask], vis_pred_all[inout_mask], vis_gt_all[inout_mask]
    img_info_inout, img_info_gf = img_info_inout[inout_mask], img_info_gf[gf_mask]
    head_valid_gf, head_valid_inout = head_valid_other[gf_mask], head_valid_other[inout_mask]   
    vis_other_gf, vis_other_inout = vis_gt_other[gf_mask], vis_gt_other[inout_mask] 
    
    logger.info("Total evaluation samples: {}, for inout: {}".format(gf_mask.sum() / 5, inout_mask.sum() / 5))
    
    ang_mask = ang_all != -1
    ang_all, img_info_ang = ang_all[ang_mask], img_info_ang[ang_mask]
    head_valid_ang = head_valid_other[ang_mask]
    vis_other_ang = vis_gt_other[ang_mask]
    
    # Filter images based on condition
    dist_info, ap_info, ang_info = {}, {}, {}
    
    # Evaluate by condition
    for head_query in [0, 1]:
        for vis_query in [0, 1]:
            mask_gf = np.bitwise_and(head_valid_gf == head_query, vis_other_gf == vis_query)
            mask_inout = np.bitwise_and(head_valid_inout == head_query, vis_other_inout == vis_query)
            main_gf_this, main_inout_this = img_info_gf[mask_gf], img_info_inout[mask_inout]
            dist_this, vis_pred_this, vis_gt_this = dist_all[mask_gf], vis_pred_all[mask_inout], vis_gt_all[mask_inout]
            
            mask_ang = np.bitwise_and(head_valid_ang == head_query, vis_other_ang == vis_query)
            main_ang_this = img_info_ang[mask_ang]
            ang_this = ang_all[mask_ang]
            ang_avg, ang_best, _ = metric_avg_best(ang_this, main_ang_this, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            
            dist_avg, dist_best, _ = metric_avg_best(dist_this, main_gf_this, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            vis_avg, vis_best, vis_gt = metric_avg_best(vis_pred_this, main_inout_this, gt_list=vis_gt_this, is_auc=False, is_dist=False, is_ap=True)
            
            num_gf, num_inout = len(dist_avg), len(vis_avg)
            ap_avg = ap(vis_avg, vis_gt) if len(vis_avg) > 0 else 0
            ap_best = ap(vis_best, vis_gt) if len(vis_avg) > 0 else 0
            logger.info(f'Head query: {head_query}, Target query:{vis_query} num_gf: {num_gf}, num_inout: {num_inout}, num_ang:{len(ang_avg)}')
            dist_avg_score = np.mean(dist_avg) if len(dist_avg) > 0 else 0
            dist_best_score = np.mean(dist_best) if len(dist_best) > 0 else 0
            ang_avg_score = np.mean(ang_avg) if len(ang_avg) > 0 else 0
            ang_best_score = np.mean(ang_best) if len(ang_best) > 0 else 0
            
            logger.info("Dist {:.3f}, AP {:.3f}, Ang {:.3f}".format(dist_avg_score, ap_avg, ang_avg_score))
            
            dist_info[(head_query, vis_query)] = (dist_avg_score, num_gf)
            ap_info[(head_query, vis_query)] = (ap_avg, num_inout)
            ang_info[(head_query, vis_query)] = (ang_avg_score, len(ang_avg))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20, help='in the unit of dual views, so the real batch size is 40')
    parser.add_argument("--image_size", type=int, nargs='+', required=True, help='input size of image to the model')
    parser.add_argument("--remark", default='')
    parser.add_argument("--model", default='transformer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay for AdamW')
    parser.add_argument('--not_use_var', dest='use_var', action='store_false')  
    parser.add_argument("--test_scene", type=str, default='', help='scene for validation')
    parser.add_argument("--init_weights", default='')
    parser.add_argument("--project_name", default='Eval_Multiview')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument('--fov_thres', type=float, default=0.9, help='threshold for fov heatmap in ChildPlay')
    parser.add_argument('--alpha', type=float, default=10.0, help='weight for heatmap loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for visibility loss')
    parser.add_argument('--dir_weight', type=float, default=0.1, help='weight for gaze direction loss')
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--log_dir", type=str, default='./logs')
    parser.add_argument("--no_epi_attn", action='store_false', dest='use_epi_attn')
    parser.add_argument("--no_select", action='store_false', dest='use_select') 
    parser.add_argument("--simtype", type=str, default='softmax') 
    parser.add_argument('--base_dir', default='/data/add_disk0/qiaomu/datasets/gaze/Multiview_Gaze')
    parser.add_argument("--ckpt_dir", type=str, default="/nfs/bigrod/add_disk0/qiaomu/ckpts/gaze/multiview_gaze")
    parser.add_argument('--epipolar_sample', type=int, default=48)
    parser.add_argument("--device", type=str, default='0')
    opt = parser.parse_args()
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    evaluate(opt)

