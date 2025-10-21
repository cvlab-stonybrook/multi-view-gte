import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import pdb
from data.dataset_multiview import Gaze_Dataset_Multiview_CrossView
from utils.file_utils import load_pretrained_weights
from model.model_transformer import Transformer_fov_cat_CrossView
from utils.evaluation import euclid_dist, ap, metric_avg_best
from utils.log_utils import get_logger
from tqdm import tqdm


def evaluate(args):
    """
    Evaluate cross-view gaze estimation using a Transformer model.
    """
    print(args)
    
    # Configuration constants
    base_dir = args.base_dir
    img_out_size = args.image_size
    head_out_shape = (224, 224)
    hm_size = (64, 64)
    
    # Load cross-view dataset
    val_dataset = Gaze_Dataset_Multiview_CrossView(
        base_dir, img_out_size, head_out_shape, hm_size, 
        eval_scene=args.test_scene, test=True
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=14)

    # Setup logging
    setting_name = f'model_{args.model}_test{args.test_scene}'
    if args.remark:
        setting_name = f'{args.remark}_{setting_name}'
    
    log_dir = os.path.join('./logs', args.project_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'{setting_name}.log')
    
    # Load transformer model
    model = Transformer_fov_cat_CrossView(
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
    load_pretrained_weights(model, weight_path = args.init_weights)
    # Initialize result containers
    dist_all = []          # Euclidean distances
    ang_err_list = []      # Angular errors (not used in this version)
    vis_pred_all = []      # Predicted visibility scores
    vis_gt_all = []        # Ground truth visibility labels
    head_valid_other = []  # Head validity in other view (not used here)
    main_info_inout = []   # Main view indices for visibility evaluation
    main_info_gf = []      # Main view indices for gaze following evaluation
    
    # Setup model for evaluation
    model.cuda()
    model.eval()
    
    logger = get_logger(log_path)
    num_image_groups = len(val_dataset) // 5  # Each group has 5 views
    logger.info(f"Num image groups: {num_image_groups}")
    logger.info(f"Evaluating {args.model}")
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(val_loader)):
            # Unpack batch data
            data = batch['data']
            img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT, quat = data
            fund_mat = batch['fund_mat']
            abs_depth = batch['abs_depth']
            main_view = batch['main_id']
            
            bs, num_views = img.size()[:2]
            loss_mask = torch.logical_and(visib == 1, head_valid == 1)
            
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
            abs_depth = abs_depth.cuda()
            
            # Transformer forward pass
            model_outputs = model((img, head_img, head_mask, depth, abs_depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
            hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = model_outputs
            
            # Process predictions
            hm_pred = hm_pred.squeeze(1)
            vis_pred = vis_pred.view(-1)
            
            # Apply sigmoid to visibility predictions
            vis_pred = torch.sigmoid(vis_pred)
            
            # Move to CPU for metric computation
            hm_pred = hm_pred.cpu().numpy()
            vis_pred = vis_pred.cpu().numpy()
            
            # Extract primary view data
            gaze_heatmap = gaze_heatmap[:, 0]
            gaze_coords = gaze_coords[:, 0]
            gaze_coords = gaze_coords.cpu().numpy()
            
            visib = visib[:, 0]
            head_valid = head_valid[:, 0]
            loss_mask = loss_mask[:, 0]
            
            # Valid mask for gaze following (only when visible)
            valid_mask = visib == 1
            
            # Compute gaze following metrics
            dist_list = euclid_dist(hm_pred, gaze_coords, valid_mask, with_holder=True)
            
            # Handle occlusion (label 2) by treating as visible (label 1)
            occ_mask = visib == 2
            visib[occ_mask] = 1
            
            # Accumulate results
            vis_gt = visib.cpu().numpy().tolist()
            dist_all += dist_list
            vis_pred_all += vis_pred.tolist()
            vis_gt_all += vis_gt
            
 
            
            # Record main view indices for per-view evaluation
            main_view_idx = main_view.cpu().numpy().tolist()
            main_info_inout += main_view_idx
            main_info_gf += main_view_idx

    # Convert accumulated results to numpy arrays
    dist_all = np.array(dist_all)
    vis_pred_all = np.array(vis_pred_all)
    vis_gt_all = np.array(vis_gt_all)
    img_info_inout = np.array(main_info_inout)
    img_info_gf = np.array(main_info_gf)
    
    # Filter out invalid samples (marked with -1)
    valid_gf_mask = dist_all != -1
    valid_inout_mask = vis_pred_all != -1
    
    dist_all = dist_all[valid_gf_mask]
    vis_pred_all = vis_pred_all[valid_inout_mask]
    vis_gt_all = vis_gt_all[valid_inout_mask]
    img_info_inout = img_info_inout[valid_inout_mask]
    img_info_gf = img_info_gf[valid_gf_mask]
    
    # Compute overall metrics (averaging across all view pairs)
    dist_avg_all, dist_best_all, _ = metric_avg_best(
        dist_all, img_info_gf, gt_list=None, is_auc=False, is_dist=True, is_ap=False
    )
    vis_avg_all, vis_best_all, vis_gt_unique = metric_avg_best(
        vis_pred_all, img_info_inout, gt_list=vis_gt_all, is_auc=False, is_dist=False, is_ap=True
    )
    
    # Compute final average scores
    dist_avg = np.mean(dist_avg_all)
    dist_best = np.mean(dist_best_all)
    
    vis_gt_unique = np.array(vis_gt_unique).astype(np.int32)
    ap_avg = ap(vis_avg_all, vis_gt_unique)
    ap_best = ap(vis_best_all, vis_gt_unique)
    
    logger.info(f"Evaluation for {args.test_scene}:")
    logger.info(f"  Dist: {dist_avg:.3f} (best: {dist_best:.3f})")
    logger.info(f"  AP (avg): {ap_avg:.3f}, AP (best): {ap_best:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=20, help='in the unit of dual views, so the real batch size is 40')
    parser.add_argument("--image_size", type=int, nargs='+', required=True, help='input size of image to the model')
    parser.add_argument("--remark", default='')
    parser.add_argument("--model", default='transformer')
    parser.add_argument('--not_use_var', dest='use_var', action='store_false')  
    parser.add_argument("--test_scene", type=str, default='', help='scene for validation')
    parser.add_argument("--project_name", default='Eval_Crossview')
    parser.add_argument('--fov_thres', type=float, default=0.9, help='threshold for fov heatmap')
    parser.add_argument('--alpha', type=float, default=10.0, help='weight for heatmap loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for visibility loss')
    parser.add_argument('--dir_weight', type=float, default=0.1, help='weight for gaze direction loss')
    parser.add_argument("--no_epi_attn", action='store_false', dest='use_epi_attn')
    parser.add_argument("--no_select", action='store_false', dest='use_select') 
    parser.add_argument("--simtype", type=str, default='softmax')
    parser.add_argument("--init_weights", default='')
    parser.add_argument('--base_dir', default='/data/add_disk0/qiaomu/datasets/gaze/Multiview_Gaze')
    parser.add_argument('--epipolar_sample', type=int, default=48)
    parser.add_argument("--device", type=str, default='0')
    opt = parser.parse_args()
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    evaluate(opt)

