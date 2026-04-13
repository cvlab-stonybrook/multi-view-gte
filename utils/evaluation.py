import torch
import math
import numpy as np
from . import utils
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score


def L2_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# # Metric functions
def euclid_dist(pred,target, valid_mask, with_holder=False):

    batch_dist_list = []
    batch_size=pred.shape[0]
    pred_H,pred_W=pred.shape[1:]

    for b_idx in range(batch_size):
        if valid_mask[b_idx]:
            pred_x,pred_y=utils.argmax_pts(pred[b_idx])
            norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])
            sample_target=target[b_idx].reshape(-1)
            sample_dist=np.linalg.norm(sample_target-norm_p)
            batch_dist_list.append(sample_dist)
        else:
            if with_holder:
                batch_dist_list.append(-1)

    return batch_dist_list

def euc_dist_single(pred, target):
    pred_x,pred_y=utils.argmax_pts(pred)
    pred_H,pred_W=pred.shape
    norm_p=np.array([pred_x,pred_y])/np.array([pred_W,pred_H])
    sample_dist=np.linalg.norm(target-norm_p)
    return sample_dist


def auc(gt_gaze, pred_heatmap, hm_size, valid_mask, with_holder=False):
    # use -1 for place holder for invalid samples
    batch_size=len(gt_gaze)
    auc_score_list=[]
    for b_idx in range(batch_size):        
        if valid_mask[b_idx]:
            binary_hm = utils.hm2binary(gt_gaze[b_idx], hm_size)
            scaled_heatmap = resize(pred_heatmap[b_idx], (hm_size[1], hm_size[0]))
            sample_auc_score=roc_auc_score(np.reshape(binary_hm,binary_hm.size),
                            np.reshape(scaled_heatmap,scaled_heatmap.size))
            auc_score_list.append(sample_auc_score)
        else:
            if with_holder:
                auc_score_list.append(-1)
            
    return auc_score_list

def auc_single(heatmap, gt_gaze, is_im=True, hm_size=(64,64)):
    
    binary_hm = utils.hm2binary(gt_gaze, hm_size)
    scaled_heatmap = resize(heatmap, (hm_size[1], hm_size[0]))
    if is_im:
        auc_score = roc_auc_score(np.reshape(binary_hm,binary_hm.size), np.reshape(scaled_heatmap,scaled_heatmap.size))
    else:
        auc_score = roc_auc_score(binary_hm, scaled_heatmap)
    return auc_score


def eval_classify(pred, label):
    pred_labels = np.argmax(pred, axis=-1) 
    label = label.astype(int)
    return np.array(label==pred_labels).astype(int)
    #return list(average_precision_score(label,pred, average=None))


def ap(pred_logits, gt_label):
    # gt_label need to be binary label
    return average_precision_score(gt_label, pred_logits)

def precision(pred_label, gt_label):
    # gt_label need to be binary label
    return precision_score(gt_label, pred_label, average='binary')

def recall(pred_label, gt_label):
    # gt_label need to be binary label
    return recall_score(gt_label, pred_label, average='binary')

def metric_cls(pred_label, gt_label, cls_idx, metric='precision'):
    binary_pred, binary_gt = pred_label.copy(), gt_label.copy()
    pred_mask, gt_mask = pred_label==cls_idx, gt_label==cls_idx
    if gt_mask.sum()==0:
        return 0.0
    binary_pred[pred_mask] = 1
    binary_pred[~pred_mask] = 0
    binary_gt[gt_mask] = 1
    binary_gt[~gt_mask] = 0
    if metric=='precision':
        score = precision(binary_pred, binary_gt)
    elif metric=='recall':
        score = recall(binary_pred, binary_gt)
        
    return score

def metric_avg_best(scores_all, main_id, gt_list=None, is_auc=False, is_dist=False, is_ap=False, print_view=False):
    # compute the average and best score for pairs cooresponding to the same main view (according to main_id)
    # 5 pairs for each main view
    metric_avg, metric_best = [], []
    #gt_unique = [gt_list[0]] if gt_list is not None and len(gt_list)>0 else []
    gt_unique = []
    if len(scores_all)==0:
        return metric_avg, metric_best, gt_unique
    
    assert len(scores_all)==len(main_id), f'{len(scores_all)} != {len(main_id)}'    
    idx_last = 0
    this_id = main_id[0]
    
    for i in range(len(scores_all)+1):
        # In evaluation the pairs are already aggregated by main view, so we directly find the index when the main view changes
        if i==len(scores_all) or main_id[i]!=this_id:
            this_scores = scores_all[idx_last:i]
            if print_view:
                print("Main view id", main_id[idx_last:i])
            if not is_ap:
                metric_avg.append(np.mean(this_scores))
                if is_auc:
                    metric_best.append(np.max(this_scores))
                elif is_dist:
                    metric_best.append(np.min(this_scores))
            else:
                gt_this = gt_list[i-1]
                metric_avg.append(np.mean(this_scores))
                if gt_this == 1:
                    metric_best.append(np.max(this_scores))
                else:
                    metric_best.append(np.min(this_scores))
                if gt_list is not None:
                    gt_unique.append(gt_this)    
                
            if i<len(scores_all):
                this_id = main_id[i]
                idx_last = i
                
    return metric_avg, metric_best, gt_unique


def metric_multipairs(scores_all, main_id, pred_var, num_pairs, gt_list=None, is_auc=False, is_dist=False, is_ap=False, print_view=False):
    # compute the average and best score for pairs cooresponding to the same main view (according to main_id)
    # 5 pairs for each main view
    metric_avg, metric_best = [], []
    #gt_unique = [gt_list[0]] if gt_list is not None and len(gt_list)>0 else []
    gt_unique = []
    if len(scores_all)==0:
        return metric_avg, metric_best, gt_unique
    
    assert len(scores_all)==len(main_id), f'{len(scores_all)} != {len(main_id)}'    
    idx_last = 0
    this_id = main_id[0]
    
    for i in range(len(scores_all)+1):
        if i==len(scores_all) or main_id[i]!=this_id:
            this_scores = scores_all[idx_last:i]
            eval_index = np.arange(0, len(this_scores))
            eval_scores = []
            if num_pairs==2:
                scores_eval = np.mean(this_scores)
                eval_scores.append(scores_eval)
            elif num_pairs<5:
                for eval_time in range(5):
                    sampled_index = np.random.choice(eval_index, num_pairs, replace=False)
                    scores_eval = this_scores[sampled_index]
                    var_eval = pred_var[idx_last:i][sampled_index]
                    var_min_idx = np.argmin(var_eval)
                    eval_scores.append(scores_eval[var_min_idx])
            else:
                var_min_idx = np.argmin(pred_var[idx_last:i])
                eval_scores.append(this_scores[var_min_idx])
            eval_scores = np.array(eval_scores)
            this_scores = eval_scores
            if print_view:
                print("Main view id", main_id[idx_last:i])
            if not is_ap:
                metric_avg.append(np.mean(this_scores))
                if is_auc:
                    metric_best.append(np.max(this_scores))
                elif is_dist:
                    metric_best.append(np.min(this_scores))
            else:
                gt_this = gt_list[i-1]
                metric_avg.append(np.mean(this_scores))
                if gt_this == 1:
                    metric_best.append(np.max(this_scores))
                else:
                    metric_best.append(np.min(this_scores))
                if gt_list is not None:
                    gt_unique.append(gt_this)    
                
            if i<len(scores_all):
                this_id = main_id[i]
                idx_last = i
                
    #if gt_list is not None:
    #    del gt_unique[-1]
    #del metric_avg[-1]
    #del metric_best[-1]
    return metric_avg, metric_best, gt_unique


def metric_from_head(scores_all, main_id, head_other, gt_list=None, print_view=False):
    # compute the average and best score for pairs cooresponding to the same main view (according to main_id)
    # 5 pairs for each main view
    metric_head, metric_nohead = [], []  # compute separately for whether the head is visible in the other view
    assert len(scores_all)==len(main_id), f'{len(scores_all)} != {len(main_id)}'    
    idx_last = 0
    this_id = main_id[0]
    
    gt_unique_head, gt_unique_nohead = [],[]
    for i in range(len(scores_all)+1):
        if i==len(scores_all) or main_id[i]!=this_id:
            this_scores = scores_all[idx_last:i]
            head_other_this = head_other[idx_last:i]
            if print_view:
                print("Main view id", main_id[idx_last:i])
        
            if np.sum(head_other_this)>0:
                metric_head.append(np.mean(this_scores[head_other_this]))
            if np.sum(head_other_this)<len(head_other_this):
                metric_nohead.append(np.mean(this_scores[~head_other_this]))

            if gt_list is not None:
                if np.sum(head_other_this)>0:
                    gt_unique_head.append(gt_list[i-1])
                if np.sum(head_other_this)<len(head_other_this):
                    gt_unique_nohead.append(gt_list[i-1])
            
            if i<len(scores_all):
                this_id = main_id[i]
                idx_last = i
    if len(metric_head)==0:
        metric_head = [-1]
        
    if len(metric_nohead)==0:
        metric_nohead = [-1]
    
    return metric_head, metric_nohead, gt_unique_head, gt_unique_nohead



def metric_from_facevis(scores_all, main_id, facevis_other, gt_list=None, print_view=False):
    # compute the average and best score for pairs cooresponding to the same main view (according to main_id)
    # 5 pairs for each main view
    
    metric_front, metric_half, metric_back, metric_overall = [],[],[],[]
    assert len(scores_all)==len(main_id), f'{len(scores_all)} != {len(main_id)}'    
    idx_last = 0
    this_id = main_id[0]
    # face vis: 0: fully visible, 1: half visible, -1: not visible, -2: no head in image
    gt_unique_front, gt_unique_half, gt_unique_back, gt_unique_overall = [],[],[],[]
    for i in range(len(scores_all)+1):
        if i==len(scores_all) or main_id[i]!=this_id:
            this_scores = scores_all[idx_last:i]
            facevis_other_this = facevis_other[idx_last:i]
            if print_view:
                print("Main view id", main_id[idx_last:i])

            if np.sum(facevis_other_this==1)>0:
                metric_front.append(np.mean(this_scores[facevis_other_this==1]))
                if gt_list is not None:
                    gt_unique_front.append(gt_list[i-1])
            if np.sum(facevis_other_this==0)>0:
                metric_half.append(np.mean(this_scores[facevis_other_this==0]))
                if gt_list is not None:
                    gt_unique_half.append(gt_list[i-1])
            if np.sum(facevis_other_this==-1)>0:
                metric_back.append(np.mean(this_scores[facevis_other_this==-1]))
                if gt_list is not None:
                    gt_unique_back.append(gt_list[i-1])
            if np.sum(facevis_other_this==-2)< len(facevis_other_this):
                metric_overall.append(np.mean(this_scores[facevis_other_this!=-2]))
                if gt_list is not None:
                    gt_unique_overall.append(gt_list[i-1])            
            
            if i<len(scores_all):
                this_id = main_id[i]
                idx_last = i
                
    #if len(metric_front)==0:
    #    metric_front = [0]
        
    #if len(metric_half)==0:
    #    metric_half = [0]
        
    #if len(metric_back)==0:
    #    metric_back = [0]
    
    #if len(metric_overall)==0:
    #    metric_overall = [0]
    
    return {"metrics": [metric_front, metric_half, metric_back, metric_overall], "gt": [gt_unique_front, gt_unique_half, gt_unique_back, gt_unique_overall]}


def compute_angular_error(input,target, spherical=False, gaze_vec_dim=3):

    if spherical:
        input = utils.spherical2cartesial(input)
        target = utils.spherical2cartesial(target)

    input = input.view(-1,gaze_vec_dim,1)
    target = target.view(-1,1,gaze_vec_dim)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    #output_dot = 180*torch.mean(output_dot)/math.pi
    output_dot = 180*output_dot/math.pi
    
    return output_dot

def compute_angular_error_single(input,target, spherical=False):

    if spherical:
        input = utils.spherical2cartesial(input)
        target = utils.spherical2cartesial(target)

    input = input.reshape(-1)
    target = target.reshape(-1)
    output_dot = np.dot(target,input)
    output_dot = np.arccos(output_dot)
    #output_dot = 180*torch.mean(output_dot)/math.pi
    output_dot = 180*output_dot/math.pi
    
    return output_dot

def get_metrics_all(auc_list, dist_list, vis_pred_list, vis_gt_list, ang_from_tgt_list, main_info_inout, main_info_gf):
    
    auc_all, dist_all = torch.tensor(auc_list), torch.tensor(dist_list)
    vis_pred_all, vis_gt_all = torch.tensor(vis_pred_list), torch.tensor(vis_gt_list)
    img_info_inout, img_info_gf = torch.tensor(main_info_inout), torch.tensor(main_info_gf)
    ang_all, img_info_ang = torch.tensor(ang_from_tgt_list), torch.tensor(main_info_gf)
    
    gf_mask = auc_all!=-1
    inout_mask = vis_pred_all!=-1
    auc_all, dist_all, vis_pred_all, vis_gt_all = auc_all[gf_mask].tolist(), dist_all[gf_mask].tolist(), vis_pred_all[inout_mask].tolist(), vis_gt_all[inout_mask].tolist()
    img_info_inout, img_info_gf = img_info_inout[inout_mask].tolist(), img_info_gf[gf_mask].tolist()
    ang_mask = ang_all!=-1
    ang_all, img_info_ang = ang_all[ang_mask].tolist(), img_info_ang[ang_mask].tolist()
    
     
    auc_avg_all, auc_best_all, _ = metric_avg_best(auc_all, img_info_gf, gt_list=None, is_auc=True, is_dist=False, is_ap=False, print_view=False)
    dist_avg_all, dist_best_all, _ = metric_avg_best(dist_all, img_info_gf, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
    vis_avg_all, vis_best_all, vis_gt_unique = metric_avg_best(vis_pred_all, img_info_inout, gt_list=vis_gt_all, is_auc=False, is_dist=False, is_ap=True)
    ang_avg_all, ang_best_all, _ = metric_avg_best(ang_all, img_info_ang, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
    #print(vis_gt_unique)
    
    auc_avg, auc_best, dist_avg, dist_best = np.mean(auc_avg_all), np.mean(auc_best_all), np.mean(dist_avg_all), np.mean(dist_best_all)
    vis_pred_all, vis_gt_unique = np.array(vis_pred_all), np.array(vis_gt_unique)
    ang_avg, ang_best = np.mean(ang_avg_all), np.mean(ang_best_all)
    vis_gt_unique = vis_gt_unique.astype(np.int32)
    ap_avg = ap(vis_avg_all, vis_gt_unique)
    ap_best = ap(vis_best_all, vis_gt_unique)
    
    return auc_avg, auc_best, dist_avg, dist_best, ap_avg, ap_best, ang_avg, ang_best
    
    
