import torch
import torch.nn as nn
from torch.nn import functional as F

eps = 1e-11
def heatmap_loss(hm_pred, hm_gt, loss_mask):
    loss_apply_idx = loss_mask.float()  # for now, only for true label, ignore self-occlusion
    l2_loss = F.mse_loss(hm_pred, hm_gt, reduction='none')
    l2_loss = l2_loss.mean(dim=-1).mean(dim=-1)
    l2_loss = (l2_loss * loss_apply_idx).sum() / (loss_apply_idx.sum() + eps)
    
    return l2_loss

def gaze_dir_loss(gaze_vec, gt_vec, loss_mask):
    loss_apply_idx = loss_mask.float()
    inner_prod = torch.sum(gaze_vec * gt_vec, dim=1)
    dir_loss = 1 - inner_prod
    dir_loss = (dir_loss * loss_apply_idx).sum() / (loss_apply_idx.sum() + eps)    
    return dir_loss

def dir_variance_loss(pred_dir, gt_dir, pred_var, loss_mask, loss_coef=2.0):
    # from the Aleatoric Uncertainty loss function published in "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
    
    COS_ = (-torch.sum(pred_dir * gt_dir, dim=1)+1)/2
    loss = loss_coef * (COS_*torch.exp(-pred_var))/2 + (pred_var/2)
    #loss = loss.mean()
    loss_apply_idx = loss_mask.float()    
    loss = (loss * loss_apply_idx).sum() / (loss_apply_idx.sum() + eps)        
    
    return loss



class KL_div_modified(nn.Module):
    def __init__(self, epsilon=2.2204e-16, reduction='batchmean'):
        # eps value is adopted from the paper "What do different evaluation metrics tell us about saliency models" which says in the MIT saliency benchmark that eps is 2.2204e-16
        super(KL_div_modified, self).__init__()
        self.eps = epsilon
        self.reduction=reduction
    def forward(self, input, target):
        kl_div = target * (torch.log(self.eps+torch.divide(target, input+self.eps)))
        if self.reduction=='batchmean':
            kl_div = torch.mean(kl_div.sum(dim=1))
        elif self.reduction=='sum':
            kl_div = torch.sum(kl_div)
        elif self.reduction=='none':
            kl_div = kl_div
        return kl_div
