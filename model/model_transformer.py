import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import pytorch_lightning as pl
from model.gaze_estimator import GazeEstimator_Res18_pair, GazeEstimator_Res18_CrossAtt
from utils.utils import spherical2cartesial, get_fov_hm, get_fov_hm_crossview
from model import models_vit
from model.gaze_decoder import CrossAttention_Layer, GazeTransformer
from model.transformer_heads import Prediction_Head
from model.position_encoding import PositionalEncoding2D
from einops import rearrange
from model.loss import heatmap_loss, gaze_dir_loss, dir_variance_loss
from utils.utils import get_fov_hm, spherical2cartesial, get_gt_gaze_from_other
from utils.evaluation import euclid_dist, ap, metric_avg_best, compute_angular_error, metric_from_head


class Transformer_fov_cat(pl.LightningModule):
    def __init__(self, lr, image_size, alpha, beta, dir_weight, fov_thres=0.9, num_decoder_layers=1, use_var=False, hm_size=(64,64), sample_num=64, use_epi_attn=True, use_select=True, sim_type='softmax', freeze_gaze_backbone=False):
        super(Transformer_fov_cat, self).__init__()
        self.lr = lr
        self.image_size = image_size    
        self.alpha = alpha
        self.beta = beta
        self.dir_weight = dir_weight
        self.fov_thres = fov_thres
        self.use_var = use_var 
        self.use_epi_attn = use_epi_attn
        self.use_select=use_select
        self.freeze_gaze_backbone = freeze_gaze_backbone
        print("EpiAttn: ", self.use_epi_attn, "Select: ", self.use_select)
        
        if self.use_var:
            print("Use Var")
            self.gaze_estimator = GazeEstimator_Res18_CrossAtt(lr=lr, use_var=use_var)
        else:
            print("No Var")
            self.gaze_estimator = GazeEstimator_Res18_pair(lr=lr, use_var=use_var)
        if not self.use_epi_attn:
            self.scene_backbone = models_vit.vit_base_patch16(img_size=(image_size[1], image_size[0]), num_classes=1000, drop_path_rate=0.1, in_chans=5, global_pool=True)
        else:
            self.scene_backbone = models_vit.vit_base_patch16_epipolar(img_size=(image_size[1], image_size[0]), num_classes=1000, drop_path_rate=0.1, in_chans=5, global_pool=True, sample_num=sample_num, sim_type=sim_type)
        decoder_layer = CrossAttention_Layer(512, context_dim=768, heads=8, dim_head=64, dropout=0.1, activation='relu')
        self.gaze_decoder = GazeTransformer(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(512))
        self.scene_pos_enc = PositionalEncoding2D(768)
        self.hm_size = hm_size
        
        self.head_coord_project = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 512))    
        
        self.hidden_dim = 256
        self.gaze_inout_head = nn.Sequential(nn.Linear(1024, 512), 
                                             nn.ReLU(), 
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, 1))
        
        #self.gaze_inout_head = MLP(1024, self.hidden_dim, 1, 5)
        self.att_head = Prediction_Head(512)
        
        self.feat_size = (self.image_size[0]//16, self.image_size[1]//16)
        self.patch_w, self.patch_h = self.feat_size
        self.scene_feat_map = nn.Linear(768, 512)
        self.reset_global_metrics()
        print("Output patch size: {}".format(self.feat_size))
    
     
    def reset_global_metrics(self):   
        # accumulating statistics
        self.all_valid_samples_val, self.all_samples_val = 0,0
        self.num_inside_val, self.num_outside_val, self.num_occlusion_val = 0, 0, 0
        self.ang_err_list = []
        self.dist_list = []
        self.vis_pred_list, self.vis_gt_list = [], []
        self.main_info_gf, self.main_info_inout = [],[]
        self.head_valid_other = [] 
    
 
    def forward(self, input):
        img, head_img, head_mask, depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid = input
        bs, num_views = img.size()[:2]
    
        img, head_mask = img.view(bs * num_views, *img.size()[2:]), head_mask.view(bs * num_views, *head_mask.size()[2:])
        depth, eye_loc, intri, gaze_coords = depth.view(bs * num_views, *depth.size()[2:]), eye_loc.view(bs * num_views, 2), intri.view(bs * num_views, *intri.size()[2:]), gaze_coords.view(bs * num_views, 2)
        head_coords = head_coords.view(bs * num_views, 4)
        R = RT[:,:, :3, :3]
        if self.use_var: 
            gaze_vec, head_feat = self.gaze_estimator(head_img, R, head_valid)
            gaze_vec, gaze_var = gaze_vec[:, :2], gaze_vec[:, 2]
        else:
            gaze_vec, head_feat = self.gaze_estimator(head_img)  # headfeat: (B, 512)
        gaze_vec = spherical2cartesial(gaze_vec) 
        if self.use_var:
            gaze_vec_ori = gaze_vec.clone()
            if self.use_select:
                # for multiview: select more confident gaze vector
                gaze_vec = self.select_gaze_uncertainty(gaze_vec, gaze_var, RT, num_views=num_views, head_valid=head_valid) 
        else:
            gaze_vec_ori = gaze_vec.clone()
            gaze_var = torch.zeros(gaze_vec.size(0)).to(gaze_vec)
        
        head_centers = torch.stack((head_coords[:,0]+head_coords[:,2]/2, head_coords[:,1]+head_coords[:,3]/2), dim=1)
        # for multiview: select more confident gaze vector
        
        #fov_hm_ori, gt_vec, = get_fov_hm(eye_loc, gaze_vec_ori, depth, intri, image_size=self.image_size, fov_thres=self.fov_thres) 
        
        fov_hm, gt_vec = get_fov_hm(eye_loc, gaze_vec, depth, intri, image_size=self.image_size, fov_thres=self.fov_thres, tgt_gt=gaze_coords)
        fov_hm = fov_hm.unsqueeze(1)
        
        img_cat = torch.cat((img, head_mask, fov_hm), dim=1) 
        fund_mat_v1 = fund_mat
        fund_mat_v2 = fund_mat_v1.transpose(1,2)
        if not self.use_epi_attn:
            scene_feat = self.scene_backbone.forward_features(img_cat, spatial_only=True)  # (B, HW, 768)
        else:
            scene_feat = self.scene_backbone.forward_features(img_cat, spatial_only=True, fund_mat_v1=fund_mat_v1, fund_mat_v2=fund_mat_v2)  # (B, HW, 768)
        if type(scene_feat)==tuple:
            scene_feat, attn_weights, loc_valid_mask = scene_feat
        else:
            attn_weights, loc_valid_mask = None, None
        feat_dim = scene_feat.size(-1)
        scene_feat = scene_feat.reshape(-1, self.patch_h, self.patch_w, feat_dim)
        scene_pos_enc = self.scene_pos_enc(scene_feat)
        scene_feat, scene_pos_enc = scene_feat.flatten(1, 2), scene_pos_enc.flatten(1, 2)
        # encode head coordinates
        
        head_loc_embed = self.head_coord_project(head_centers)
        head_embed = head_feat + head_loc_embed
        head_embed = head_embed.unsqueeze(1)

        
        gaze_token = self.gaze_decoder(head_embed, context=scene_feat, attn_mask=None, query_pos_embed=None, context_pos_embed=scene_pos_enc)                    
        gaze_token, head_embed = gaze_token.squeeze(1), head_embed.squeeze(1)
        inout_token = torch.cat((head_embed, gaze_token), dim=1)
        inout_pred = self.gaze_inout_head(inout_token)
        
        scene_feat = self.scene_feat_map(scene_feat)
        att_feat = gaze_token.unsqueeze(1).expand(-1, self.patch_h*self.patch_w, -1) * scene_feat
        att_feat = rearrange(att_feat, 'B (H W) C -> B C H W', H=self.patch_h, W=self.patch_w).contiguous()
        hm_pred = self.att_head(att_feat)
        return hm_pred, inout_pred, fov_hm, gaze_vec, gaze_var, gt_vec
    
    
    
    def configure_optimizers(self):
        if self.freeze_gaze_backbone:
            print("***Freezing gaze backbone***")
            for param in self.gaze_estimator.backbone.parameters():
                param.requires_grad = False
            params = [p for p in self.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(params, lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  
        
        return optimizer    
    
    def on_train_epoch_start(self):
                
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        current_epoch = self.current_epoch
        self.log('lr', current_lr, on_epoch=True)
        print(f'Epoch {current_epoch} starting, Learning Rate: {current_lr}')
        
    def training_step(self, batch, batch_idx):
        data = batch['data']
        img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT = data
        bs, num_views = img.size()[:2]
         
        fund_mat = batch['fund_mat']
        hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = self.forward((img, head_img, head_mask, depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
        loss_mask = torch.logical_and(visib==1, head_valid==1)
        gt_vec, gazevec_loss_mask = get_gt_gaze_from_other(gt_vec, RT, loss_mask, num_views)
        
        
        gt_vec, gaze_vec, gaze_var = gt_vec.view(-1, num_views, 3), gaze_vec.view(-1, num_views, 3), gaze_var.view(-1, num_views)
        gt_vec, gaze_vec, gaze_var = gt_vec[:,0], gaze_vec[:,0], gaze_var[:,0]      
        gaze_heatmap = gaze_heatmap[:,0]  # have head and target
        visib, head_valid, loss_mask, loss_mask_other = visib[:,0], head_valid[:,0], loss_mask[:,0], loss_mask[:,1]
        hm_pred, vis_pred = hm_pred.view(bs, num_views, *hm_pred.size()[1:]), vis_pred.view(bs, num_views)
        hm_pred, hm_pred_other, vis_pred, vis_pred_other = hm_pred[:,0], hm_pred[:,1], vis_pred[:,0], vis_pred[:,1]
        
        
        num_valid_samples = loss_mask.sum().item()
        #gaze_heatmap = gaze_heatmap.view(-1, *gaze_heatmap.size()[2:])
        #visib, head_valid, loss_mask = visib.view(-1), head_valid.view(-1), loss_mask.view(-1)
        gazevec_loss_mask = torch.logical_and(gazevec_loss_mask, head_valid==1)
        
        if self.use_var:
            dir_loss = dir_variance_loss(gaze_vec, gt_vec, gaze_var, loss_mask=gazevec_loss_mask, loss_coef=2.0) * self.dir_weight
        else:
            dir_loss = gaze_dir_loss(gaze_vec, gt_vec, loss_mask) * self.dir_weight
        self.log('Train/dir_loss', dir_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=gazevec_loss_mask.sum().item())
        
            
        head_valid_mask = head_valid==1        
        num_head_samples = head_valid_mask.sum().item()
                

        hm_pred = hm_pred.squeeze(1)
        hm_loss = heatmap_loss(hm_pred, gaze_heatmap, loss_mask) * self.alpha
        occ_mask = visib==2
        visib[occ_mask] = 1  # assign occluded as inside, to perform in/out classification
            #visib_loss = F.cross_entropy(vis_pred, visib) * self.beta
        vis_pred = vis_pred.view(-1)
        vis_pred, visib = vis_pred[head_valid_mask], visib[head_valid_mask]
        visib_loss = F.binary_cross_entropy_with_logits(vis_pred, visib.float()) * self.beta
        loss = hm_loss + visib_loss + dir_loss
        

        self.log('Train/heatmap_loss', hm_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_valid_samples)
        self.log('Train/inout_loss', visib_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_head_samples)
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_head_samples)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        # note: use place holder -1 for the cases that the sample is not valid for evaluation, to ease multi-gpu evaluation 
        data = batch['data']
        img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT  = data
        fund_mat = batch['fund_mat']
        main_view = batch['main_id']
        bs, num_views = img.size()[:2]
        loss_mask = torch.logical_and(visib==1, head_valid==1) 
        hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = self.forward((img, head_img, head_mask, depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
        
        gt_vec, gazevec_loss_mask = get_gt_gaze_from_other(gt_vec, RT, loss_mask, num_views)
        gt_vec, gaze_vec, gaze_var = gt_vec.view(-1, num_views, 3), gaze_vec.view(-1, num_views, 3), gaze_var.view(-1, num_views)
        gt_vec, gaze_vec, gaze_var = gt_vec[:,0], gaze_vec[:,0], gaze_var[:,0]        
        gaze_vec = F.normalize(gaze_vec, dim=1)
        
        
        gaze_heatmap, gaze_coords = gaze_heatmap[:,0], gaze_coords[:,0] 
        num_samples = img.size(0)
        #visib, head_valid = visib.view(-1), head_valid.view(-1)
        
        head_valid_other = head_valid[:,1].bool()
        visib, head_valid, loss_mask = visib[:,0], head_valid[:,0], loss_mask[:,0]
        valid_mask = torch.logical_and(visib==1, head_valid==1) 
        gazevec_loss_mask = torch.logical_and(gazevec_loss_mask, head_valid==1) 
        num_valid_samples = valid_mask.sum().item()
        gaze_coords = gaze_coords.cpu().numpy() 
        
        if self.use_var:
            dir_loss = dir_variance_loss(gaze_vec, gt_vec, gaze_var, loss_mask=gazevec_loss_mask, loss_coef=2.0) * self.dir_weight
        else:
            dir_loss = gaze_dir_loss(gaze_vec, gt_vec, loss_mask) * self.dir_weight
        self.log('Validation/dir_loss', dir_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_valid_samples, sync_dist=True)
        
        hm_pred = hm_pred.squeeze(1)
        hm_pred, vis_pred = hm_pred.view(bs, num_views, *hm_pred.size()[1:]), vis_pred.view(bs, num_views)
        hm_pred, vis_pred = hm_pred[:,0], vis_pred[:,0]
        hm_loss = heatmap_loss(hm_pred, gaze_heatmap, valid_mask) * self.alpha
        occ_mask = visib==2
        visib[occ_mask] = 1  
        vis_pred = vis_pred.view(-1)
        #visib_loss = F.cross_entropy(vis_pred, visib) * self.beta
        head_valid_mask = head_valid==1
        vis_pred_for_loss, visib_for_loss = vis_pred[head_valid_mask], visib[head_valid_mask]
        visib_loss = F.binary_cross_entropy_with_logits(vis_pred_for_loss, visib_for_loss.float()) * self.beta
        loss = hm_loss + visib_loss
        self.num_inside_val += (visib==1).sum().item()
        self.num_outside_val += (visib==0).sum().item()
        self.all_samples_val+= num_samples
        self.all_valid_samples_val += num_valid_samples
        
        self.log('Validation/heatmap_loss', hm_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_valid_samples, sync_dist=True)
        self.log('Validation/inout_loss', visib_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_samples, sync_dist=True)
        self.log('Validation/loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_samples, sync_dist=True)
        
        vis_pred = torch.sigmoid(vis_pred)
        vis_pred[~head_valid_mask] = -1
        hm_pred, vis_pred = hm_pred.cpu().numpy(), vis_pred.cpu().numpy().tolist()
        dist_list = euclid_dist(hm_pred, gaze_coords, valid_mask, with_holder=True)
        
        # angular error
        gtvec_3d = gtvec_3d[:,0]
        vec_valid_mask = torch.logical_and(gtvec_3d[:,0]==0, gtvec_3d[:,1]==0)
        vec_valid_mask = ~vec_valid_mask
        vec_valid_mask = torch.logical_and(vec_valid_mask, head_valid.bool())
        ang_err_val = torch.ones(num_samples).to(gaze_vec) * -1
        ang_err = compute_angular_error(gaze_vec, gtvec_3d, spherical=False)
        ang_err_val[vec_valid_mask] = ang_err[vec_valid_mask]
        self.ang_err_list += ang_err_val.tolist()
        #print(f"Batch:{batch_idx}")
        #print(ang_err[vec_valid_mask])
        
        vis_gt= visib.cpu().numpy().tolist()
        self.dist_list += dist_list
        self.vis_pred_list += vis_pred 
        self.vis_gt_list += vis_gt
        self.head_valid_other += head_valid_other.tolist()
        
        # record which images are used, for evaluation based on the same main view
        img_idx_inout = main_view.cpu().numpy().tolist()
        img_idx_gf = main_view.cpu().numpy().tolist()  
        self.main_info_inout += img_idx_inout
        self.main_info_gf += img_idx_gf 
            
        
    def on_validation_epoch_end(self):
        
        dist_all = torch.tensor(self.dist_list)
        vis_pred_all, vis_gt_all = torch.tensor(self.vis_pred_list), torch.tensor(self.vis_gt_list)
        img_info_inout, img_info_gf = torch.tensor(self.main_info_inout), torch.tensor(self.main_info_gf)
        ang_all, img_info_ang = torch.tensor(self.ang_err_list), torch.tensor(self.main_info_gf)
        head_valid_other = torch.tensor(self.head_valid_other)
         
        if torch.distributed.get_world_size() > 1:
            dist_all = self.all_gather(dist_all)
            vis_pred_all, vis_gt_all = self.all_gather(vis_pred_all), self.all_gather(vis_gt_all)
            img_info_inout, img_info_gf = self.all_gather(img_info_inout), self.all_gather(img_info_gf)
            ang_all, img_info_ang = self.all_gather(ang_all), self.all_gather(img_info_ang)
            head_valid_other = self.all_gather(head_valid_other)
        assert len(img_info_inout) == len(vis_pred_all), f"{len(img_info_inout)} != {len(vis_pred_all)}"
        assert len(img_info_gf) == len(dist_all), f"{len(img_info_gf)} != {len(dist_all)}"
            
        #print(img_info_inout)
        
        if self.trainer.is_global_zero:
            if torch.distributed.get_world_size() > 1:
                dist_all = torch.transpose(dist_all, 0, 1).flatten()
                vis_pred_all, vis_gt_all = torch.transpose(vis_pred_all, 0, 1).flatten(), torch.transpose(vis_gt_all, 0, 1).flatten()
                img_info_inout, img_info_gf = torch.transpose(img_info_inout, 0, 1).flatten(), torch.transpose(img_info_gf, 0, 1).flatten()
                ang_all, img_info_ang = torch.transpose(ang_all, 0, 1).flatten(), torch.transpose(img_info_ang, 0, 1).flatten()
                head_valid_other = torch.transpose(head_valid_other, 0, 1).flatten()
                
            gf_mask = dist_all!=-1
            inout_mask = vis_pred_all!=-1
            dist_all, vis_pred_all, vis_gt_all = dist_all[gf_mask].cpu().numpy(), vis_pred_all[inout_mask].cpu().numpy(), vis_gt_all[inout_mask].cpu().numpy()
            img_info_inout, img_info_gf = img_info_inout[inout_mask].cpu().numpy(), img_info_gf[gf_mask].cpu().numpy()
            head_valid_gf, head_valid_inout = head_valid_other[gf_mask].cpu().numpy(), head_valid_other[inout_mask].cpu().numpy()
            ang_mask = ang_all!=-1
            ang_all, img_info_ang = ang_all[ang_mask].cpu().numpy(), img_info_ang[ang_mask].cpu().numpy()
            head_valid_ang = head_valid_other[ang_mask].cpu().numpy()
             
            dist_avg_all, dist_best_all, _ = metric_avg_best(dist_all, img_info_gf, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            vis_avg_all, vis_best_all, vis_gt_unique = metric_avg_best(vis_pred_all, img_info_inout, gt_list=vis_gt_all, is_auc=False, is_dist=False, is_ap=True)
            ang_avg_all, ang_best_all, _ = metric_avg_best(ang_all, img_info_ang, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            #print(vis_gt_unique)
            
            dist_avg, dist_best = np.mean(dist_avg_all), np.mean(dist_best_all)
            vis_pred_all, vis_gt_unique = np.array(vis_pred_all), np.array(vis_gt_unique)
            ang_avg, ang_best = np.mean(ang_avg_all), np.mean(ang_best_all)
            vis_gt_unique = vis_gt_unique.astype(np.int32)
            ap_avg = ap(vis_avg_all, vis_gt_unique)
            ap_best = ap(vis_best_all, vis_gt_unique)
            
            # metric based on whether head is visible in the other view
            dist_head, dist_nohead, _, _ = metric_from_head(dist_all, img_info_gf, head_valid_gf, gt_list=None)
            vis_avg_head, vis_avg_nohead, vis_gt_unique_head, vis_gt_unique_nohead = metric_from_head(vis_pred_all, img_info_inout, head_valid_inout, gt_list=vis_gt_all)
            ang_avg_head, ang_avg_nohead, _, _ = metric_from_head(ang_all, img_info_ang, head_valid_ang, gt_list=None)
            #ang_from_tgt_avg_head, ang_from_tgt_avg_nohead, _, _ = metric_from_head(ang_from_tgt, img_info_ang, head_valid_ang, gt_list=None)
            
             
            dist_head, dist_nohead = np.mean(dist_head), np.mean(dist_nohead)
            ang_avg_head, ang_avg_nohead = np.mean(ang_avg_head), np.mean(ang_avg_nohead)
            #ang_tgt_avg_head, ang_tgt_avg_nohead = np.mean(ang_from_tgt_avg_head), np.mean(ang_from_tgt_avg_nohead)
            ap_avg_head = ap(vis_avg_head, vis_gt_unique_head) if len(vis_gt_unique_head)>0 else 0
            ap_avg_nohead = ap(vis_avg_nohead, vis_gt_unique_nohead) if len(vis_gt_unique_nohead)>0 else 0
            
                                
            if not self.trainer.sanity_checking:
                self.log("Val/Dist_avg", dist_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/Dist_head", dist_head, on_step=False, on_epoch=True, rank_zero_only=True)
                #self.log("Val/Dist_best", dist_best, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/AP_avg", ap_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/AP_head", ap_avg_head, on_step=False, on_epoch=True, rank_zero_only=True)
                #self.log("Val/AP_best", ap_best, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/Ang_avg", ang_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/Ang_head", ang_avg_head, on_step=False, on_epoch=True, rank_zero_only=True)
                #self.log("Val/Ang_best", ang_best, on_step=False, on_epoch=True, rank_zero_only=True)
                
            print("Epoch {}: Avg: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(self.current_epoch, dist_avg, ap_avg, ang_avg))
            print("Best: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(dist_best, ap_best, ang_best))
            print("---------------------------------------------------------------------------------")
            print("Head: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(dist_head, ap_avg_head, ang_avg_head))
            print("NoHead: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(dist_nohead, ap_avg_nohead, ang_avg_nohead))
            print("---------------------------------------------------------------------------------")
            print("Val: Num inside: {}, num outside: {}".format(self.num_inside_val, self.num_outside_val))
            print("Val: Total valid samples:{}, total samples: {}, ratio: {:.3f}%".format(self.all_valid_samples_val, self.all_samples_val, self.all_valid_samples_val/self.all_samples_val*100))    
        
        self.reset_global_metrics()



class Transformer_fov_cat_CrossView(pl.LightningModule):
    def __init__(self, lr, image_size, alpha, beta, dir_weight, fov_thres=0.9, num_decoder_layers=1, use_var=False, hm_size=(64,64), sample_num=64, use_epi_attn=True, sim_type='softmax'):
        super(Transformer_fov_cat_CrossView, self).__init__()
        self.lr = lr
        self.image_size = image_size    
        self.alpha = alpha
        self.beta = beta
        self.dir_weight = dir_weight
        self.fov_thres = fov_thres
        self.use_var = use_var 
        self.use_epi_attn = use_epi_attn
        
        if self.use_var:
            print("Use Var")
            self.gaze_estimator = GazeEstimator_Res18_CrossAtt(lr=lr, use_var=use_var)
        else:
            print("No Var")
            self.gaze_estimator = GazeEstimator_Res18_pair(lr=lr, use_var=use_var)
        if not self.use_epi_attn:
            self.scene_backbone = models_vit.vit_base_patch16(img_size=(image_size[1], image_size[0]), num_classes=1000, drop_path_rate=0.1, in_chans=5, global_pool=True)
        else:
            self.scene_backbone = models_vit.vit_base_patch16_epipolar(img_size=(image_size[1], image_size[0]), num_classes=1000, drop_path_rate=0.1, in_chans=5, global_pool=True, sample_num=sample_num, sim_type=sim_type)
        decoder_layer = CrossAttention_Layer(512, context_dim=768, heads=8, dim_head=64, dropout=0.1, activation='relu')
        self.gaze_decoder = GazeTransformer(decoder_layer, num_decoder_layers, norm=nn.LayerNorm(512))
        self.scene_pos_enc = PositionalEncoding2D(768)
        self.hm_size = hm_size
        self.outside_head_embedding = nn.Parameter(torch.randn(512))
        self.outside_head_project = nn.Sequential(nn.Linear(521, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 512))
        
        self.head_coord_project = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 512))    
        
        self.hidden_dim = 256
        self.gaze_inout_head = nn.Sequential(nn.Linear(1024, 512), 
                                             nn.ReLU(), 
                                             nn.Linear(512, 256),
                                             nn.ReLU(),
                                             nn.Linear(256, 128),
                                             nn.ReLU(),
                                             nn.Linear(128, 1))
        
        #self.gaze_inout_head = MLP(1024, self.hidden_dim, 1, 5)
        self.att_head = Prediction_Head(512)
        
        self.feat_size = (self.image_size[0]//16, self.image_size[1]//16)
        self.patch_w, self.patch_h = self.feat_size
        self.scene_feat_map = nn.Linear(768, 512)
        self.reset_global_metrics()
        print("Output patch size: {}".format(self.feat_size))
    
     
    def reset_global_metrics(self):   
        # accumulating statistics
        self.all_valid_samples_val, self.all_samples_val = 0,0
        self.num_inside_val, self.num_outside_val, self.num_occlusion_val = 0, 0, 0
        self.ang_err_list = []
        self.dist_list = []
        self.vis_pred_list, self.vis_gt_list = [], []
        self.main_info_gf, self.main_info_inout = [],[]
        self.head_valid_other = [] 
    
    
    def forward(self, input):
        img, head_img, head_mask, depth, abs_depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid = input
        bs, num_views = img.size()[:2]

        head_other = head_img[:,1]
        intri_head, intri_scene = intri[:, 1], intri[:, 0]
        depth_head, depth_scene = abs_depth[:, 1], abs_depth[:, 0]
        img, head_mask = img.view(bs * num_views, *img.size()[2:]), head_mask.view(bs * num_views, *head_mask.size()[2:])
        head_coords = head_coords.view(bs * num_views, 4)
        R = RT[:,:, :3, :3]
        gaze_coords = gaze_coords[:,0]
        if self.use_var: 
            gaze_vec, head_feat = self.gaze_estimator(head_img, R, head_valid)
            gaze_vec = gaze_vec.view(-1, num_views, 3)  
            gaze_vec, gaze_var = gaze_vec[..., :2], gaze_vec[..., 2]
            gaze_var = gaze_var[:,1]
        else:
            gaze_vec, head_feat = self.gaze_estimator(head_img)  # headfeat: (B, 512)
            gaze_vec = gaze_vec.view(-1, num_views, 3)  
            
        gaze_vec = gaze_vec[:,1]  # only use the 2nd view's head information
        gaze_vec = spherical2cartesial(gaze_vec)
        head_feat = head_feat.view(-1, num_views, 512)
        head_feat = head_feat[:,1]
        
        R_1, R_2 = RT[:,0,:,:3], RT[:,1,:,:3]    
        T_1, T_2 = RT[:,0,:,3], RT[:,1,:,3]
        R_2to1 = torch.bmm(R_1, R_2.transpose(1,2))
        T_2to1 = T_1 - torch.bmm(R_2to1, T_2.unsqueeze(-1)).squeeze(-1)
        RT_2to1 = torch.cat((R_2to1, T_2to1.unsqueeze(-1)), dim=2)
        RT_transform = RT_2to1
        R_transform = RT_transform[:,:,:3]
        
        pred_vec_cam = torch.bmm(R_transform, gaze_vec.unsqueeze(-1)).squeeze(-1)
        pred_vec_cam = pred_vec_cam / (torch.norm(pred_vec_cam, dim=1, keepdim=True) + 1e-9)
        # for multiview: select more confident gaze vector
        fov_hm_other, gt_vec = get_fov_hm(eye_loc[:,1], gaze_vec, depth_head, intri[:,1], image_size=self.image_size, fov_thres=self.fov_thres)
        # cross view fov heatmap
        fov_main, gt_vec = get_fov_hm_crossview(eye_loc[:,1], pred_vec_cam, depth_head, depth_scene, intri_head, intri_scene, RT_transform, image_size=self.image_size, 
                                                fov_thres=0.9, scaleshift_vhead=None, scaleshift_vscene=None, tgt_gt=gaze_coords)
        
        fov_main, fov_hm_other = fov_main.unsqueeze(1), fov_hm_other.unsqueeze(1)
        fov_hm = torch.stack((fov_main, fov_hm_other), dim=1).flatten(0,1)
       
        # encode face embedding from another view with camera parameters
        img_cat = torch.cat((img, head_mask, fov_hm), dim=1)
        fund_mat_v1 = fund_mat
        fund_mat_v2 = fund_mat_v1.transpose(1,2)
        if not self.use_epi_attn:
            scene_feat = self.scene_backbone.forward_features(img_cat, spatial_only=True)  # (B, HW, 768)
        else:
            scene_feat, attn_all, loc_mask_all = self.scene_backbone.forward_features(img_cat, spatial_only=True, fund_mat_v1=fund_mat_v1, fund_mat_v2=fund_mat_v2)  # (B, HW, 768)
        feat_dim = scene_feat.size(-1)
        scene_feat = scene_feat.view(bs, num_views, -1, feat_dim)
        scene_feat = scene_feat[:,0]  # only use the first view's scene feature 
        
        scene_feat = scene_feat.reshape(-1, self.patch_h, self.patch_w, feat_dim)
        scene_pos_enc = self.scene_pos_enc(scene_feat)
        scene_feat, scene_pos_enc = scene_feat.flatten(1, 2), scene_pos_enc.flatten(1, 2)
        
        
        R_transform = R_transform.flatten(1,2)
        head_feat_cam = torch.cat((head_feat, R_transform), dim=1)
        head_feat_other = self.outside_head_project(head_feat_cam)
        head_loc_embed = self.outside_head_embedding.unsqueeze(0).expand(head_feat_other.shape[0], -1)  
        head_embed = head_feat_other + head_loc_embed
        head_embed = head_embed.unsqueeze(1)
        
        gaze_token = self.gaze_decoder(head_embed, context=scene_feat, attn_mask=None, query_pos_embed=None, context_pos_embed=scene_pos_enc)                    
        gaze_token, head_embed = gaze_token.squeeze(1), head_embed.squeeze(1)
        inout_token = torch.cat((head_embed, gaze_token), dim=1)
        inout_pred = self.gaze_inout_head(inout_token)
        
        scene_feat = self.scene_feat_map(scene_feat)
        att_feat = gaze_token.unsqueeze(1).expand(-1, self.patch_h*self.patch_w, -1) * scene_feat
        att_feat = rearrange(att_feat, 'B (H W) C -> B C H W', H=self.patch_h, W=self.patch_w).contiguous()
        hm_pred = self.att_head(att_feat)
        
        
        return hm_pred, inout_pred, fov_hm, pred_vec_cam, gaze_var, gt_vec
    
    def select_gaze_uncertainty(self, gaze_vec, gaze_var, RT, num_views=2, head_valid=None):    
        gaze_vec, gaze_var = gaze_vec.reshape(-1, num_views, 3), gaze_var.squeeze().reshape(-1, num_views)
        num_tgts = gaze_vec.size(0)
        select_idx = torch.argmin(gaze_var, dim=1)
        batch_idx = torch.arange(num_tgts).to(gaze_vec).long()
        # if head is invalid in the other view, forcily select the first view
        if head_valid is not None:
            other_view_invalid = head_valid[:, 1] == 0
            select_idx[other_view_invalid] = 0   
            
            other_view_invalid = head_valid[:, 0] == 0
            select_idx[other_view_invalid] = 1      
        
        other_idx = 1 - select_idx
        gaze_vec_select = gaze_vec[batch_idx, select_idx]
        R_vselect, R_vother = RT[batch_idx, select_idx, :, :3], RT[batch_idx, other_idx, :, :3]
        gaze_other = torch.bmm(R_vselect.transpose(1,2), gaze_vec_select.unsqueeze(-1))
        gaze_other = torch.bmm(R_vother, gaze_other).squeeze(-1)
        gaze_vec[batch_idx, other_idx] = gaze_other
        gaze_vec = gaze_vec.reshape(-1, 3)
        return gaze_vec
    
    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)  
        
        return optimizer    
    
    def on_train_epoch_start(self):
                
        optimizer = self.optimizers()
        current_lr = optimizer.param_groups[0]['lr']
        current_epoch = self.current_epoch
        self.log('lr', current_lr, on_epoch=True)
        print(f'Epoch {current_epoch} starting, Learning Rate: {current_lr}')
        
    def training_step(self, batch, batch_idx):
        data = batch['data']
        abs_depth = batch['abs_depth']
        img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT, quat = data
        bs, num_views = img.size()[:2]
        
        assert torch.all(head_valid[:,1]), head_valid[:,1]
        fund_mat = batch['fund_mat']
        hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec = self.forward((img, head_img, head_mask, depth, abs_depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
        loss_mask = visib==1
        
        
        gaze_heatmap = gaze_heatmap[:,0]  # have head and target
        visib, head_valid, loss_mask, loss_mask_other = visib[:,0], head_valid[:,0], loss_mask[:,0], loss_mask[:,1]
    
        
        num_valid_samples = loss_mask.sum().item()
        
        if self.use_var:
            dir_loss = dir_variance_loss(gaze_vec, gt_vec, gaze_var, loss_mask=loss_mask, loss_coef=2.0) * self.dir_weight
        else:
            dir_loss = gaze_dir_loss(gaze_vec, gt_vec, loss_mask) * self.dir_weight
        self.log('Train/dir_loss', dir_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=loss_mask.sum().item())
        
            
        head_valid_mask = head_valid==1        
        num_head_samples = head_valid_mask.sum().item()
                

        hm_pred = hm_pred.squeeze(1)
        hm_loss = heatmap_loss(hm_pred, gaze_heatmap, loss_mask) * self.alpha
        occ_mask = visib==2
        visib[occ_mask] = 1  # assign occluded as inside, to perform in/out classification
        vis_pred = vis_pred.view(-1)
        #vis_pred, visib = vis_pred[head_valid_mask], visib[head_valid_mask]
        visib_loss = F.binary_cross_entropy_with_logits(vis_pred, visib.float()) * self.beta
        loss = hm_loss + visib_loss + dir_loss
        

        self.log('Train/heatmap_loss', hm_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_valid_samples)
        self.log('Train/inout_loss', visib_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_head_samples)
        self.log('Train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=num_head_samples)

        return loss
    
    
    def validation_step(self, batch, batch_idx):
        # note: use place holder -1 for the cases that the sample is not valid for evaluation, to ease multi-gpu evaluation 
        data = batch['data']
        abs_depth = batch['abs_depth']
        img, head_img, head_mask, depth, gaze_heatmap, visib, gaze_coords, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT, quat  = data
        fund_mat = batch['fund_mat']
        main_view = batch['main_id']
        bs, num_views = img.size()[:2]
    
        assert not torch.any(head_valid[:,0]), head_valid[:,0]
        assert torch.all(head_valid[:,1]), head_valid[:,1]
        
        hm_pred, vis_pred, fov_hm, gaze_vec, gaze_var, gt_vec  = self.forward((img, head_img, head_mask, depth, abs_depth, eye_loc, head_coords, intri, RT, fund_mat, gaze_coords, head_valid))
        
        gaze_heatmap, gaze_coords = gaze_heatmap[:,0], gaze_coords[:,0] 
        num_samples = img.size(0)
        #visib, head_valid = visib.view(-1), head_valid.view(-1)
        loss_mask = visib==1
        visib, head_valid, loss_mask = visib[:,0], head_valid[:,0], loss_mask[:,0]
        
        num_valid_samples = loss_mask.sum().item()
        #gaze_heatmap = gaze_heatmap.view(-1, *gaze_heatmap.size()[2:])
        #visib, head_valid, loss_mask = visib.view(-1), head_valid.view(-1), loss_mask.view(-1)
        #gazevec_loss_mask = torch.logical_and(gazevec_loss_mask, head_valid==1)
        
        #print(gaze_vec.size(), gt_vec.size(), gaze_var.size(), loss_mask.size())
        if self.use_var:
            dir_loss = dir_variance_loss(gaze_vec, gt_vec, gaze_var, loss_mask=loss_mask, loss_coef=2.0) * self.dir_weight
        else:
            dir_loss = gaze_dir_loss(gaze_vec, gt_vec, loss_mask) * self.dir_weight
        
        num_valid_samples = loss_mask.sum().item()
        gaze_coords = gaze_coords.cpu().numpy() 
        self.log('Validation/dir_loss', dir_loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=num_valid_samples, sync_dist=True)
        
        hm_pred = hm_pred.squeeze(1)
        hm_loss = heatmap_loss(hm_pred, gaze_heatmap, loss_mask) * self.alpha
        occ_mask = visib==2
        visib[occ_mask] = 1  
        vis_pred = vis_pred.view(-1)
        #visib_loss = F.cross_entropy(vis_pred, visib) * self.beta
        
        visib_loss = F.binary_cross_entropy_with_logits(vis_pred, visib.float()) * self.beta
        loss = hm_loss + visib_loss
        self.num_inside_val += (visib==1).sum().item()
        self.num_outside_val += (visib==0).sum().item()
        self.all_samples_val+= num_samples
        self.all_valid_samples_val += num_valid_samples
        
        self.log('Validation/heatmap_loss', hm_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_valid_samples, sync_dist=True)
        self.log('Validation/inout_loss', visib_loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_samples, sync_dist=True)
        self.log('Validation/loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=num_samples, sync_dist=True)
        
        vis_pred = torch.sigmoid(vis_pred)
        hm_pred, vis_pred = hm_pred.cpu().numpy(), vis_pred.cpu().numpy().tolist()
        dist_list = euclid_dist(hm_pred, gaze_coords, loss_mask, with_holder=True)
        
        # angular error
        ang_err_val = torch.ones(num_samples).to(gaze_vec) * -1
        ang_err = compute_angular_error(gaze_vec, gt_vec, spherical=False)
        ang_err_val[loss_mask] = ang_err[loss_mask]
        self.ang_err_list += ang_err_val.tolist()
        #print(f"Batch:{batch_idx}")
        #print(ang_err[vec_valid_mask])
        
        vis_gt= visib.cpu().numpy().tolist()
        self.dist_list += dist_list
        self.vis_pred_list += vis_pred 
        self.vis_gt_list += vis_gt
        
        # record which images are used, for evaluation based on the same main view
        img_idx_inout = main_view.cpu().numpy().tolist()
        img_idx_gf = main_view.cpu().numpy().tolist()  
        self.main_info_inout += img_idx_inout
        self.main_info_gf += img_idx_gf 
            
        
    def on_validation_epoch_end(self):
        
        dist_all = torch.tensor(self.dist_list)
        vis_pred_all, vis_gt_all = torch.tensor(self.vis_pred_list), torch.tensor(self.vis_gt_list)
        img_info_inout, img_info_gf = torch.tensor(self.main_info_inout), torch.tensor(self.main_info_gf)
        ang_all, img_info_ang = torch.tensor(self.ang_err_list), torch.tensor(self.main_info_gf)
         
        if torch.distributed.get_world_size() > 1:
            dist_all = self.all_gather(dist_all)
            vis_pred_all, vis_gt_all = self.all_gather(vis_pred_all), self.all_gather(vis_gt_all)
            img_info_inout, img_info_gf = self.all_gather(img_info_inout), self.all_gather(img_info_gf)
            ang_all, img_info_ang = self.all_gather(ang_all), self.all_gather(img_info_ang)
            
        assert len(img_info_inout) == len(vis_pred_all), f"{len(img_info_inout)} != {len(vis_pred_all)}"
        assert len(img_info_gf) == len(dist_all), f"{len(img_info_gf)} != {len(dist_all)}"
            
        #print(img_info_inout)
        
        if self.trainer.is_global_zero:
            if torch.distributed.get_world_size() > 1:
                dist_all = torch.transpose(dist_all, 0, 1).flatten()
                vis_pred_all, vis_gt_all = torch.transpose(vis_pred_all, 0, 1).flatten(), torch.transpose(vis_gt_all, 0, 1).flatten()
                img_info_inout, img_info_gf = torch.transpose(img_info_inout, 0, 1).flatten(), torch.transpose(img_info_gf, 0, 1).flatten()
                ang_all, img_info_ang = torch.transpose(ang_all, 0, 1).flatten(), torch.transpose(img_info_ang, 0, 1).flatten()

                
            gf_mask = dist_all!=-1
            inout_mask = vis_pred_all!=-1
            dist_all, vis_pred_all, vis_gt_all = dist_all[gf_mask].cpu().numpy(), vis_pred_all[inout_mask].cpu().numpy(), vis_gt_all[inout_mask].cpu().numpy()
            img_info_inout, img_info_gf = img_info_inout[inout_mask].cpu().numpy(), img_info_gf[gf_mask].cpu().numpy()
            ang_mask = ang_all!=-1
            ang_all, img_info_ang = ang_all[ang_mask].cpu().numpy(), img_info_ang[ang_mask].cpu().numpy()
             
            dist_avg_all, dist_best_all, _ = metric_avg_best(dist_all, img_info_gf, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            vis_avg_all, vis_best_all, vis_gt_unique = metric_avg_best(vis_pred_all, img_info_inout, gt_list=vis_gt_all, is_auc=False, is_dist=False, is_ap=True)
            ang_avg_all, ang_best_all, _ = metric_avg_best(ang_all, img_info_ang, gt_list=None, is_auc=False, is_dist=True, is_ap=False)
            #print(vis_gt_unique)
            
            dist_avg, dist_best = np.mean(dist_avg_all), np.mean(dist_best_all)
            vis_pred_all, vis_gt_unique = np.array(vis_pred_all), np.array(vis_gt_unique)
            ang_avg, ang_best = np.mean(ang_avg_all), np.mean(ang_best_all)
            vis_gt_unique = vis_gt_unique.astype(np.int32)
            ap_avg = ap(vis_avg_all, vis_gt_unique)
            ap_best = ap(vis_best_all, vis_gt_unique)
            
                                
            if not self.trainer.sanity_checking:
                self.log("Val/Dist_avg", dist_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/AP_avg", ap_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                self.log("Val/Ang_avg", ang_avg, on_step=False, on_epoch=True, rank_zero_only=True)
                
            print("Epoch {}: Avg: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(self.current_epoch, dist_avg, ap_avg, ang_avg))
            print("Best: Dist: {:.3f}, Ap: {:.3f}, Ang:{:.2f}".format(dist_best, ap_best, ang_best))
            print("---------------------------------------------------------------------------------")
            print("Val: Num inside: {}, num outside: {}".format(self.num_inside_val, self.num_outside_val))
            print("Val: Total valid samples:{}, total samples: {}, ratio: {:.3f}%".format(self.all_valid_samples_val, self.all_samples_val, self.all_valid_samples_val/self.all_samples_val*100))    
            print("total pairs: gaze: {}, inout: {} ".format(len(dist_all), len(vis_pred_all)))
            print("---------------------------------------------------------------------------------")
                
        self.reset_global_metrics()