import torch
import torch.nn as nn
import pytorch_lightning as pl
import itertools
import numpy as np
from model.resnet import resnet18
from model.Attention_mine import Gaze_View_Attention
from torch.nn import functional as F
from .loss import dir_variance_loss, gaze_dir_loss
from utils import utils
from utils.evaluation import compute_angular_error, metric_avg_best


class GazeEstimator_Res18(pl.LightningModule):
    ''' The Module for Estimating Gaze Direction '''

    def __init__(self, lr, loss_coef=1.0, use_var=False):

        super(GazeEstimator_Res18,self).__init__()
        self.feature_dim = 512

        # backbone in the module
        #self.backbone=nn.Sequential(*list(org_resnet.children())[:-1])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.backbone = resnet18(pretrained=False)
        self.lr = lr
        self.loss_coef = loss_coef
        self.use_var = use_var
        output_dim = 3 if use_var else 2
        self.backbone.fc=nn.Sequential(nn.Linear(512, self.feature_dim),    # if res50: change to 2048
                              nn.ReLU(),
                              nn.Linear(self.feature_dim, output_dim))
        self.val_loss = 0.0
        self.ang_err = 0.0
        self.valid_samples_train, self.valid_samples_val = 0,0
        self.eps = 1e-8

    def forward(self, himg):

        """
        Args:
            himg: cropped head image
        Returns:
            gazevector: normalized gaze direction predicted by the module.
        """
        if len(himg.size())==5:
            bs, num_views = himg.size()[:2]
            himg = himg.view(-1, *himg.size()[2:])
        
        headfeat = self.backbone(himg, extract_feature=True)   
        headfeat = self.avgpool(headfeat)
        headfeat=torch.flatten(headfeat,1)
        gaze_pred=self.backbone.fc(headfeat)  # design to predict spherical coordinates and variance
        
        return gaze_pred, headfeat



class GazeEstimator_Res18_CrossAtt(pl.LightningModule):
    ''' The Module for Estimating Gaze Direction '''

    def __init__(self, lr, loss_coef=2.0, use_var=False, num_heads=8, tf_depth=1, optim=None, consist_weight=0.0, freeze_backbone=False):
        # tf_depth: the depth of transformer block

        super(GazeEstimator_Res18_CrossAtt,self).__init__()
        self.feature_dim = 512
        self.freeze_backbone = freeze_backbone

        # backbone in the module
        #self.backbone=nn.Sequential(*list(org_resnet.children())[:-1])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.backbone = resnet18(pretrained=False, no_fc=True)
        self.lr = lr
        self.loss_coef = loss_coef
        self.use_var = use_var
        output_dim = 3 if use_var else 2
        self.fc=nn.Sequential(nn.Linear(512, self.feature_dim),    # if res50: change to 2048
                              nn.ReLU(),
                              nn.Linear(self.feature_dim, output_dim))
        
        self.val_loss = 0.0
        self.ang_err = 0.0
        self.valid_samples_train, self.valid_samples_val = 0,0
        self.eps = 1e-8
        self.optim = optim
        self.tf_block = Gaze_View_Attention(512, num_heads, num_decoder_layers=tf_depth, dim_feedforward=512, normalize_before=False)
        #self.skip_mapping = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        
        #### added for no head available 
        #self.cam_param_proj = nn.Conv2d(512+9, 512, kernel_size=1, stride=1, padding=0)
         
        self.consist_weight = consist_weight         
        self.image_size = (512, 384)
        self.ang_err_list = []
        self.main_info = []

    def forward(self, himg, R, head_valid):

        """
        Args:
            himg: cropped head image
        Returns:
            gazevector: normalized gaze direction predicted by the module.
        """
        bs, num_views = himg.size()[:2]
        himg = himg.view(-1, *himg.size()[2:])   
             
        headfeat = self.backbone(himg, extract_feature=True)   
        headfeat = headfeat.reshape(bs, num_views, *headfeat.size()[1:]).contiguous()
        R_0, R_1 = R[:,0], R[:,1]
        head_valid = head_valid.reshape(bs, num_views)
        head_valid_0, head_valid_1 = head_valid[:,0], head_valid[:,1]   
        R_1to0, R_0to1 = torch.bmm(R_0, R_1.transpose(1,2)), torch.bmm(R_1, R_0.transpose(1,2))
        
        headfeat_main, headfeat_other = headfeat[:,0], headfeat[:,1]
        headfeat_main_att = self.tf_block(headfeat_main, headfeat_other, R_1to0)
        headfeat_other_att = self.tf_block(headfeat_other, headfeat_main, R_0to1)
        
        #headfeat_skip_main = self.skip_mapping(headfeat_main)   # when the other view does not have head (no information), directly map the original feature
        #headfeat_skip_other = self.skip_mapping(headfeat_other)
        #headfeat_main_att[~head_valid_1] = headfeat_skip_main[~head_valid_1]
        #headfeat_other_att[~head_valid_0] = headfeat_skip_other[~head_valid_0]
    
        # original implementation
        headfeat_att = torch.stack([headfeat_main_att, headfeat_other_att], dim=1).reshape(bs*num_views, *headfeat_main.size()[1:]).contiguous()
        headfeat_att = self.avgpool(headfeat_att)
        headfeat_att = torch.flatten(headfeat_att,1)
        gaze_pred = self.fc(headfeat_att)
            
        return gaze_pred, headfeat_att
    
    


class GazeEstimator_Res18_pair(pl.LightningModule):
    ''' The Module for Estimating Gaze Direction '''

    def __init__(self, lr, loss_coef=1.0, use_var=False, optim='adamw', freeze_backbone=False):

        super(GazeEstimator_Res18_pair,self).__init__()
        self.feature_dim = 512
        self.freeze_backbone = freeze_backbone
        # backbone in the module
        #self.backbone=nn.Sequential(*list(org_resnet.children())[:-1])
        self.backbone = resnet18(pretrained=False, no_fc=True)
        self.lr = lr
        self.loss_coef = loss_coef
        self.use_var = use_var
        self.optim = optim
        output_dim = 3 if use_var else 2
        self.fc=nn.Sequential(nn.Linear(512, self.feature_dim),    # if res50: change to 2048
                              nn.ReLU(),
                              nn.Linear(self.feature_dim, output_dim))
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.val_loss = 0.0
        self.ang_err = 0.0
        self.valid_samples_train, self.valid_samples_val = 0,0
        self.all_samples_train, self.all_samples_val = 0,0
        self.eps = 1e-8
        self.image_size = (512, 384)
        self.ang_err_list = []
        self.main_info = []

        
    def forward(self, himg):

        """
        Args:
            himg: cropped head image
        Returns:
            gazevector: normalized gaze direction predicted by the module.
        """
        if len(himg.size())==5:
            bs, num_views = himg.size()[:2]
            himg = himg.view(-1, *himg.size()[2:])
        
        headfeat = self.backbone(himg, extract_feature=True)   
        headfeat = self.avgpool(headfeat)
        headfeat=torch.flatten(headfeat,1)
        gaze_pred=self.fc(headfeat)  # design to predict spherical coordinates and variance        
        return gaze_pred, headfeat

    
        
            