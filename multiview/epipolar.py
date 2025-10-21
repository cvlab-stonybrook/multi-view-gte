import torch
from torch import nn
import torch.nn.functional as F
from .multiview import pix2coord, coord2pix, normalize, de_normalize, coord2pix_xy
from model.Attention_mine import CrossAttention
from einops import rearrange, repeat


class Epipolar_Attn(nn.Module):
    def __init__(self, feat_size, feat_channels, sample_size=64, downsample_x=4, downsample_y=4, attn_heads=1, dim_head=64, sim_type='cos'):
        super(Epipolar_Attn, self).__init__()
        
        self.feat_w, self.feat_h = feat_size
        self.sample_size = sample_size
        self.epsilon = 0.001 # for avoiding floating point error
        self.attn_heads = attn_heads   
        self.sim_type = sim_type
        
        y = torch.arange(0, self.feat_h, dtype=torch.float) # 0 .. 128
        x = torch.arange(0, self.feat_w, dtype=torch.float) # 0 .. 96
        
        # rescale to original resolution 
        self.downsample_x, self.downsample_y = downsample_x, downsample_y
        y = pix2coord(y, self.downsample_y)
        x = pix2coord(x, self.downsample_x)   # 128 -> 512
        
        grid_y, grid_x = torch.meshgrid(y, x)
        self.grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x))).view(3, -1)
        self.xmin = x[0]
        self.ymin = y[0]
        self.xmax = x[-1]
        self.ymax = y[-1]
        self.tmp_tensor = torch.tensor([True, True, False, False])
        self.outrange_tensor = torch.tensor([
            self.xmin-10000, self.ymin-10000, 
            self.xmin-10000, self.ymin-10000]).view(2, 2)
        #self.sample_steps = torch.range(0, 1, 1./(self.sample_size-1)).view(-1, 1, 1, 1)
        self.sample_steps = torch.linspace(0, 1, self.sample_size).view(-1, 1, 1, 1)

        self.feat_channels = feat_channels
        self.softmax_scale = 1 / self.sample_size**.5
        
        # model parameters
        self.layernorm = nn.LayerNorm(self.feat_channels)
        self.project = nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=1, stride=1, padding=0, bias=True)
        #self.project_bn = nn.BatchNorm2d(self.feat_channels)
         
        if self.sim_type=='softmax':
            self.cross_attn = CrossAttention(self.feat_channels, heads=attn_heads, dim_head=dim_head, dropout=0.1)  # default: batch first
            #self.cross_attn = TransformerDecoderLayer_NoSA(d_model=self.feat_channels, nhead=attn_heads, dim_feedforward=dim_head*attn_heads, dropout=0.1, batch_first=True)   
            self.epi_pos_embed = nn.Parameter(torch.randn(self.sample_size, self.feat_channels))
         
    def forward(self, feat1, feat2, fund_mat, depth=None, camera=None, other_camera=None, ref1=None, ref2=None):
        """ 
        Args:
            feat1         : N x C x H x W
            feat2         : N x C x H x W
            P1          : N x 3 x 4
            P2          : N x 3 x 4
        1. Compute epipolar lines: NHW x 3 (http://users.umiacs.umd.edu/~ramani/cmsc828d/lecture27.pdf)
        2. Compute intersections with the image: NHW x 2 x 2
            4 intersections with each boundary of the image NHW x 4 x 2
            Convert to (-1, 1)
            find intersections on the rectangle NHW x 4 T/F, NHW x 2 x 2
            sample N*sample_size x H x W x 2
                if there's no intersection, the sample points are out of (-1, 1), therefore ignored by pytorch
        3. Sample points between the intersections: sample_size x N x H x W x 2
        4. grid_sample: sample_size*N x C x H x W -> sample_size x N x C x H x W
            trick: compute feat1 feat2 dot product first: N x HW x H x W
        5. max pooling/attention: N x C x H x W
        """
        
            
        
        N, C, H, W = feat1.shape        
        feat2 = feat2.view(1, N, C, H, W)
        feat2 = feat2.expand(self.sample_size, N, C, H, W)

        
        with torch.no_grad():
            sample_locs, loc_valid_mask = self.sample_locs(self.grid, H, W, fund_mat)   # sample_size x N x H x W x 2
            sample_locs = sample_locs.float()
        out = []
        corr_pos = []
        other_feat = []
        # added
        #attn_all = []
        for i in range(N):
            # sample_size x C x H x W
            # feat2: sample_size x N x C x H x W
            # feat2[:, i]: sample_size x C x H x W
            # sample_locs[:, i]: sample_size x H x W x 2

            # ---
            # ref2: sample_size x N x 3 x H x W
            # ref1: sample_size x N x 3 x H x W
            #added
            #attn_pad = torch.zeros(H*W, self.attn_heads, self.sample_size).to(feat1) 
            if loc_valid_mask[i, :].sum() == 0:
                # no eipolar line overlayed on the other view for any pixel
                feat_this = rearrange(feat1[i], 'c h w -> (h w) c').contiguous()
                out.append(feat_this)
                continue
            
            # tmp sample_size x C x H x W
            other1_sampled = F.grid_sample(feat2[:, i], sample_locs[:, i], padding_mode='zeros', align_corners=True)
            if self.sim_type == 'cos':
                feat1_this = feat1[i].flatten(1).unsqueeze(0)  # 1 x C x HW
                feat2_this = other1_sampled.flatten(2)   # sample_size x C x HW
                
                loc_valid_this = loc_valid_mask[i, :].view(-1)
                feat_1_att, feat_2_att = feat1_this[:, :, loc_valid_this], feat2_this[:, :, loc_valid_this]  # 1 x C x N_valid,  sample_size x C x N_valid
                sim = self.epipolar_similarity_1d(feat_1_att, feat_2_att) # sample_size x N_valid
                # weighted average
                #idx = sim.argmax(0)
                #with torch.no_grad():
                    # H x W x 2
                #    pos = torch.gather(sample_locs[:, i], 0, idx.view(1, H, W, 1).expand(-1, -1, -1, 2)).squeeze()
                #    pos = de_normalize(pos, H, W)
                #    corr_pos.append(pos)
                tmp = torch.zeros_like(feat1_this[0])
                tmp[:, loc_valid_this] = (feat_2_att * sim.unsqueeze(1)).sum(0)
                #tmp = rearrange(tmp, 'c h w -> (h w) c').contiguous()
                tmp = tmp.transpose(0, 1).contiguous()
                out.append(tmp)
            elif self.sim_type=='softmax':
                feat1_this = rearrange(feat1[i], 'c h w -> (h w) c').contiguous().unsqueeze(1) # HW x 1 x C
                feat2_this = rearrange(other1_sampled, 's c h w -> (h w) s c').contiguous() # HW x S x C
                # TODO: add filtering of no intersection
                loc_valid_this = loc_valid_mask[i, :].view(-1)
                feat_1_att, feat_2_att = feat1_this[loc_valid_this], feat2_this[loc_valid_this]  # N_valid x 1 x C, N_valid x S x C
                
                feat_2_att = feat_2_att + self.epi_pos_embed.unsqueeze(0)
                #attn_out, attn_weights = self.cross_attn(feat_1_att, feat_2_att)
                attn_out = self.cross_attn(feat_1_att, feat_2_att)
                feat_add = torch.zeros_like(feat1_this)
                feat_add[loc_valid_this] = attn_out
                #feat1_this[loc_valid_this] = feat_1_att + attn_out
                feat1_this = feat1_this + feat_add
                out.append(feat1_this.squeeze(1))   #
                # added for attn weights visualization
                #attn_pad[loc_valid_this] = attn_weights.squeeze(2)
                #attn_all.append(attn_pad)
            
        out = torch.stack(out)
        out = self.layernorm(out)
        out  = rearrange(out, 'n (h w) c -> n c h w', h=H, w=W).contiguous()
        #out = self.project(out)
        out = out + self.project(out)
        
        # added for attn weights visualization
        #attn_weights = torch.stack(attn_all)
        #return out, attn_weights, loc_valid_mask
        
        return out
    
    def epipolar_similarity_1d(self, feat1, sampled_feat2, cam1=None, cam2=None, softmax_enabled=True, prior_mul=False):
        """ 
        Args:
            feat1: 1 x C x L
            sampled_feat2: sample_size x C x L
        Return:
            sim: sample_size x L
        """
        sample_size, c, L = sampled_feat2.shape
        
        sim = (sampled_feat2 * feat1.expand(sample_size, -1, -1)).sum(1)
        sim[sim==0] = -1e10

        if softmax_enabled:
                # following https://arxiv.org/pdf/1706.03762.pdf d_k
                # TODO: the result is bad
                sim = sim * self.softmax_scale 
                sim = F.softmax(sim, 0)
                if prior_mul:
                    sim = sim * self.prior[(cam1, cam2)].to(sim)
        else:
            sim /= sample_size
            
        return sim  
    
    def epipolar_similarity(self, feat1, sampled_feat2, cam1=None, cam2=None, softmax_enabled=True, prior_mul=False):
        """ 
        Args:
            fea1: C, H, W
            sampled_feat2: sample_size, C, H, W
        Return:
            sim: sample_size H W
        """
        C, H, W = feat1.shape
        sample_size = sampled_feat2.shape[0]
       
        sim = (sampled_feat2 * feat1.view(1, C, H, W).expand(sample_size, -1, -1, -1)).sum(1)
        sim[sim==0] = -1e10

        if softmax_enabled:
                # following https://arxiv.org/pdf/1706.03762.pdf d_k
                # TODO: the result is bad
                sim = sim * self.softmax_scale 
                sim = F.softmax(sim, 0)
                if prior_mul:
                    sim = sim * self.prior[(cam1, cam2)].to(sim)
        else:
            sim /= sample_size
            
        return sim  
    
    
    
    def sample_locs(self, grid, H, W, fund_mat):
        """ get intersected pixel points on the other view
        """
        
        # F: N x 3 x 3        
        N = fund_mat.shape[0]

        # N x 3 x HW
        l2 = torch.matmul(fund_mat, grid.to(fund_mat))
        # N x HW x 3
        l2 = l2.transpose(1, 2)
        l2 = l2 / torch.sqrt(torch.sum(l2[..., :2]**2, -1, keepdim=True) + 1e-8) # normalize
        
        xmin = self.xmin.to(l2)
        xmax = self.xmax.to(l2)
        ymin = self.ymin.to(l2)
        ymax = self.ymax.to(l2)

        #numerical stability
        EPS = torch.tensor(self.epsilon).to(l2)
        by1 = -(xmin * l2[..., 0] + l2[..., 2]) / l2[..., 1]
        by2 = -(xmax * l2[..., 0] + l2[..., 2]) / l2[..., 1]
        bx0 = -(ymin * l2[..., 1] + l2[..., 2]) / l2[..., 0]
        bx3 = -(ymax * l2[..., 1] + l2[..., 2]) / l2[..., 0]
        # N x HW x 4
        intersections = torch.stack((
            bx0,
            by1,
            by2,
            bx3,
            ), -1)

        # N x HW x 4 x 2
        intersections = intersections.view(N, H*W, 4, 1).repeat(1, 1, 1, 2)
        intersections[..., 0, 1] = ymin
        intersections[..., 1, 0] = xmin
        intersections[..., 2, 0] = xmax
        intersections[..., 3, 1] = ymax
        # N x HW x 4
        mask = torch.stack((
            (bx0 >= xmin + self.epsilon) & (bx0 <  xmax - self.epsilon),
            (by1 >  ymin + self.epsilon) & (by1 <= ymax - self.epsilon),
            (by2 >= ymin + self.epsilon) & (by2 <  ymax - self.epsilon),
            (bx3 >  xmin + self.epsilon) & (bx3 <= xmax - self.epsilon),
            ), -1)
        # N x HW
        Nintersections = mask.sum(-1)
        loc_valid_mask = Nintersections >= 2
        loc_valid_mask = loc_valid_mask.view(N, H, W)
        
        # rule out all lines have no intersections
        mask[Nintersections < 2] = 0
        tmp_mask = mask.clone()
        tmp_mask[Nintersections < 2] = self.tmp_tensor.to(tmp_mask)
        # assert (Nintersections <= 2).all().item(), intersections[Nintersections > 2]
        # N x HW x 2 x 2
        valid_intersections = intersections[tmp_mask].view(N, H*W, 2, 2)
        valid_intersections[Nintersections < 2] = self.outrange_tensor.to(valid_intersections)
        # N x HW x 2
        start = valid_intersections[..., 0, :]
        vec = valid_intersections[..., 1, :] - start
        vec = vec.view(1, N, H*W, 2)
        # sample_size x N x HW x 2
        sample_locs = start.view(1, N, H*W, 2) + vec * self.sample_steps.to(vec)
        
        # normalize
        # sample_size*N x H x W x 2
        sample_locs = coord2pix_xy(sample_locs, self.downsample_x, self.downsample_y)
        sample_locs = normalize(sample_locs, H, W).view(-1, H, W, 2)
        sample_locs = sample_locs.view(self.sample_size, N, H, W, 2)
       
        return sample_locs, loc_valid_mask 
    
    