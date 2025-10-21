import torch
import os
import numpy as np
from skimage.transform import resize
import cv2
import pdb
from PIL import Image
import torchvision.transforms.functional as TF
from torch.nn import functional as F
import albumentations as A

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray).float()
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def hm2binary(gaze_pt, hm_size):
    
    gaze_x, gaze_y = gaze_pt
    hm_width, hm_height = hm_size
    gt_hm = torch.zeros(hm_height, hm_width)
    gt_hm = draw_labelmap(gt_hm, [gaze_x * hm_width, gaze_y * hm_height], 3, type='Gaussian')
    gt_hm = (gt_hm > 0).float() * 1 # make GT heatmap as binary labels
    gt_hm = to_numpy(gt_hm)
    
    return gt_hm

def argmax_pts(heatmap):

    idx=np.unravel_index(heatmap.argmax(),heatmap.shape)
    pred_y,pred_x=map(float,idx)

    return pred_x,pred_y


def xywh2xyxy(bbox):
    isnumpy=False
    if type(bbox).__module__ == 'numpy':
        isnumpy=True
        
    bbox = to_torch(bbox)
    singlerow = False
    if len(bbox.shape)==1:
        singlerow=True
        bbox = bbox.unsqueeze(0)
    
    bbox[:, 2] = bbox[:,0] + bbox[:,2]
    bbox[:, 3] = bbox[:,1] + bbox[:,3]
    if singlerow:
        bbox.squeeze_(0)
    if isnumpy:
        bbox = to_numpy(bbox)
    return bbox


def spherical2cartesial(x):
    # convert from spehrical to cartesian coordinates    
    output = torch.zeros(x.size(0),3).to(x)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output


def unnorm(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if type(img)!=np.ndarray:
        img = img.cpu().numpy()
    std = np.array(std).reshape(3,1,1)
    mean = np.array(mean).reshape(3,1,1)
    return img * std + mean

def imgtensor_to_np(img, reverse_channel=True): 
    img = unnorm(img) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))
    if reverse_channel:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def init_image_coor(height, width, u0, v0):
    bs = u0.size(0)
    x_row = torch.arange(0, width).to(u0)
    x = x_row.repeat(height, 1).unsqueeze(0).expand(bs, -1, -1)    
    u_u0 = x - u0.unsqueeze(-1).unsqueeze(-1)

    y_col = torch.arange(0, height).to(u0)
    y = y_col.repeat(width, 1).T.unsqueeze(0).expand(bs, -1, -1)
    v_v0 = y - v0.unsqueeze(-1).unsqueeze(-1)
    return u_u0, v_v0

def depth_to_pcd(depth, u_u0, v_v0, fx, fy, invalid_value=0):
    # for getting 3d coordinates with 2 focal lengths for x and y
    mask_invalid = depth <= invalid_value
    depth[mask_invalid] = 0.0
    fx, fy = fx.unsqueeze(-1).unsqueeze(-1), fy.unsqueeze(-1).unsqueeze(-1)
    x = u_u0 / fx * depth
    y = v_v0 / fy * depth
    z = depth
    pcd = torch.stack([x, y, z], dim=1)
    return pcd

def point_to_line(point, line):
    x,y = point[:,0], point[:,1]
    a,b,c = line[:,0], line[:,1], line[:,2]
    distance = np.absolute((a*x + b*y + c)) / np.sqrt(a*a + b*b)
    return distance

def get_normalized_coordinates(coords, input_size):
    is_numpy = False if type(coords)==torch.Tensor else True
    coords = to_torch(coords)
    if len(coords)==0:
        coords = torch.tensor([-1,-1]).float()
    elif coords[0]!=-1:
        coords = torch.tensor([coords[0]/input_size[0], coords[1]/input_size[1]])
    if is_numpy:
        coords = to_numpy(coords)
    return coords


def getCamToEyeMatrix(dirEyes):
    # Define left? hand coordinate system in the eye plane orthogonal to the camera ray
    bs = dirEyes.size(0)
    upVector = torch.tensor([0,0,1]).to(dirEyes).unsqueeze(0).expand(bs, -1)
    zAxis = dirEyes.view(-1, 3)
    zAxis = zAxis /  torch.linalg.norm(zAxis, dim=1, keepdim=True)
    xAxis = torch.cross(upVector, zAxis, dim=1)
    xAxis = xAxis / torch.linalg.norm(xAxis, dim=1, keepdim=True)
    yAxis = torch.cross(zAxis, xAxis, dim=1)
    yAxis = yAxis / torch.linalg.norm(yAxis, dim=1, keepdim=True) # not really necessary
    gazeCS = torch.stack([xAxis, yAxis, zAxis], dim=1) # bs x 3 x 3
    return gazeCS

def get_gtvec_from_depth(eye_loc, tgt_gt, depth, intri, image_size=(224,224), eps=1e-8):
    # get 3d gaze vector from monocular depth map and eye/gaze target coordinates
    assert depth.size(1)==image_size[1] and depth.size(2)==image_size[0], "Depth map size does not match image size"
    fx, fy, u0, v0 = intri[:,0,0], intri[:,1,1], intri[:,0,2], intri[:,1,2]
    u_u0, v_v0 = init_image_coor(image_size[1], image_size[0], u0, v0)
    batch_idx = torch.arange(0, eye_loc.size(0)).to(depth).long()
    invalid_mask = torch.logical_and(eye_loc[:,0]==-1, eye_loc[:,1]==-1)
    image_width, image_height = image_size
    eye_loc_img = eye_loc.clone()
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] * image_width, eye_loc_img[:,1] * image_height
    eye_loc_img[invalid_mask,0] = 0
    eye_loc_img[invalid_mask,1] = 0
    eye_loc_img = torch.round(eye_loc_img).long()
    eye_loc_img[:,0] = torch.clip(eye_loc_img[:,0], 0, image_width-1)  # clip to image size
    eye_loc_img[:,1] = torch.clip(eye_loc_img[:,1], 0, image_height-1)
    depth_eye = depth[batch_idx, eye_loc_img[:,1], eye_loc_img[:,0]]
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] - u0, eye_loc_img[:,1] - v0 
    eye_x = eye_loc_img[:,0] / fx * depth_eye
    eye_y = eye_loc_img[:,1] / fy * depth_eye
    eye_3d = torch.stack((eye_x, eye_y, depth_eye), dim=1)
    
    gaze_coord_img = tgt_gt.clone()
    invalid_mask = torch.logical_and(gaze_coord_img[:,0]==-1, gaze_coord_img[:,1]==-1)
    gaze_coord_img[:,0], gaze_coord_img[:,1] = gaze_coord_img[:,0] * image_width, gaze_coord_img[:,1] * image_height
    gaze_coord_img[invalid_mask,0] = 0
    gaze_coord_img[invalid_mask,1] = 0
    gaze_coord_img = torch.round(gaze_coord_img).long()
    gaze_coord_img[:,0] = torch.clip(gaze_coord_img[:,0], 0, image_width-1)  # clip to image size
    gaze_coord_img[:,1] = torch.clip(gaze_coord_img[:,1], 0, image_height-1)
    depth_gazetgt = depth[batch_idx, gaze_coord_img[:,1], gaze_coord_img[:,0]]
    gaze_coord_img[:,0], gaze_coord_img[:,1] = gaze_coord_img[:,0] - u0, gaze_coord_img[:,1] - v0 
    gaze_x = gaze_coord_img[:,0] / fx * depth_gazetgt
    gaze_y = gaze_coord_img[:,1] / fy * depth_gazetgt
    gaze_3d = torch.stack((gaze_x, gaze_y, depth_gazetgt), dim=1)
    gt_vec = gaze_3d - eye_3d
    gt_vec = gt_vec / (torch.linalg.norm(gt_vec, dim=1, keepdim=True)+eps) 
    
    return gt_vec
    

def get_pcd_vecs(eye_loc, depth, intri, image_size=(224,224), fov_thres=0.9, eps=1e-8, tgt_gt=None):
    # get the ground truth gaze vector from point clouds, according to the eye and gaze target coordinates
    
    fov_thres = torch.tensor(fov_thres).to(eye_loc)
    fx, fy, u0, v0 = intri[:,0,0], intri[:,1,1], intri[:,0,2], intri[:,1,2]
    u_u0, v_v0 = init_image_coor(image_size[1], image_size[0], u0, v0)
    pcd = depth_to_pcd(depth, u_u0, v_v0, fx, fy) # bs, 3, 512, 384
    batch_idx = torch.arange(0, pcd.size(0)).to(depth).long()
    invalid_mask = torch.logical_and(eye_loc[:,0]<0, eye_loc[:,1]<0)
    image_width, image_height = image_size
    eye_loc_img = eye_loc.clone()
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] * image_width, eye_loc_img[:,1] * image_height
    eye_loc_img[invalid_mask,0] = 0
    eye_loc_img[invalid_mask,1] = 0
    eye_loc_img = torch.round(eye_loc_img).long()
    
    if torch.any(eye_loc_img[:,0] >= image_width) or torch.any(eye_loc_img[:,1] >= image_height):
        print(eye_loc_img)
    
    eye_loc_img[:,0] = torch.clip(eye_loc_img[:,0], 0, image_width-1)  # clip to image size
    eye_loc_img[:,1] = torch.clip(eye_loc_img[:,1], 0, image_height-1)
    
    depth_eye = depth[batch_idx, eye_loc_img[:,1], eye_loc_img[:,0]]
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] - u0, eye_loc_img[:,1] - v0 
    eye_x = eye_loc_img[:,0] / fx * depth_eye
    eye_y = eye_loc_img[:,1] / fy * depth_eye
    eye_3d = torch.stack((eye_x, eye_y, depth_eye), dim=1)
    pcd_vec = pcd - eye_3d.unsqueeze(-1).unsqueeze(-1)
    pcd_vec = pcd_vec.view(pcd_vec.size(0), 3, -1)
   
    # convert to eye coordinate system: following https://github.com/erkil1452/gaze360/issues/30#:~:text=edited-,def%20getGazeDirection(...)%3A,-%23%20Gaze%20direction%20in
    #eye_mat = getCamToEyeMatrix(eye_3d)
    #pcd_vec = torch.bmm(eye_mat, pcd_vec)
    pcd_vec = pcd_vec / (torch.linalg.norm(pcd_vec, dim=1, keepdim=True)+eps) 
    if tgt_gt is not None:
        tgt_invalid_mask = tgt_gt[:,0]==-1
        tgt_gt = tgt_gt.clone()
        tgt_gt[:,0], tgt_gt[:,1] = tgt_gt[:,0] * image_width, tgt_gt[:,1] * image_height
        tgt_gt[tgt_invalid_mask,:] = 0
        tgt_gt = tgt_gt.int()
        gt_vec = pcd_vec.view(-1, 3, image_size[1], image_size[0])
        tgt_gt[:, 0] = torch.clip(tgt_gt[:, 0], 0, image_width-1)
        tgt_gt[:, 1] = torch.clip(tgt_gt[:, 1], 0, image_height-1)
        tgt_gt = tgt_gt.long() 
        gt_vec = gt_vec[batch_idx, :, tgt_gt[:,1], tgt_gt[:,0]].clone()
    else:
        gt_vec  = torch.zeros(eye_loc.size(0), 3).to(eye_loc)
        
    return gt_vec, pcd, pcd_vec, eye_3d, eye_loc_img, invalid_mask



def get_fov_hm(eye_loc, gaze_vec, depth, intri, image_size=(224,224), fov_thres=0.9, eps=1e-8, tgt_gt=None, head_coords=None):
    
    gt_vec, pcd, pcd_vec, eye_3d, eye_loc_img, eye_invalid_mask = get_pcd_vecs(eye_loc, depth, intri, image_size, fov_thres, eps, tgt_gt)
    
    fov_thres = torch.tensor(fov_thres).to(eye_loc)
    gaze_vec = gaze_vec.unsqueeze(1)  # bs, 1, 3
    fov_hm = torch.bmm(gaze_vec, pcd_vec).squeeze(1)
    #fov_hm  = (fov_hm - fov_hm.min()) / (fov_hm.max() - fov_hm.min())

    mask = fov_hm <= fov_thres
    mask_assign = fov_thres * (torch.exp(5*fov_hm[mask]) / torch.exp(5*fov_thres))
    fov_hm[mask] = mask_assign.to(fov_hm)
    fov_hm = fov_hm.view(-1, image_size[1], image_size[0])
    fov_hm[eye_invalid_mask] = 0.0
    
    if head_coords is not None:
        fov_h, fov_w = fov_hm.size(1), fov_hm.size(2)
        head_fov_x1, head_fov_x2 = (head_coords[:,0]*fov_w).long(), (head_coords[:,2]*fov_w).long()
        head_fov_y1, head_fov_y2 = (head_coords[:,1]*fov_h).long(), (head_coords[:,3]*fov_h).long()
        head_fov_x1, head_fov_x2 = torch.clip(head_fov_x1, 0, fov_w-1).int(), torch.clip(head_fov_x2, 0, fov_w-1).int() 
        head_fov_y1, head_fov_y2 = torch.clip(head_fov_y1, 0, fov_h-1).int(), torch.clip(head_fov_y2, 0, fov_h-1).int()
        batch_idx = torch.arange(0, head_coords.size(0)).to(fov_hm).int()
        for b_i in range(fov_hm.size(0)):
            fov_hm[b_i, head_fov_y1[b_i]:head_fov_y2[b_i], head_fov_x1[b_i]:head_fov_x2[b_i]] = 0.0
    
    
    return fov_hm, gt_vec

def get_fov_hm_crossview(eye_loc, gaze_vec, depth_vhead, depth_vscene, intri_head, intri_scene, RT_htos, image_size=(224,224), fov_thres=0.9, eps=1e-8, scaleshift_vhead=None, scaleshift_vscene=None, tgt_gt=None):
    # load depth from both head and scene view along with scale shift so that we can convert 3d location from head view to scene view
    
    # convert to abs depth with scale and shift
    if scaleshift_vhead is not None:
        depth_vhead = depth_vhead * scaleshift_vhead[:,0].view(-1, 1, 1) + scaleshift_vhead[:,1].view(-1, 1, 1)
    if scaleshift_vscene is not None:
        depth_vscene = depth_vscene * scaleshift_vscene[:,0].view(-1, 1, 1) + scaleshift_vscene[:,1].view(-1, 1, 1)
    
    fov_thres = torch.tensor(fov_thres).to(gaze_vec)
    fx, fy, u0, v0 = intri_scene[:,0,0], intri_scene[:,1,1], intri_scene[:,0,2], intri_scene[:,1,2]
    u_u0, v_v0 = init_image_coor(image_size[1], image_size[0], u0, v0)
    pcd_scene = depth_to_pcd(depth_vscene, u_u0, v_v0, fx, fy) # bs, 3, 512, 384
    fx, fy, u0, v0 = intri_head[:,0,0], intri_head[:,1,1], intri_head[:,0,2], intri_head[:,1,2]
    u_u0, v_v0 = init_image_coor(image_size[1], image_size[0], u0, v0)
    pcd_head = depth_to_pcd(depth_vhead, u_u0, v_v0, fx, fy) # bs, 3, 512, 384
    
    batch_idx = torch.arange(0, pcd_head.size(0)).to(depth_vhead).long()
    invalid_mask = torch.logical_and(eye_loc[:,0]==-1, eye_loc[:,1]==-1)
    image_width, image_height = image_size
    eye_loc_img = eye_loc.clone()
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] * image_width, eye_loc_img[:,1] * image_height
    eye_loc_img[invalid_mask,0] = 0
    eye_loc_img[invalid_mask,1] = 0
    eye_loc_img = torch.round(eye_loc_img).long()
    #if torch.any(eye_loc_img[:,0] >= image_width) or torch.any(eye_loc_img[:,1] >= image_height):
    #    print(eye_loc_img)
    eye_loc_img = torch.round(eye_loc_img).long()
    eye_loc_img[:,0] = torch.clip(eye_loc_img[:,0], 0, image_width-1)  # clip to image size
    eye_loc_img[:,1] = torch.clip(eye_loc_img[:,1], 0, image_height-1)
     
    depth_eye = depth_vhead[batch_idx, eye_loc_img[:,1], eye_loc_img[:,0]]
    eye_loc_img[:,0], eye_loc_img[:,1] = eye_loc_img[:,0] - u0, eye_loc_img[:,1] - v0 
    eye_x = eye_loc_img[:,0] / fx * depth_eye
    eye_y = eye_loc_img[:,1] / fy * depth_eye
    eye_3d_head = torch.stack((eye_x, eye_y, depth_eye, torch.ones(eye_x.shape[0]).to(eye_x)), dim=1)
    eye_3d = torch.bmm(RT_htos, eye_3d_head.unsqueeze(-1)).squeeze(-1)  # rotate from head view to scene view
    
    pcd_vec = pcd_scene - eye_3d.unsqueeze(-1).unsqueeze(-1)
    pcd_vec = pcd_vec.view(pcd_vec.size(0), 3, -1) 
    pcd_vec = pcd_vec / (torch.linalg.norm(pcd_vec, dim=1, keepdim=True)+eps) 
    
    if tgt_gt is not None:
        tgt_invalid_mask = tgt_gt[:,0]==-1
        tgt_gt = tgt_gt.clone()
        tgt_gt[:,0], tgt_gt[:,1] = tgt_gt[:,0] * image_width, tgt_gt[:,1] * image_height
        tgt_gt[tgt_invalid_mask,:] = 0
        tgt_gt = tgt_gt.int()
        gt_vec = pcd_vec.view(-1, 3, image_size[1], image_size[0])
        tgt_gt[:, 0] = torch.clip(tgt_gt[:, 0], 0, image_width-1)
        tgt_gt[:, 1] = torch.clip(tgt_gt[:, 1], 0, image_height-1)
        tgt_gt = tgt_gt.long() 
        gt_vec = gt_vec[batch_idx, :, tgt_gt[:,1], tgt_gt[:,0]].clone()
    else:
        gt_vec  = torch.zeros(eye_loc.size(0), 3).to(eye_loc)
    
    
    gaze_vec = gaze_vec.unsqueeze(1)  # bs, 1, 3
    fov_hm = torch.bmm(gaze_vec, pcd_vec).squeeze(1)
    #fov_hm  = (fov_hm - fov_hm.min()) / (fov_hm.max() - fov_hm.min())

    mask = fov_hm <= fov_thres
    mask_assign = fov_thres * (torch.exp(5*fov_hm[mask]) / torch.exp(5*fov_thres))
    fov_hm[mask] = mask_assign.to(fov_hm)
    fov_hm = fov_hm.view(-1, image_size[1], image_size[0])
    
    return fov_hm, gt_vec



def get_bbox_mask(x_min, y_min, x_max, y_max, width, height, resolution, coordconv=False):
    bbox = np.array([x_min/width * resolution[0], y_min/height * resolution[1], x_max/width * resolution[0], y_max/height * resolution[1]])
    bbox = bbox.astype(int)
    bbox[0] = np.clip(bbox[0], 0, resolution[0]-1)
    bbox[1] = np.clip(bbox[1], 0, resolution[1]-1)
    bbox[2] = np.clip(bbox[2], 0, resolution[0]-1)
    bbox[3] = np.clip(bbox[3], 0, resolution[1]-1)
    if coordconv:
        unit = np.array(range(0,resolution), dtype=np.float32)
        head_channel = []
        for i in unit:
            head_channel.append([unit+i])
        head_channel = np.squeeze(np.array(head_channel)) / float(np.max(head_channel))
        head_channel[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 0
    else:
        head_channel = np.zeros((resolution[1],resolution[0]), dtype=np.float32)
        head_channel[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    head_channel = torch.from_numpy(head_channel)
    return head_channel


def draw_labelmap(img, pt, sigma_set, type='Gaussian', maximum=False):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)
    sigma = 3
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma_set ** 2))
    elif type == 'Cauchy':
        g = sigma_set / (((x - x0) ** 2 + (y - y0) ** 2 + sigma_set ** 2) ** 1.5)

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    if not maximum:
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] += g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    else:
        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(img[img_y[0]:img_y[1], img_x[0]:img_x[1]], g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    
    img = img/np.max(img) # normalize heatmap so it has max value of 1
    
    return to_torch(img)


def get_gt_gaze_from_other(gt_vec, RT, gt_available, num_views=2):
    gt_vec, gt_available, RT = gt_vec.view(-1, num_views, 3), gt_available.view(-1, num_views), RT.view(-1, num_views, 3, 4)
    # for the images that the gaze vector ground truth is not available, rotate from the other view which has ground truth
    bs = gt_vec.size(0)
    valid_mask_1, valid_mask_2 = gt_available[:,0]==1, gt_available[:,1]==1
    rotate_mask = torch.logical_and(~valid_mask_1, valid_mask_2)
    if rotate_mask.sum()>0:
        R_v1, R_v2 = RT[rotate_mask, 0, :, :3], RT[rotate_mask, 1, :, :3]
        gt_transformed = torch.bmm(R_v2.transpose(1,2), gt_vec[rotate_mask, 1].unsqueeze(-1))
        gt_transformed = torch.bmm(R_v1, gt_transformed).squeeze(-1)
        gt_transformed = F.normalize(gt_transformed, dim=1)
        gt_vec[rotate_mask, 0] = gt_transformed
    
    rotate_mask = torch.logical_and(~valid_mask_2, valid_mask_1)
    if rotate_mask.sum()>0:
        R_v1, R_v2 = RT[rotate_mask, 0, :, :3], RT[rotate_mask, 1, :, :3]
        gt_transformed = torch.bmm(R_v1.transpose(1,2), gt_vec[rotate_mask, 0].unsqueeze(-1))
        gt_transformed = torch.bmm(R_v2, gt_transformed).squeeze(-1)
        gt_transformed = F.normalize(gt_transformed, dim=1)
        gt_vec[rotate_mask, 1] = gt_transformed
    
    gt_vec = gt_vec.reshape(-1, 3)
    loss_mask_new = torch.logical_or(valid_mask_1, valid_mask_2)
    
    return gt_vec, loss_mask_new


def get_eye_keypoint(annt, conf_thres=1.2):
        eye_loc = annt['eye']
        if len(eye_loc)==0:
            eye_loc = np.array([-1.0,-1.0])
        else:
            eye_loc = np.array(annt['eye'])
            valid_mask = eye_loc[:,-1] > conf_thres
            eye_loc = eye_loc[valid_mask]
            if eye_loc.shape[0]==0:
                eye_loc = np.array([-1.0,-1.0])
            elif eye_loc.shape[0]==2:
                eye_loc = np.mean(eye_loc, axis=0)
            else:
                eye_loc = eye_loc[0]
        if eye_loc[0]==-1 and len(annt['head'])>0:
            # assign center of head box as eye location
            head_x_min, head_y_min, head_width, head_height = annt['head']
            head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            eye_loc = np.array([(head_x_min + head_x_max)/2, (head_y_min + head_y_max)/2])
            
        return eye_loc
    
def data_augmentation(img, body_box, head_box, body_valid, head_valid, width, height, test=False):
    if body_valid:
        body_x_min, body_y_min, body_width, body_height = body_box
        body_x_max, body_y_max = body_x_min + body_width, body_y_min + body_height
    else:
        body_x_min, body_y_min, body_x_max, body_y_max = -1, -1, -1, -1
    
    if head_valid:
        head_x_min, head_y_min, head_width, head_height = head_box
        head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
        k = 0.1   
    else:
        head_x_min, head_y_min,head_x_max, head_y_max = -1, -1, -1, -1
            
    body_jitter, head_jitter, crop, flip, colorchange = False, False, False, False, False

    if not test:
        if np.random.random_sample() <= 0.5 and body_valid:
            body_jitter = True
            k = np.random.random_sample() * 0.1
            body_x_min -= k * body_width
            body_y_min -= k * body_height
            body_x_max += k * body_width
            body_y_max += k * body_height
            
            body_x_min, body_y_min, body_x_max, body_y_max = np.clip(body_x_min, 0, width-1), np.clip(body_y_min, 0, height-1), np.clip(body_x_max, 0, width-1), np.clip(body_y_max, 0, height-1)
            body_width, body_height = body_x_max - body_x_min, body_y_max - body_y_min
            if body_width<=0 or body_height<=0:
                body_valid = False

        if np.random.random_sample() <= 0.5 and head_valid:
            head_jitter = True
            k = np.random.random_sample() * 0.1
            head_x_min -= k * head_width
            head_y_min -= k * head_height
            head_x_max += k * head_width
            head_y_max += k * head_height
            # constrain the edge of head box within the body box
            
            head_x_min, head_y_min, head_x_max, head_y_max = np.clip(head_x_min, 0, width-1), np.clip(head_y_min, 0, height-1), np.clip(head_x_max, 0, width-1), np.clip(head_y_max, 0, height-1)
            head_width, head_height = head_x_max - head_x_min, head_y_max - head_y_min
            if head_width<=0 or head_height<=0:
                head_valid = False
            
        # Random Crop # no crop and flip here
            
        if np.random.random_sample() <= 0.5:
            colorchange = True
            n1, n2, n3 = np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), np.random.uniform(0, 1.5)
            img = TF.adjust_brightness(img, brightness_factor=n1)
            img = TF.adjust_contrast(img, contrast_factor=n2)
            img = TF.adjust_saturation(img, saturation_factor=n3)
        
    body_box, head_box = np.array([body_x_min, body_y_min, body_x_max, body_y_max]), np.array([ head_x_min, head_y_min,head_x_max, head_y_max])
        
    return img, body_box, head_box 


def process_head(img, head_box, head_valid, width, height, img_out_size, head_out_size):
    head_x_min, head_y_min, head_x_max, head_y_max = head_box
    
    head_img = torch.zeros((3, head_out_size[1], head_out_size[0]))
    head_mask_scene = torch.zeros((1, img_out_size[1], img_out_size[0]))  # head w.r.t scene image
    
    if head_valid:
        head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * img_out_size[0]), round(head_y_min / height * img_out_size[1]), round(head_x_max / width * img_out_size[0]), round(head_y_max / height * img_out_size[1])
        head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
        head_img = img.crop((int(head_x_min), int(head_y_min), int(head_x_max), int(head_y_max))) 
        
    return head_img, head_mask_scene


def process_head_albument(img, head_box, head_valid, width, height, img_out_size, head_out_size):
    head_x_min, head_y_min, head_x_max, head_y_max = head_box
    
    head_img = np.zeros((head_out_size[1], head_out_size[0], 3))
    head_mask_scene = torch.zeros((1, img_out_size[1], img_out_size[0]))  # head w.r.t scene image
    
    if head_valid:
        head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * img_out_size[0]), round(head_y_min / height * img_out_size[1]), round(head_x_max / width * img_out_size[0]), round(head_y_max / height * img_out_size[1])
        head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
        head_img = img[int(head_y_min):int(head_y_max), int(head_x_min):int(head_x_max), :]
        
    return head_img, head_mask_scene 

def data_augmentation_albument(img, depth_img, head_box, gaze_coord, eye_loc, intri, width, height):
        
    x_min, y_min, x_max, y_max = head_box
    head_width, head_height = x_max - x_min, y_max - y_min
    
    head_valid = x_min>=0
    gaze_valid = gaze_coord[0]>=0
    depth_height, depth_width = depth_img.shape[:2]
    
    if np.random.random_sample() <= 0.5 and head_valid:
        head_jitter = True
        k = np.random.random_sample() * 0.1
        x_min -= k * head_width
        y_min -= k * head_height
        x_max += k * head_width
        y_max += k * head_height
        
        # constrain the edge of head box within the body box
        x_min, y_min, x_max, y_max = np.clip(x_min, 0, width-1), np.clip(y_min, 0, height-1), np.clip(x_max, 0, width-1), np.clip(y_max, 0, height-1)
        head_width, head_height = x_max - x_min, y_max - y_min
        if head_width<=0 or head_height<=0:
            head_valid = False
    
    gaze_x, gaze_y = gaze_coord
    # Random Crop # no flip here
    if np.random.random_sample() <= 0.5:
        crop = True
        if (not head_valid) and (not gaze_valid): 
            # neither head or gaze target is in the frame, randomly crop the image, but keep at least half of the image height or width
            crop_width, crop_height = np.random.randint(width//2, width), np.random.randint(height//2, height)
            crop_x_min = np.random.randint(0, width-crop_width) 
            crop_y_min = np.random.randint(0, height-crop_height)   
        else:
            # Calculate the minimum valid range of the crop that doesn't exclude the face and the gaze target
            gaze_x_bound = width-1 if gaze_x<0 else max(gaze_x - 20, 0)
            gaze_y_bound = height-1 if gaze_y<0 else max(gaze_y - 20, 0)
            xmin_bound = width-1 if x_min<0 else max(x_min - 20, 0)
            ymin_bound = height-1 if y_min<0 else min(y_min - 20, 0)
            
            gaze_x_upper = 0 if gaze_x<0 else min(gaze_x + 20, width-1)
            xmax_upper = 0 if x_max<0 else min(x_max + 20, width-1)
            gaze_y_upper = 0 if gaze_y<0 else min(gaze_y + 20, height-1)
            ymax_upper = 0 if y_max<0 else min(y_max + 20, height-1)
            
            crop_x_min = np.min([gaze_x_bound, xmin_bound])
            crop_y_min = np.min([gaze_y_bound, ymin_bound])
            crop_x_max = np.max([gaze_x_upper, xmax_upper])
            crop_y_max = np.max([gaze_y_upper, ymax_upper])
            crop_x_min, crop_y_min, crop_x_max, crop_y_max = map(int, [crop_x_min, crop_y_min, crop_x_max, crop_y_max])
            
            # Randomly select a random top left corner
            if crop_x_min<=0:
                crop_x_min = 0
            else:
                crop_x_min = np.random.randint(0, crop_x_min)
            if crop_y_min<=0:
                crop_y_min = 0
            else:
                crop_y_min = np.random.randint(0, crop_y_min)
    
            # Find the range of valid crop width and height starting from the (crop_x_min, crop_y_min)
            crop_width_min = crop_x_max - crop_x_min
            crop_height_min = crop_y_max - crop_y_min
            crop_width_max = width - crop_x_min
            crop_height_max = height - crop_y_min
            # Randomly select a width and a height
            if crop_width_min > crop_width_max or crop_height_min > crop_height_max:
                pdb.set_trace()
            
            if crop_width_min==crop_width_max:
                crop_width = crop_width_min
            else:
                crop_width = np.random.randint(crop_width_min, crop_width_max)
            if crop_height_min==crop_height_max:
                crop_height = crop_height_min
            else:
                crop_height = np.random.randint(crop_height_min, crop_height_max)
            
        crop_x_max, crop_y_max = crop_x_min + crop_width, crop_y_min + crop_height
        
        # Crop it
        img = A.Crop(crop_x_min, crop_y_min, crop_x_max, crop_y_max)(image=img)['image']
        crop_xmin_depth, crop_xmax_depth = int(crop_x_min * depth_width / width), int(crop_x_max * depth_width / width)
        crop_ymin_depth, crop_ymax_depth = int(crop_y_min * depth_height / height), int(crop_y_max * depth_height / height)
        
        depth_img = A.Crop(crop_xmin_depth, crop_ymin_depth, crop_xmax_depth, crop_ymax_depth)(image=depth_img)['image']

        # Record the crop's (x, y) offset
        offset_x, offset_y = crop_x_min, crop_y_min
        # convert coordinates into the cropped frame
        x_min, y_min, x_max, y_max = x_min - offset_x, y_min - offset_y, x_max - offset_x, y_max - offset_y
        if gaze_valid:
            gaze_x, gaze_y = gaze_x - offset_x, gaze_y - offset_y
            assert gaze_x>=0 and gaze_y>=0, print(gaze_x, gaze_y, offset_x, offset_y)
            gaze_coord = np.array([gaze_x, gaze_y])
        if eye_loc[0]>=0:
            eye_loc[0] = eye_loc[0] - offset_x
            eye_loc[1] = eye_loc[1] - offset_y
        
        width, height = crop_width, crop_height
        
        intri[0,2] -= offset_x
        intri[1,2] -= offset_y
    
    if head_valid:
        head_box = np.array([x_min, y_min, x_max, y_max])
        
    if np.random.random_sample() <= 0.5:
        colorchange = True
        transform = A.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.5, 1.5), hue=0.05)
        img = transform(image=img)['image']   
    
    return img, depth_img, head_box, gaze_coord, eye_loc, intri  # return the later 4 to apply consistent color changes

def get_transform(out_shape):
    transform_list = []
    transform_list.append(A.Resize(out_shape[0], out_shape[1], interpolation=cv2.INTER_AREA))
    transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    return A.Compose(transform_list)

 
    
def process_body_head(img, body_box, head_box, body_valid, head_valid, width, height, img_out_shape, body_img_shape, head_out_shape):
    body_x_min, body_y_min, body_x_max, body_y_max = body_box
    head_x_min, head_y_min, head_x_max, head_y_max = head_box
    body_width, body_height = body_x_max - body_x_min, body_y_max - body_y_min
    
    body_img = torch.zeros(3, body_img_shape[1], body_img_shape[0])    # use black image for padding
    head_img = torch.zeros((3, head_out_shape[1], head_out_shape[0]))
    head_mask_body = torch.zeros((1, body_img_shape[1], body_img_shape[0]))  # mask of the head w.r.t. the body
    head_mask_scene = torch.zeros((1, img_out_shape[1], img_out_shape[0]))  # head w.r.t scene image
    
    if body_valid:
        body_img = img.crop((int(body_x_min), int(body_y_min), int(body_x_max), int(body_y_max)))  
        body_mask = get_bbox_mask(body_x_min, body_y_min, body_x_max, body_y_max, width, height, resolution=img_out_shape).unsqueeze(0)
        body_out_h, body_out_w = body_img_shape
    else:
        body_mask = torch.zeros((1, img_out_shape[1], img_out_shape[0]))  # black mask for no body box
        
    if head_valid:
        if body_valid:
            # get mask of head relative to the body
            head_x1, head_y1, head_x2, head_y2 = max(head_x_min - body_x_min, 0), max(head_y_min - body_y_min, 0), head_x_max - body_x_min, head_y_max - body_y_min
            head_x1_mask, head_y1_mask, head_x2_mask, head_y2_mask = round(head_x1 / body_width * body_out_w), round(head_y1 / body_height * body_out_h), round(head_x2 / body_width * body_out_w), round(head_y2 / body_height * body_out_h)
            head_mask_body[:, head_y1_mask:head_y2_mask, head_x1_mask:head_x2_mask] = 1
        head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * img_out_shape[0]), round(head_y_min / height * img_out_shape[1]), round(head_x_max / width * img_out_shape[0]), round(head_y_max / height * img_out_shape[1])
        head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
        head_img = img.crop((int(head_x_min), int(head_y_min), int(head_x_max), int(head_y_max)))
        
    return body_img, body_mask, head_img, head_mask_body, head_mask_scene



# used in sharingan
def square_bbox(bboxes):
    """
    Adjust bounding boxes to be squared while ensuring the center of the box doesn't change.
    If the bounding box is too close to the edge, recenter the box to keep it within the image frame.

    Args:
        bboxes: a tensor of size (B, 4) containing B bounding boxes in the format [xmin, ymin, xmax, ymax]
        img_width: a scalar value indicating the width of the image
        img_height: a scalar value indicating the height of the image

    Returns:
        A tensor of size (B, 4) containing the squared bounding boxes.
    """
    n = len(bboxes)
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    # Calculate original widths and heights
    widths = xmax - xmin
    heights = ymax - ymin

    # Calculate centers
    center_x = xmin + widths / 2
    center_y = ymin + heights / 2

    # Calculate maximum side length
    max_side_length = torch.max(widths, heights)

    # Calculate new xmin, ymin, xmax, ymax
    new_xmin = center_x - max_side_length / 2
    new_ymin = center_y - max_side_length / 2
    new_xmax = center_x + max_side_length / 2
    new_ymax = center_y + max_side_length / 2

    # Create the squared bounding boxes
    squared_bboxes = torch.stack([new_xmin, new_ymin, new_xmax, new_ymax], dim=1)

    return squared_bboxes


def interpolate_pos_embed_2d(checkpoint_model, new_patch_h, new_patch_w, ori_patch_h=14, ori_patch_w=14):	

	pos_embed_checkpoint = checkpoint_model['image_tokenizer.pos_emb']
	#embedding_size = pos_embed_checkpoint.shape[1]
	
	num_patches = new_patch_h * new_patch_w
	#num_extra_tokens = model.pos_embed.shape[-2] - num_patches
	
	#orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
	orig_size = (ori_patch_h, ori_patch_w)
	#import pdb
	#pdb.set_trace()
	# height (== width) for the new position embedding
	#new_size = int(num_patches ** 0.5)
	new_size = (new_patch_h, new_patch_w)
	# class_token and dist_token are kept unchanged
	if orig_size != new_size:
		print("Position interpolate from %dx%d to %dx%d" % (ori_patch_h, ori_patch_w, new_patch_h, new_patch_w))
		# only the position tokens are interpolated
		pos_tokens = pos_embed_checkpoint
		#pos_tokens = pos_tokens.reshape(-1, ori_patch_h, ori_patch_w, embedding_size).permute(0, 3, 1, 2)
		pos_tokens = torch.nn.functional.interpolate(
			pos_tokens, size=(new_patch_h, new_patch_w), mode='bicubic', align_corners=False)
		#pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
		#new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
		new_pos_embed = pos_tokens
		checkpoint_model['image_tokenizer.pos_emb'] = new_pos_embed