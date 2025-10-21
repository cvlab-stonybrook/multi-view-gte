import torch

def pix2coord(x, downsample):
    """convert pixels indices to real coordinates for 3D 2D projection
    """
    return x * downsample + downsample / 2.0 - 0.5

def coord2pix(y, downsample):
    """convert real coordinates to pixels indices for 3D 2D projection
    """
    # x * downsample + downsample / 2.0 - 0.5 = y
    return (y + 0.5 - downsample / 2.0) / downsample

def coord2pix_xy(coords, downsample_x, downsample_y):
    """convert real coordinates to pixels indices for 3D 2D projection
    """
    x, y = coords[..., 0], coords[..., 1]
    x_new = (x + 0.5 - downsample_x / 2.0) / downsample_x
    y_new = (y + 0.5 - downsample_y / 2.0) / downsample_y
    coords_new = torch.stack([x_new, y_new], dim=-1)
    # x * downsample + downsample / 2.0 - 0.5 = y
    return coords_new


def normalize(pts, H, W):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    pts[..., 0] = -1. + 2. * (pts[..., 0] + 0.5) / W
    pts[..., 1] = -1. + 2. * (pts[..., 1] + 0.5) / H
    return pts

def de_normalize(pts, H, W, engine='numpy'):
    """
    Args:
        pts: *N x 2 (x, y -> W, H)
    """
    
    pts[..., 0] = (pts[..., 0] + 1) * W / 2. - 0.5
    pts[..., 1] = (pts[..., 1] + 1) * H / 2. - 0.5
    return pts


def compute_epipolar_distance(pts_1, pts_2, F, patch_size=(64, 64), ori_size=(512, 384)):
    #pts_1: bs x 2
    bs = pts_1.size(0)
    
    downsample_x, downsample_y = ori_size[0] // patch_size[0], ori_size[1] // patch_size[1]
    
    pts1_x, pts1_y, pts2_x, pts2_y = pts_1[:, 0], pts_1[:, 1], pts_2[:, 0], pts_2[:, 1]
    pts1_y = pix2coord(pts1_y, downsample_y)
    pts1_x = pix2coord(pts1_x, downsample_x)   # 128 -> 512
    pts2_y = pix2coord(pts2_y, downsample_y)
    pts2_x = pix2coord(pts2_x, downsample_x)   # 128 -> 512
    pts1_ori, pts2_ori = torch.stack([pts1_x, pts1_y, torch.ones(bs).to(pts1_x)], dim=1), torch.stack([pts2_x, pts2_y, torch.ones(bs).to(pts2_x)], dim=1)
    
    epi_lines_2 = torch.matmul(F, pts1_ori.unsqueeze(-1)).squeeze(-1)  # bs x 3
    epi_dist = torch.abs(torch.sum(epi_lines_2 * pts2_ori, dim=1)) / torch.norm(epi_lines_2[:, :2], dim=1)
    epi_dist_normed = epi_dist / min(ori_size)
    
    return epi_dist_normed, epi_dist