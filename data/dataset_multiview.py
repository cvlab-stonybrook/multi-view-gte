import torch
import os
import albumentations as A
import json
import copy
import numpy as np
import cv2
import pickle
import pdb
import pandas as pd
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation 
from multiview.camera_utils import read_camera
from utils.utils import draw_labelmap, data_augmentation_albument, square_bbox
from utils.file_utils import read_json
import torch.nn.functional as F




eps = 1e-9
class Gaze_Dataset_Multiview_CrossScene_Head(Dataset):
    def __init__(self, base_dir, img_out_size, head_out_size, hm_size, eval_scene='Lab', test=False, no_aug=False, adapt=False):
        # note img_out_size and hm_size are int as they are squares, body_img_size is tuple as body img is rectangle
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'Data')
        annt_dir = os.path.join(self.base_dir, 'Annotations_facevis')
        self.depth_dir = os.path.join(base_dir, 'Depth_Metric3d')
        
        if test:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        self.eval_scene = eval_scene    
        all_subjs = sorted(glob(os.path.join(self.data_dir, '*', '*')))   # data/scene/subj
        test_subjs = sorted(glob(os.path.join(self.data_dir, eval_scene, '*')))
        train_subjs = [each_subj for each_subj in all_subjs if each_subj not in test_subjs]
        if adapt:
            adapt_path = os.path.join(base_dir, "Split_info", "fewshot_adapt.txt")
            df = pd.read_csv(adapt_path, names=['scene', 'adapt_subj'], sep=' ')
            adapt_subj = df[df['scene']==eval_scene]['adapt_subj'].values[0]
            print(f"Adapt on {adapt_subj}")
            adapt_subj = os.path.join(self.data_dir, eval_scene, adapt_subj)
            test_subjs.remove(adapt_subj)
            train_subjs = [adapt_subj]
            #train_subjs.append(adapt_subj)
            
        if test:
            subj_folders = test_subjs
            print(f"{self.mode} on {eval_scene},  {len(test_subjs)} subjects")
        else:
            subj_folders = train_subjs
            print(f"Train on {len(train_subjs)} subjects.")
        
        self.cam_params = {}
        self.annotations_dct = {}
        self.input_list = []
        self.annt_list = []
        self.cam_params_list = []
        self.img_index = []  # indices of the image of the main view, for evaluation
        img_index = 0
        self.gaze_3d_list = []
        self.reproj_err_thres = 35.0
        
        for folder_path in subj_folders:
            scene, subj_folder = folder_path.split('/')[-2], folder_path.split('/')[-1]
            cam_params = None
            if os.path.exists(os.path.join(folder_path, "Calibration", 'extri.yml')):
                extri_path = os.path.join(folder_path, 'Calibration', 'extri.yml')
                intri_path = os.path.join(folder_path, 'Calibration', 'intri.yml')
                cam_params = read_camera(intri_path, extri_path)
            else:
                # in this case, one subject has multiple calibrations (due to camera movement, etc)
                all_calibs = []
                all_calib_names = sorted([foldername for foldername in os.listdir(os.path.join(folder_path)) if foldername.startswith('Calib')])
                calib_ranges = []
                for calib_folder in all_calib_names:
                    ranges = calib_folder.split('_')[1].split('to')
                    calib_ranges.append((int(ranges[0]), int(ranges[1])))
                    extri_path = os.path.join(folder_path, calib_folder, 'extri.yml')
                    intri_path = os.path.join(folder_path, calib_folder, 'intri.yml')
                    all_calibs.append(read_camera(intri_path, extri_path))
                
            annt_json = os.path.join(annt_dir, scene, f'{subj_folder}.json')
            with open(annt_json, 'r') as file:
                annt_subj = json.load(file)
            
            # add for triangulated 3d points
            annt3d_path = os.path.join(annt_dir, scene, "triangulate_3d", f"{subj_folder}.json")
            annt_3d = read_json(annt3d_path)
                
            all_cams = sorted([key for key in list(annt_subj[0].keys()) if key.startswith('Cam')])
            for idx, annt_img in enumerate(annt_subj):
                filename = annt_img['filename']
                annt_copy = copy.deepcopy(annt_subj[idx])
                annt_copy.pop('filename')
                self.annotations_dct[(scene, subj_folder, filename)] = annt_copy
                
                filename_noext = os.path.splitext(filename)[0]
                assert annt_3d[idx]['filename'] == filename_noext, print(annt_3d[idx]['filename'], filename_noext, idx)
                eye_3d, tgt_3d = annt_3d[idx]['eye'], annt_3d[idx]['target']
                eye_3d_valid, tgt3d_valid = True, True
                if len(eye_3d)==0 or annt_3d[idx]['eye_err'] > self.reproj_err_thres:
                    eye_3d = torch.zeros(3)
                    eye_3d_valid = False
                else:
                    eye_3d = torch.tensor(eye_3d).float()
                
                if len(tgt_3d)==0 or annt_3d[idx]['target_err'] > self.reproj_err_thres:
                    tgt_3d = torch.zeros(3)
                    tgt3d_valid = False
                else:
                    tgt_3d = torch.tensor(tgt_3d).float()
                
                if eye_3d_valid and tgt3d_valid:
                    gaze_vec = tgt_3d - eye_3d
                else:   # for training gaze estimator: just keep the ones with valid 3d gaze vector
                    gaze_vec = torch.zeros(3)
                
                cam_params_this = None
                if cam_params is not None:
                    cam_params_this = cam_params
                else:
                    if int(os.path.splitext(filename)[0])>calib_ranges[-1][1]:
                        pdb.set_trace()
                        raise NotImplementedError
                    for this_idx, ranges in enumerate(calib_ranges):
                        if int(os.path.splitext(filename)[0])<=ranges[1]:
                            cam_params_this = all_calibs[this_idx]
                            break
                # treat each pair of camera view as one sample of input
                # note: Here each image from a camera is paired with a image from every other camera, we only treat the first image and the main view
                for idx_1 in range(0, len(all_cams)):
                    cam_1 = all_cams[idx_1]
                    if len(annt_copy[cam_1]['head'])==0:  # no head annotated
                        continue
                    for idx_2 in range(0, len(all_cams)):
                        if idx_2==idx_1:
                            continue
                        cam_2 = all_cams[idx_2]
                        self.input_list.append((scene, subj_folder, cam_1, cam_2, filename))
                        self.cam_params_list.append(cam_params_this)
                        self.img_index.append(img_index)
                        self.gaze_3d_list.append(gaze_vec)
                    img_index += 1
                        
        self.vis_mapping = {'false':0, 'true':1, 'occlusion':2}
        self.img_out_size = img_out_size
        self.head_out_size = head_out_size
        self.hm_size = hm_size
        self.test = test     
        self.no_aug = no_aug   # if set to True then no data augmentation will be performed   
        
        self.head_transform = self.get_transform((self.head_out_size[1], self.head_out_size[0]))
        self.img_transform = self.get_transform((self.img_out_size[1], self.img_out_size[0]))
        #self.head_transform, self.img_transform = self.get_transform(), self.get_transform()
        
        def skew_op(x):
            res = np.zeros((3, 3), dtype=x.dtype)
            # 0, -z, y
            res[0, 1] = -x[2, 0]
            res[0, 2] =  x[1, 0]
            # z, 0, -x
            res[1, 0] =  x[2, 0]
            res[1, 2] = -x[0, 0]
            # -y, x, 0
            res[2, 0] = -x[1, 0]
            res[2, 1] =  x[0, 0]
            return res 
        # multi-view geometry
        self.skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        #self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            #R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_op = lambda K_1, R_1, T_1, K_0, R_0, T_0: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: self.fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3:], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3:] )
        
        self.annt_errors = set()
        
    def __getitem__(self, index):
        scene, subj_folder, cam_1, cam_2, filename = self.input_list[index]
        main_view_index = self.img_index[index]
        
        subj_path = os.path.join(self.data_dir, scene, subj_folder, "Images")
        img_path_1, img_path_2 = os.path.join(subj_path, cam_1, filename), os.path.join(subj_path, cam_2, filename)
        annt = self.annotations_dct[(scene, subj_folder, filename)]
        gaze_coord_1, gaze_coord_2 = annt[cam_1]['coordinate'], annt[cam_2]['coordinate']
        head_box_1, head_box_2 = annt[cam_1]['head'], annt[cam_2]['head']
        

        head_valid_1 = False if len(head_box_1)==0 else True
        head_valid_2 = False if len(head_box_2)==0 else True
        
        facevis_1  = annt[cam_1]['Face_vis'] if head_valid_1 else -2
        facevis_2  = annt[cam_2]['Face_vis'] if head_valid_2 else -2
            
        imgname = os.path.splitext(filename)[0]
        depth_path_1, depth_path_2 = os.path.join(self.depth_dir, scene, subj_folder, cam_1, f"{imgname}.npy"), os.path.join(self.depth_dir, scene, subj_folder, cam_2, f"{imgname}.npy")
        depth_v1, depth_v2 = np.load(depth_path_1), np.load(depth_path_2)
                 
        visib_name_1, visib_name_2 = annt[cam_1]['visibility'].lower(), annt[cam_2]['visibility'].lower()
        visib_1, visib_2 = self.vis_mapping[visib_name_1], self.vis_mapping[visib_name_2]

        #self.body_transform = self.get_transform((self.body_img_shape[1], self.body_img_shape[0]))
        img_1, img_2 = cv2.imread(img_path_1), cv2.imread(img_path_2)
        img_1, img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        
        head_box_list = [head_box_1, head_box_2]
        for idx, head_box in enumerate(head_box_list):
            head_valid = head_valid_1 if idx==0 else head_valid_2
            if head_valid:
                head_x_min, head_y_min, head_width, head_height = head_box
                head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            else:
                head_x_min, head_y_min, head_x_max, head_y_max = -1, -1, -1, -1
            head_box = np.array([ head_x_min, head_y_min,head_x_max, head_y_max])
            head_box_list[idx] = head_box
        head_box_1, head_box_2 = head_box_list
        
        if len(gaze_coord_1)==0:
            gaze_x1, gaze_y1 = -1, -1
            gaze_valid_1 = False
        else:
            gaze_valid_1 = True
            gaze_x1, gaze_y1 = gaze_coord_1[0], gaze_coord_1[1]
        gaze_coord_1 = np.array([gaze_x1, gaze_y1])
        
        if len(gaze_coord_2)==0:
            #assert visib_2!=1, img_path_2
            gaze_x2, gaze_y2 = -1, -1
            gaze_valid_2 = False
        else:
            #assert visib_2!=0, img_path_2
            gaze_valid_2 = True
            gaze_x2, gaze_y2 = gaze_coord_2[0], gaze_coord_2[1]

        gaze_coord_2 = np.array([gaze_x2, gaze_y2])        
        eye_loc_v1, eye_loc_v2 = self.get_eye_keypoint(annt[cam_1]), self.get_eye_keypoint(annt[cam_2])
        eye_loc_head_v1, eye_loc_head_v2 = torch.tensor([-1, -1]).float(), torch.tensor([-1, -1]).float()
        
        cam_params = copy.deepcopy(self.cam_params_list[index])
        RT_1, RT_2 = cam_params[cam_1]['RT'], cam_params[cam_2]['RT']
        intri_1, intri_2 = cam_params[cam_1]['K'], cam_params[cam_2]['K']    
    
        height_1, width_1 = img_1.shape[:2]
        height_2, width_2 = img_2.shape[:2] 
        
        aug = not (self.test or self.no_aug)
        head_img_1, head_mask_scene_1, head_box_1 = self.data_augmentation_albument_nocrop(img_1, head_box_1, width_1, height_1, aug=aug)
        head_img_2, head_mask_scene_2, head_box_2 = self.data_augmentation_albument_nocrop(img_2, head_box_2, width_2, height_2, aug=aug)
        
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1]
        if gaze_x1>=0 and gaze_y1>=0:
            gaze_coord_1 = torch.tensor([gaze_x1 / width_1, gaze_y1 / height_1]).float()
        if gaze_x2>=0 and gaze_y2>=0:
            gaze_coord_2 = torch.tensor([gaze_x2 / width_2, gaze_y2 / height_2]).float()
        
        
        head_coords_1 = torch.tensor([head_box_1[0]/width_1, head_box_1[1]/height_1, head_box_1[2]/width_1, head_box_1[3]/height_1]).float()
        head_coords_2 = torch.tensor([head_box_2[0]/width_2, head_box_2[1]/height_2, head_box_2[2]/width_2, head_box_2[3]/height_2]).float()
        
        eye_loc_v1, eye_loc_v2 = self.get_eye_keypoint(annt[cam_1]), self.get_eye_keypoint(annt[cam_2])
        eye_loc_head_v1, eye_loc_head_v2 = torch.tensor([-1, -1]).float(), torch.tensor([-1, -1]).float()
        if eye_loc_v1[0]!=-1.0:
            eye_loc_head_v1 = torch.tensor([(eye_loc_v1[0]-head_box_1[0])/(head_box_1[2]-head_box_1[0]), (eye_loc_v1[1]-head_box_1[1])/(head_box_1[3]-head_box_1[1])]).float()
            eye_loc_v1 = np.array([eye_loc_v1[0]/width_1, eye_loc_v1[1]/height_1])
        if eye_loc_v2[0]!=-1.0:
            eye_loc_head_v2 = torch.tensor([(eye_loc_v2[0]-head_box_2[0])/(head_box_2[2]-head_box_2[0]), (eye_loc_v2[1]-head_box_2[1])/(head_box_2[3]-head_box_2[1])]).float()
            eye_loc_v2 = np.array([eye_loc_v2[0]/width_2, eye_loc_v2[1]/height_2])
        eye_loc_v1, eye_loc_v2 = torch.tensor(eye_loc_v1).float(), torch.tensor(eye_loc_v2).float()    
        
    
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1] 
        gaze_heatmap_1, gaze_heatmap_2 = torch.zeros(self.hm_size[1], self.hm_size[0]), torch.zeros(self.hm_size[1], self.hm_size[0])
        if gaze_valid_1 and visib_1!=0:
            gaze_heatmap_1 = draw_labelmap(gaze_heatmap_1, [gaze_x1 * self.hm_size[0], gaze_y1 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        if gaze_valid_2 and visib_2!=0:
            gaze_heatmap_2 = draw_labelmap(gaze_heatmap_2, [gaze_x2 * self.hm_size[0], gaze_y2 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        
        img_1, img_2 = self.img_transform(image=img_1)["image"], self.img_transform(image=img_2)['image']
        if head_valid_1:
            head_img_1 = self.head_transform(image=head_img_1)["image"]
        if head_valid_2:
            head_img_2 = self.head_transform(image=head_img_2)["image"]
        
        head_img_1, head_img_2 = torch.tensor(head_img_1).float(), torch.tensor(head_img_2).float()   
        head_img_1, head_img_2 = head_img_1.permute(2, 0, 1), head_img_2.permute(2, 0, 1)
        gaze_coord_1, gaze_coord_2 = torch.tensor([gaze_x1, gaze_y1]).float(), torch.tensor([gaze_x2, gaze_y2]).float()
        visib_1, visib_2 = torch.tensor(visib_1).long(), torch.tensor(visib_2).long()
        head_valid_1, head_valid_2 = torch.tensor(head_valid_1).long(), torch.tensor(head_valid_2).long()
        
        depth_v1, depth_v2 = torch.tensor(depth_v1).float(), torch.tensor(depth_v2).float()
        if depth_v1.size(0)!=self.img_out_size[1] or depth_v1.size(1)!=self.img_out_size[0]:
            depth_v1 = F.interpolate(depth_v1.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if depth_v2.size(0)!=self.img_out_size[1] or depth_v2.size(1)!=self.img_out_size[0]:
            depth_v2 = F.interpolate(depth_v2.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        
        intri1_head, intri2_head = intri_1.copy(), intri_2.copy()
        if head_valid_1:
            intri1_head[0, 2] -= head_box_1[0]
            intri1_head[1, 2] -= head_box_1[1]
        if head_valid_2:
            intri2_head[0, 2] -= head_box_2[0]
            intri2_head[1, 2] -= head_box_2[1] 
        ratio_w, ratio_h = self.head_out_size[0] / width_1, self.head_out_size[1]/height_1
        intri1_head[0,:] = intri1_head[0,:] * ratio_w
        intri1_head[1,:] = intri1_head[1,:] * ratio_h
        ratio_w, ratio_h = self.head_out_size[0] / width_2, self.head_out_size[1]/height_2
        intri2_head[0,:] = intri2_head[0,:] * ratio_w
        intri2_head[1,:] = intri2_head[1,:] * ratio_h
        fund_mat_head = torch.tensor(self.fundamental_RT_op(intri1_head, RT_1, intri2_head, RT_2)).float()
        
        # process camera parameters
        ratio_w, ratio_h = self.img_out_size[0] / width_1, self.img_out_size[1]/height_1
        intri_1[0,:] = intri_1[0,:] * ratio_w
        intri_1[1,:] = intri_1[1,:] * ratio_h
        ratio_w, ratio_h = self.img_out_size[0] / width_2, self.img_out_size[1]/height_2
        intri_2[0,:] = intri_2[0,:] * ratio_w
        intri_2[1,:] = intri_2[1,:] * ratio_h
    
        fund_mat = torch.tensor(self.fundamental_RT_op(intri_1,RT_1, intri_2, RT_2)).float()
        intri_1, intri_2 = torch.from_numpy(intri_1).float(), torch.from_numpy(intri_2).float()
        RT_1, RT_2 = torch.from_numpy(RT_1).float(), torch.from_numpy(RT_2).float()
        R1, R2 = RT_1[:, :3], RT_2[:, :3]
        # convert to quaternion
        R_2to1, R_1to2 = R1 @ R2.T, R2 @ R1.T
        Rot_1, Rot_2 = Rotation.from_matrix(R_2to1.numpy()), Rotation.from_matrix(R_1to2.numpy())
        quat_1, quat_2 = torch.tensor(Rot_1.as_quat()).float(), torch.tensor(Rot_2.as_quat()).float()
        
        gaze_3d = self.gaze_3d_list[index]
        if torch.all(gaze_3d==0):
            gtvec_3d_1, gtvec_3d_2 = torch.zeros(3).float(), torch.zeros(3).float()
        else:
            gaze_3d_1 = R1 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_1 = torch.linalg.norm(gaze_3d_1)
            gtvec_3d_1 = gaze_3d_1 / (gaze_3d_norm_1 + eps)
            gaze_3d_2 = R2 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_2 = torch.linalg.norm(gaze_3d_2)
            gtvec_3d_2 = gaze_3d_2 / (gaze_3d_norm_2 + eps)
        
        head_img, head_mask_scene, depth = torch.stack((head_img_1, head_img_2)), torch.stack((head_mask_scene_1, head_mask_scene_2)), torch.stack((depth_v1, depth_v2))
        gaze_heatmap, visib, eye_loc, gaze_coord, head_valid = torch.stack((gaze_heatmap_1, gaze_heatmap_2)), torch.stack((visib_1, visib_2)), torch.stack((eye_loc_v1, eye_loc_v2)), torch.stack((gaze_coord_1, gaze_coord_2)), torch.stack((head_valid_1, head_valid_2))
        eye_loc_head = torch.stack((eye_loc_head_v1, eye_loc_head_v2))
        gtvec_3d = torch.stack((gtvec_3d_1, gtvec_3d_2))
        head_coords = torch.stack((head_coords_1, head_coords_2))
        intri, RT = torch.stack((intri_1, intri_2)), torch.stack((RT_1, RT_2))
        path_info = os.path.join(scene, subj_folder, cam_1, cam_2, filename)
        quat = torch.stack((quat_1, quat_2))
        facevis = torch.tensor([facevis_1, facevis_2]).int()
        
        
        data_dict = {
            "data": (head_img, head_mask_scene, depth, gaze_heatmap, visib, gaze_coord, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT),
            "fund_mat": fund_mat,
            "fund_mat_head": fund_mat_head,
            "path": path_info,
            'main_id': main_view_index,
            'face_vis': facevis
        }
        
        return data_dict
         
    def __len__(self):
        return len(self.input_list)
    
    def get_transform(self, out_shape):
        transform_list = []
        transform_list.append(A.Resize(out_shape[0], out_shape[1], interpolation=cv2.INTER_AREA))
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return A.Compose(transform_list)

    
    def get_eye_keypoint(self, annt, conf_thres=1.2):
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
    
    def data_augmentation_albument_nocrop(self, img, head_box, width, height, aug=False):
        
        x_min, y_min, x_max, y_max = head_box
        head_width, head_height = x_max - x_min, y_max - y_min
        
        head_valid = x_min>=0
        
        if aug:
            if np.random.random_sample() <= 0.5 and head_valid:
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
        if head_valid:
            head_box = np.array([x_min, y_min, x_max, y_max])
        
        # crop head
        head_img = np.zeros((self.head_out_size[1], self.head_out_size[0], 3))
        head_mask_scene = torch.zeros((1, self.img_out_size[1], self.img_out_size[0]))  # head w.r.t scene image
        
        if head_valid:
            head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(x_min / width * self.img_out_size[0]), round(y_min / height * self.img_out_size[1]), round(x_max / width * self.img_out_size[0]), round(y_max / height * self.img_out_size[1])
            head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
            #head_img = img.crop((int(head_x_min), int(head_y_min), int(head_x_max), int(head_y_max)))
            head_img = img[int(y_min):int(y_max), int(x_min):int(x_max), :]
    
        if aug and head_valid and np.random.random_sample() <= 0.5:
            transform = A.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7, 1.3), saturation=(0.5, 1.5), hue=0.05)
            head_img = transform(image=head_img)['image']   
        
        return head_img, head_mask_scene, head_box  # return the later 4 to apply consistent color changes




class Gaze_Dataset_Multiview_CrossScene(Dataset):
    def __init__(self, base_dir, img_out_size, head_out_size, hm_size, eval_scene='Lab', test=False, no_aug=False, adapt=False, adapt_idx=0):
        # note img_out_size and hm_size are int as they are squares, body_img_size is tuple as body img is rectangle
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'Data')
        annt_dir = os.path.join(self.base_dir, 'Annotations')
        print("annt_dir: ", annt_dir)
        self.depth_dir = os.path.join(base_dir, 'Depth_Metric3d')
        if test:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        self.eval_scene = eval_scene    
        all_subjs = sorted(glob(os.path.join(self.data_dir, '*', '*')))   # data/scene/subj
        test_subjs = sorted(glob(os.path.join(self.data_dir, eval_scene, '*')))
        train_subjs = [each_subj for each_subj in all_subjs if each_subj not in test_subjs]
            
        if not adapt:
            if test:
                subj_folders = test_subjs   
                #print(f"{self.mode} on {len(subj_folders)} subjects")
            else:
                subj_folders = train_subjs
                #print(f"Train on {len(subj_folders)} subject folders.")
            print(f"{self.mode} on {eval_scene},  {len(subj_folders)} subjects")
        else:
            print(f"Evaluate on {test_subjs[adapt_idx]}")
            if test:
                subj_folders =[test_subjs[adapt_idx]]
            else:
                test_subjs.remove(test_subjs[adapt_idx]) 
                subj_folders = test_subjs
        
        self.cam_params = {}
        self.annotations_dct = {}
        self.input_list = []
        self.annt_list = []
        self.cam_params_list = []
        self.img_index = []  # indices of the image of the main view, for evaluation
        img_index = 0
        self.gaze_3d_list = []
        self.reproj_err_thres = 35.0
        
        for folder_path in subj_folders:
            scene, subj_folder = folder_path.split('/')[-2], folder_path.split('/')[-1]
            #if scene!='Kitchen' and subj_folder not in ['07_03', '07_12_01', '07_06_02']:
            #    continue
            
            cam_params = None
            if os.path.exists(os.path.join(folder_path, "Calibration", 'extri.yml')):
                extri_path = os.path.join(folder_path, 'Calibration', 'extri.yml')
                intri_path = os.path.join(folder_path, 'Calibration', 'intri.yml')
                cam_params = read_camera(intri_path, extri_path)
            else:
                # in this case, one subject has multiple calibrations (due to camera movement, etc)
                all_calibs = []
                all_calib_names = sorted([foldername for foldername in os.listdir(os.path.join(folder_path)) if foldername.startswith('Calib')])
                calib_ranges = []
                for calib_folder in all_calib_names:
                    ranges = calib_folder.split('_')[1].split('to')
                    calib_ranges.append((int(ranges[0]), int(ranges[1])))
                    extri_path = os.path.join(folder_path, calib_folder, 'extri.yml')
                    intri_path = os.path.join(folder_path, calib_folder, 'intri.yml')
                    all_calibs.append(read_camera(intri_path, extri_path))
                
            annt_json = os.path.join(annt_dir, scene, f'{subj_folder}.json')
            with open(annt_json, 'r') as file:
                annt_subj = json.load(file)
            
            # add for triangulated 3d points
            annt3d_path = os.path.join(annt_dir, scene, "triangulate_3d", f"{subj_folder}.json")
            annt_3d = read_json(annt3d_path)
                
            all_cams = sorted([key for key in list(annt_subj[0].keys()) if key.startswith('Cam')])
            for idx, annt_img in enumerate(annt_subj):
                filename = annt_img['filename']
                annt_copy = copy.deepcopy(annt_subj[idx])
                annt_copy.pop('filename')
                self.annotations_dct[(scene, subj_folder, filename)] = annt_copy
                
                filename_noext = os.path.splitext(filename)[0]
                assert annt_3d[idx]['filename'] == filename_noext, print(annt_3d[idx]['filename'], filename_noext, idx)
                eye_3d, tgt_3d = annt_3d[idx]['eye'], annt_3d[idx]['target']
                eye_3d_valid, tgt3d_valid = True, True
                if len(eye_3d)==0 or annt_3d[idx]['eye_err'] > self.reproj_err_thres:
                    eye_3d = torch.zeros(3)
                    eye_3d_valid = False
                else:
                    eye_3d = torch.tensor(eye_3d).float()
                
                if len(tgt_3d)==0 or annt_3d[idx]['target_err'] > self.reproj_err_thres:
                    tgt_3d = torch.zeros(3)
                    tgt3d_valid = False
                else:
                    tgt_3d = torch.tensor(tgt_3d).float()
                
                if eye_3d_valid and tgt3d_valid:
                    gaze_vec = tgt_3d - eye_3d
                else:   # for training gaze estimator: just keep the ones with valid 3d gaze vector
                    gaze_vec = torch.zeros(3)
                
                cam_params_this = None
                if cam_params is not None:
                    cam_params_this = cam_params
                else:
                    if int(os.path.splitext(filename)[0])>calib_ranges[-1][1]:
                        pdb.set_trace()
                        raise NotImplementedError
                    for this_idx, ranges in enumerate(calib_ranges):
                        if int(os.path.splitext(filename)[0])<=ranges[1]:
                            cam_params_this = all_calibs[this_idx]
                            break
                # treat each pair of camera view as one sample of input
                # note: Here each image from a camera is paired with a image from every other camera, we only treat the first image and the main view
                for idx_1 in range(0, len(all_cams)):
                    cam_1 = all_cams[idx_1]
                    if len(annt_copy[cam_1]['head'])==0:  # no head annotated
                        continue
                    #if annt_copy[cam_1]['visibility'].lower()!='occlusion':
                        # must be occluded (self-occlusion)
                        #continue
                    for idx_2 in range(0, len(all_cams)):
                        if idx_2==idx_1:
                            continue
                        cam_2 = all_cams[idx_2]
                        self.input_list.append((scene, subj_folder, cam_1, cam_2, filename))
                        self.cam_params_list.append(cam_params_this)
                        self.img_index.append(img_index)
                        self.gaze_3d_list.append(gaze_vec)
                    img_index += 1
                        
        self.vis_mapping = {'false':0, 'true':1, 'occlusion':2}
        self.img_out_size = img_out_size
        self.head_out_size = head_out_size
        self.hm_size = hm_size
        self.test = test     
        self.no_aug = no_aug   # if set to True then no data augmentation will be performed   
        
        self.head_transform = self.get_transform((self.head_out_size[1], self.head_out_size[0]))
        self.img_transform = self.get_transform((self.img_out_size[1], self.img_out_size[0]))
        #self.head_transform, self.img_transform = self.get_transform(), self.get_transform()
        
        def skew_op(x):
            res = np.zeros((3, 3), dtype=x.dtype)
            # 0, -z, y
            res[0, 1] = -x[2, 0]
            res[0, 2] =  x[1, 0]
            # z, 0, -x
            res[1, 0] =  x[2, 0]
            res[1, 2] = -x[0, 0]
            # -y, x, 0
            res[2, 0] = -x[1, 0]
            res[2, 1] =  x[0, 0]
            return res 
        # multi-view geometry
        self.skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        #self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            #R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_op = lambda K_1, R_1, T_1, K_0, R_0, T_0: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: self.fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3:], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3:] )
        
        self.annt_errors = set()
        
    def __getitem__(self, index):
        scene, subj_folder, cam_1, cam_2, filename = self.input_list[index]
        main_view_index = self.img_index[index]
        
        subj_path = os.path.join(self.data_dir, scene, subj_folder, "Images")
        img_path_1, img_path_2 = os.path.join(subj_path, cam_1, filename), os.path.join(subj_path, cam_2, filename)
        annt = self.annotations_dct[(scene, subj_folder, filename)]
        gaze_coord_1, gaze_coord_2 = annt[cam_1]['coordinate'], annt[cam_2]['coordinate']
        head_box_1, head_box_2 = annt[cam_1]['head'], annt[cam_2]['head']
        
        head_valid_1 = False if len(head_box_1)==0 else True
        head_valid_2 = False if len(head_box_2)==0 else True
        #head_valid_2 = False
        #head_box_2 = []
        
        facevis_1  = annt[cam_1]['Face_vis'] if head_valid_1 else -2
        facevis_2  = annt[cam_2]['Face_vis'] if head_valid_2 else -2
            
        imgname = os.path.splitext(filename)[0]
        depth_path_1, depth_path_2 = os.path.join(self.depth_dir, scene, subj_folder, cam_1, f"{imgname}.npy"), os.path.join(self.depth_dir, scene, subj_folder, cam_2, f"{imgname}.npy")
        depth_v1, depth_v2 = np.load(depth_path_1), np.load(depth_path_2)
        
                 
        visib_name_1, visib_name_2 = annt[cam_1]['visibility'].lower(), annt[cam_2]['visibility'].lower()
        visib_1, visib_2 = self.vis_mapping[visib_name_1], self.vis_mapping[visib_name_2]

        #self.body_transform = self.get_transform((self.body_img_shape[1], self.body_img_shape[0]))
        img_1, img_2 = cv2.imread(img_path_1), cv2.imread(img_path_2)
        img_1, img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        height, width = img_1.shape[:2]
        
        head_box_list = [head_box_1, head_box_2]
        for idx, head_box in enumerate(head_box_list):
            head_valid = head_valid_1 if idx==0 else head_valid_2
            if head_valid:
                head_x_min, head_y_min, head_width, head_height = head_box
                head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            else:
                head_x_min, head_y_min, head_x_max, head_y_max = -1, -1, -1, -1
            head_box = np.array([ head_x_min, head_y_min,head_x_max, head_y_max])
            head_box_list[idx] = head_box
        head_box_1, head_box_2 = head_box_list
        
        if len(gaze_coord_1)==0:
            gaze_x1, gaze_y1 = -1, -1
            gaze_valid_1 = False
        else:
            gaze_valid_1 = True
            gaze_x1, gaze_y1 = gaze_coord_1[0], gaze_coord_1[1]
        gaze_coord_1 = np.array([gaze_x1, gaze_y1])
        
        if len(gaze_coord_2)==0:
            #assert visib_2!=1, img_path_2
            gaze_x2, gaze_y2 = -1, -1
            gaze_valid_2 = False
        else:
            #assert visib_2!=0, img_path_2
            gaze_valid_2 = True
            gaze_x2, gaze_y2 = gaze_coord_2[0], gaze_coord_2[1]

        gaze_coord_2 = np.array([gaze_x2, gaze_y2])        
        eye_loc_v1, eye_loc_v2 = self.get_eye_keypoint(annt[cam_1], head_valid_1), self.get_eye_keypoint(annt[cam_2], head_valid_2)
        
        cam_params = copy.deepcopy(self.cam_params_list[index])
        RT_1, RT_2 = cam_params[cam_1]['RT'], cam_params[cam_2]['RT']
        intri_1, intri_2 = cam_params[cam_1]['K'], cam_params[cam_2]['K']    
    
        
        if (not self.test) and (not self.no_aug):
            img_1, depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1 = data_augmentation_albument(img_1, depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1, width, height)
            img_2, depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2 = data_augmentation_albument(img_2, depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2, width, height)
        
        height_1, width_1 = img_1.shape[:2]
        height_2, width_2 = img_2.shape[:2] 
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1]
        if gaze_x1>=0 and gaze_y1>=0:
            gaze_coord_1 = torch.tensor([gaze_x1 / width_1, gaze_y1 / height_1]).float()
        if gaze_x2>=0 and gaze_y2>=0:
            gaze_coord_2 = torch.tensor([gaze_x2 / width_2, gaze_y2 / height_2]).float()
        
        ##### For sharingan
        #head_square_1, head_square_2 = square_bbox(torch.tensor(head_box_1).float().unsqueeze(0)), square_bbox(torch.tensor(head_box_2).float().unsqueeze(0))
        head_square_1, head_square_2 = torch.tensor(head_box_1).float().unsqueeze(0), torch.tensor(head_box_2).float().unsqueeze(0)
        # Normalize Head Bboxes and clip to [0, 1]
        head_square_1 /= torch.tensor([width_1, height_1, width_1, height_1], dtype=torch.float)
        head_square_1 = torch.clamp(head_square_1, min=0.0, max=1.0)
        head_square_2 /= torch.tensor([width_2, height_2, width_2, height_2], dtype=torch.float)
        head_square_2 = torch.clamp(head_square_2, min=0.0, max=1.0)
        head_square = torch.stack((head_square_1, head_square_2))
        #####
        
        head_img_1, head_mask_scene_1 = self.process_head(img_1, head_box_1, head_valid_1, width_1, height_1)
        head_img_2, head_mask_scene_2 = self.process_head(img_2, head_box_2, head_valid_2, width_2, height_2)
        head_coords_1 = torch.tensor([head_box_1[0]/width_1, head_box_1[1]/height_1, head_box_1[2]/width_1, head_box_1[3]/height_1]).float()
        head_coords_2 = torch.tensor([head_box_2[0]/width_2, head_box_2[1]/height_2, head_box_2[2]/width_2, head_box_2[3]/height_2]).float()
        
        if eye_loc_v1[0]!=-1.0:
            eye_loc_head_v1 = torch.tensor([(eye_loc_v1[0]-head_box_1[0])/(head_box_1[2]-head_box_1[0]), (eye_loc_v1[1]-head_box_1[1])/(head_box_1[3]-head_box_1[1])]).float()
            eye_loc_v1 = np.array([eye_loc_v1[0]/width_1, eye_loc_v1[1]/height_1])
        if eye_loc_v2[0]!=-1.0:
            eye_loc_head_v2 = torch.tensor([(eye_loc_v2[0]-head_box_2[0])/(head_box_2[2]-head_box_2[0]), (eye_loc_v2[1]-head_box_2[1])/(head_box_2[3]-head_box_2[1])]).float()
            eye_loc_v2 = np.array([eye_loc_v2[0]/width_2, eye_loc_v2[1]/height_2])
        eye_loc_v1, eye_loc_v2 = torch.tensor(eye_loc_v1).float(), torch.tensor(eye_loc_v2).float()    
        
    
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1] 
        gaze_heatmap_1, gaze_heatmap_2 = torch.zeros(self.hm_size[1], self.hm_size[0]), torch.zeros(self.hm_size[1], self.hm_size[0])
        if gaze_valid_1 and visib_1!=0:
            gaze_heatmap_1 = draw_labelmap(gaze_heatmap_1, [gaze_x1 * self.hm_size[0], gaze_y1 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        if gaze_valid_2 and visib_2!=0:
            gaze_heatmap_2 = draw_labelmap(gaze_heatmap_2, [gaze_x2 * self.hm_size[0], gaze_y2 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        
        
        img_1, img_2 = self.img_transform(image=img_1)["image"], self.img_transform(image=img_2)['image']
        if head_valid_1:
            head_img_1 = self.head_transform(image=head_img_1)["image"]
        if head_valid_2:
            head_img_2 = self.head_transform(image=head_img_2)["image"]
        
        img_1, img_2 = torch.tensor(img_1).float(), torch.tensor(img_2).float()
        head_img_1, head_img_2 = torch.tensor(head_img_1).float(), torch.tensor(head_img_2).float()   
        img_1, img_2, head_img_1, head_img_2 = img_1.permute(2, 0, 1), img_2.permute(2, 0, 1), head_img_1.permute(2, 0, 1), head_img_2.permute(2, 0, 1)
        gaze_coord_1, gaze_coord_2 = torch.tensor([gaze_x1, gaze_y1]).float(), torch.tensor([gaze_x2, gaze_y2]).float()
        visib_1, visib_2 = torch.tensor(visib_1).long(), torch.tensor(visib_2).long()
        head_valid_1, head_valid_2 = torch.tensor(head_valid_1).long(), torch.tensor(head_valid_2).long()
        
        depth_v1, depth_v2 = torch.tensor(depth_v1).float(), torch.tensor(depth_v2).float()
        if depth_v1.size(0)!=self.img_out_size[1] or depth_v1.size(1)!=self.img_out_size[0]:
            depth_v1 = F.interpolate(depth_v1.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if depth_v2.size(0)!=self.img_out_size[1] or depth_v2.size(1)!=self.img_out_size[0]:
            depth_v2 = F.interpolate(depth_v2.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        
        # get normalized depth values  
        # depth_max, depth_min = depth_v1.max(), depth_v1.min()
        # depth_v1_norm = (depth_v1 - depth_min) / (depth_max - depth_min + eps)
        # depth_v1_norm = 1 - depth_v1_norm  # reverse the depth image
        # depth_max, depth_min = depth_v2.max(), depth_v2.min()
        # depth_v2_norm = (depth_v2 - depth_min) / (depth_max - depth_min + eps)
        # depth_v2_norm = 1 - depth_v2_norm  # reverse the depth image
        
        
        # process camera parameters
        ratio_w, ratio_h = self.img_out_size[0] / width_1, self.img_out_size[1]/height_1
        intri_1[0,:] = intri_1[0,:] * ratio_w
        intri_1[1,:] = intri_1[1,:] * ratio_h
        ratio_w, ratio_h = self.img_out_size[0] / width_2, self.img_out_size[1]/height_2
        intri_2[0,:] = intri_2[0,:] * ratio_w
        intri_2[1,:] = intri_2[1,:] * ratio_h
    
        fund_mat = torch.tensor(self.fundamental_RT_op(intri_1,RT_1, intri_2, RT_2)).float()
        intri_1, intri_2 = torch.from_numpy(intri_1).float(), torch.from_numpy(intri_2).float()
        RT_1, RT_2 = torch.from_numpy(RT_1).float(), torch.from_numpy(RT_2).float()
        R1, R2 = RT_1[:, :3], RT_2[:, :3]
        
        gaze_3d = self.gaze_3d_list[index]
        if torch.all(gaze_3d==0):
            gtvec_3d_1, gtvec_3d_2 = torch.zeros(3).float(), torch.zeros(3).float()
        else:
            gaze_3d_1 = R1 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_1 = torch.linalg.norm(gaze_3d_1)
            gtvec_3d_1 = gaze_3d_1 / (gaze_3d_norm_1 + eps)
            gaze_3d_2 = R2 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_2 = torch.linalg.norm(gaze_3d_2)
            gtvec_3d_2 = gaze_3d_2 / (gaze_3d_norm_2 + eps)
        
        img, head_img, head_mask_scene, depth = torch.stack((img_1, img_2)), torch.stack((head_img_1, head_img_2)), torch.stack((head_mask_scene_1, head_mask_scene_2)), torch.stack((depth_v1, depth_v2))
        gaze_heatmap, visib, eye_loc, gaze_coord, head_valid = torch.stack((gaze_heatmap_1, gaze_heatmap_2)), torch.stack((visib_1, visib_2)), torch.stack((eye_loc_v1, eye_loc_v2)), torch.stack((gaze_coord_1, gaze_coord_2)), torch.stack((head_valid_1, head_valid_2))
        gtvec_3d = torch.stack((gtvec_3d_1, gtvec_3d_2))
        head_coords = torch.stack((head_coords_1, head_coords_2))
        intri, RT = torch.stack((intri_1, intri_2)), torch.stack((RT_1, RT_2))
        path_info = os.path.join(scene, subj_folder, cam_1, cam_2, filename)
        facevis = torch.tensor([facevis_1, facevis_2]).int()
        
        data_dict = {
            "data": (img, head_img, head_mask_scene, depth, gaze_heatmap, visib, gaze_coord, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT),
            "fund_mat": fund_mat,
            "path": path_info,
            'main_id': main_view_index,
            'face_vis': facevis
        }
        
        return data_dict
         
    def __len__(self):
        return len(self.input_list)
    
    def get_transform(self, out_shape):
        transform_list = []
        transform_list.append(A.Resize(out_shape[0], out_shape[1], interpolation=cv2.INTER_AREA))
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return A.Compose(transform_list)

    def process_head(self, img, head_box, head_valid, width, height):
        head_x_min, head_y_min, head_x_max, head_y_max = head_box
        
        head_img = np.zeros((self.head_out_size[1], self.head_out_size[0], 3))
        head_mask_scene = torch.zeros((1, self.img_out_size[1], self.img_out_size[0]))  # head w.r.t scene image
        
        if head_valid:
            head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * self.img_out_size[0]), round(head_y_min / height * self.img_out_size[1]), round(head_x_max / width * self.img_out_size[0]), round(head_y_max / height * self.img_out_size[1])
            head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
            #head_img = img.crop((int(head_x_min), int(head_y_min), int(head_x_max), int(head_y_max)))
            head_img = img[int(head_y_min):int(head_y_max), int(head_x_min):int(head_x_max), :]
            
        return head_img, head_mask_scene
    
    
    def get_eye_keypoint(self, annt, head_valid, conf_thres=1.2):
        eye_loc = annt['eye']
        if len(eye_loc)==0 or not head_valid:
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
        if eye_loc[0]==-1 and head_valid:
            # assign center of head box as eye location
            head_x_min, head_y_min, head_width, head_height = annt['head']
            head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            eye_loc = np.array([(head_x_min + head_x_max)/2, (head_y_min + head_y_max)/2])
            
        return eye_loc
    

    def data_augmentation_albument(self, img, depth_img, head_box, gaze_coord, eye_loc, intri, width, height):
        
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


    

class Gaze_Dataset_Multiview_RandomSample(Dataset):
    def __init__(self, base_dir, img_out_size, head_out_size, hm_size, eval_scene='Lab', test=False, no_aug=False, adapt=False, adapt_idx=0):
        # note img_out_size and hm_size are int as they are squares, body_img_size is tuple as body img is rectangle
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'Data')
        annt_dir = os.path.join(self.base_dir, 'Annotations')

        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'Data')
        annt_dir = os.path.join(self.base_dir, 'Annotations')
        self.depth_dir = os.path.join(base_dir, 'Depth_Metric3d')
        
        self.test = test
        self.no_aug = no_aug
        if test:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        self.all_cams = [f'Cam{i}' for i in range(1, 7)]       
        self.eval_scene = eval_scene    
        all_subjs = sorted(glob(os.path.join(self.data_dir, '*', '*')))   # data/scene/subj
        test_subjs = sorted(glob(os.path.join(self.data_dir, eval_scene, '*')))
        train_subjs = [each_subj for each_subj in all_subjs if each_subj not in test_subjs ]
        
        #subj_folders = [os.path.join(self.data_dir, name) for name in subject_names]
        
        if not adapt:
            if test:
                subj_folders = test_subjs   
            
            else:
                subj_folders = train_subjs
            print(f"{self.mode} on {eval_scene},  {len(subj_folders)} subjects")
            
        else:
            print(f"Evaluate on {test_subjs[adapt_idx]}")
            if test:
                subj_folders =[test_subjs[adapt_idx]]
            else:
                test_subjs.remove(test_subjs[adapt_idx]) 
                subj_folders = test_subjs
            
            
        self.cam_params = {}
        self.annotations_dct = {}
        self.input_list = []
        self.annt_list = []
        self.cam_params_list = []
        self.gaze_3d_list = []
        self.reproj_err_thres = 35.0
        self.img_index = []
        img_index = 0
         
        for folder_path in subj_folders:
            scene, subj_folder = folder_path.split('/')[-2], folder_path.split('/')[-1]
            cam_params = None
            if os.path.exists(os.path.join(folder_path, "Calibration", 'extri.yml')):
                extri_path = os.path.join(folder_path, 'Calibration', 'extri.yml')
                intri_path = os.path.join(folder_path, 'Calibration', 'intri.yml')
                cam_params = read_camera(intri_path, extri_path)
            else:
                # in this case, one subject has multiple calibrations (due to camera movement, etc)
                all_calibs = []
                all_calib_names = sorted([foldername for foldername in os.listdir(os.path.join(folder_path)) if foldername.startswith('Calib')])
                calib_ranges = []
                for calib_folder in all_calib_names:
                    ranges = calib_folder.split('_')[1].split('to')
                    calib_ranges.append((int(ranges[0]), int(ranges[1])))
                    extri_path = os.path.join(folder_path, calib_folder, 'extri.yml')
                    intri_path = os.path.join(folder_path, calib_folder, 'intri.yml')
                    all_calibs.append(read_camera(intri_path, extri_path))
                
            annt_json = os.path.join(annt_dir, scene, f'{subj_folder}.json')
            with open(annt_json, 'r') as file:
                annt_subj = json.load(file)
            annt3d_path = os.path.join(annt_dir, scene, "triangulate_3d", f"{subj_folder}.json")
            annt_3d = read_json(annt3d_path)
            
            all_cams = sorted([key for key in list(annt_subj[0].keys()) if key.startswith('Cam')])
            for idx, annt_img in enumerate(annt_subj):
                filename = annt_img['filename']
                filename_noext = os.path.splitext(filename)[0]
                annt_copy = copy.deepcopy(annt_subj[idx])
                annt_copy.pop('filename')
                self.annotations_dct[(scene, subj_folder, filename)] = annt_copy
                
                # append 3d locations of eye and target for gaze estimation
                assert annt_3d[idx]['filename'] == filename_noext, print(annt_3d[idx]['filename'], filename_noext, idx)
                eye_3d, tgt_3d = annt_3d[idx]['eye'], annt_3d[idx]['target']
                eye_3d_valid, tgt3d_valid = True, True
                if len(eye_3d)==0 or annt_3d[idx]['eye_err'] > self.reproj_err_thres:
                    eye_3d = torch.zeros(3)
                    eye_3d_valid = False
                else:
                    eye_3d = torch.tensor(eye_3d).float()
                
                if len(tgt_3d)==0 or annt_3d[idx]['target_err'] > self.reproj_err_thres:
                    tgt_3d = torch.zeros(3)
                    tgt3d_valid = False
                else:
                    tgt_3d = torch.tensor(tgt_3d).float()
                
                if eye_3d_valid and tgt3d_valid:
                    gaze_vec = tgt_3d - eye_3d
                else:   # for training gaze estimator: just keep the ones with valid 3d gaze vector
                    gaze_vec = torch.zeros(3)
                
                # camera parameters
                for idx in range(0, len(all_cams)):
                    cam = all_cams[idx]        
                    if len(annt_copy[cam]['head'])==0:  # no head annotated
                        continue
                    self.input_list.append((scene, subj_folder, cam, filename))
                    self.gaze_3d_list.append(gaze_vec)
                    self.img_index.append(img_index)
                    img_index += 1
                    if cam_params is not None:
                        self.cam_params_list.append(cam_params)  # note that this dict may be shared across different inputs, so deep copy in loading!    
                    else:
                        if int(os.path.splitext(filename)[0])>calib_ranges[-1][1]:
                            pdb.set_trace()
                            raise NotImplementedError
                        for this_idx, ranges in enumerate(calib_ranges):
                            if int(os.path.splitext(filename)[0])<=ranges[1]:
                                self.cam_params_list.append(all_calibs[this_idx])
                                break
                    
        
        self.annt_dir = annt_dir
        self.vis_mapping = {'false':0, 'true':1, 'occlusion':2}
        self.img_out_size = img_out_size
        self.head_out_size = head_out_size
        self.head_transform = self.get_transform((self.head_out_size[1], self.head_out_size[0]))
        self.img_transform = self.get_transform((self.img_out_size[1], self.img_out_size[0]))
        self.hm_size = hm_size
        self.test = test
        
        def skew_op(x):
            res = np.zeros((3, 3), dtype=x.dtype)
            # 0, -z, y
            res[0, 1] = -x[2, 0]
            res[0, 2] =  x[1, 0]
            # z, 0, -x
            res[1, 0] =  x[2, 0]
            res[1, 2] = -x[0, 0]
            # -y, x, 0
            res[2, 0] = -x[1, 0]
            res[2, 1] =  x[0, 0]
            return res 
        # multi-view geometry
        self.skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        #self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            #R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_op = lambda K_1, R_1, T_1, K_0, R_0, T_0: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: self.fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3:], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3:] )
        
        self.annt_errors = set()
        
    def __getitem__(self, index):
        scene, subj_folder, cam, filename = self.input_list[index]
        other_cams = self.all_cams.copy()
        other_cams.remove(cam)
        cam_1, cam_2 = cam, np.random.choice(other_cams) 
        
        main_view_index = self.img_index[index]
        
        subj_path = os.path.join(self.data_dir, scene, subj_folder, "Images")
        img_path_1, img_path_2 = os.path.join(subj_path, cam_1, filename), os.path.join(subj_path, cam_2, filename)
        annt = self.annotations_dct[(scene, subj_folder, filename)]
        gaze_coord_1, gaze_coord_2 = annt[cam_1]['coordinate'], annt[cam_2]['coordinate']
        head_box_1, head_box_2 = annt[cam_1]['head'], annt[cam_2]['head']

        head_valid_1 = False if len(head_box_1)==0 else True
        head_valid_2 = False if len(head_box_2)==0 else True
        imgname = os.path.splitext(filename)[0]
        depth_path_1, depth_path_2 = os.path.join(self.depth_dir, scene, subj_folder, cam_1, f"{imgname}.npy"), os.path.join(self.depth_dir, scene, subj_folder, cam_2, f"{imgname}.npy")
        depth_v1, depth_v2 = np.load(depth_path_1), np.load(depth_path_2)
                 
        visib_name_1, visib_name_2 = annt[cam_1]['visibility'].lower(), annt[cam_2]['visibility'].lower()
        visib_1, visib_2 = self.vis_mapping[visib_name_1], self.vis_mapping[visib_name_2]

        #self.body_transform = self.get_transform((self.body_img_shape[1], self.body_img_shape[0]))
        img_1, img_2 = cv2.imread(img_path_1), cv2.imread(img_path_2)
        img_1, img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        height, width = img_1.shape[:2]    
        
        head_box_list = [head_box_1, head_box_2]
        for idx, head_box in enumerate(head_box_list):
            head_valid = head_valid_1 if idx==0 else head_valid_2
            if head_valid:
                head_x_min, head_y_min, head_width, head_height = head_box
                head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            else:
                head_x_min, head_y_min, head_x_max, head_y_max = -1, -1, -1, -1
            head_box = np.array([ head_x_min, head_y_min,head_x_max, head_y_max])
            head_box_list[idx] = head_box
        head_box_1, head_box_2 = head_box_list
        
        if len(gaze_coord_1)==0:
            gaze_x1, gaze_y1 = -1, -1
            gaze_valid_1 = False
        else:
            gaze_valid_1 = True
            gaze_x1, gaze_y1 = gaze_coord_1[0], gaze_coord_1[1]
        gaze_coord_1 = np.array([gaze_x1, gaze_y1])
        
        if len(gaze_coord_2)==0:
            #assert visib_2!=1, img_path_2
            gaze_x2, gaze_y2 = -1, -1
            gaze_valid_2 = False
        else:
            #assert visib_2!=0, img_path_2
            gaze_valid_2 = True
            gaze_x2, gaze_y2 = gaze_coord_2[0], gaze_coord_2[1]

        gaze_coord_2 = np.array([gaze_x2, gaze_y2])        
        eye_loc_v1, eye_loc_v2 = self.get_eye_keypoint(annt[cam_1]), self.get_eye_keypoint(annt[cam_2])
        
        cam_params = copy.deepcopy(self.cam_params_list[index])
        RT_1, RT_2 = cam_params[cam_1]['RT'], cam_params[cam_2]['RT']
        intri_1, intri_2 = cam_params[cam_1]['K'], cam_params[cam_2]['K']

        if (not self.test) and (not self.no_aug):
            img_1, depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1 = self.data_augmentation(img_1, depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1, width, height)
            img_2, depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2 = self.data_augmentation(img_2, depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2, width, height)
        
        height_1, width_1 = img_1.shape[:2]
        height_2, width_2 = img_2.shape[:2] 
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1]
        if gaze_x1>=0 and gaze_y1>=0:
            gaze_coord_1 = torch.tensor([gaze_x1 / width_1, gaze_y1 / height_1]).float()
        if gaze_x2>=0 and gaze_y2>=0:
            gaze_coord_2 = torch.tensor([gaze_x2 / width_2, gaze_y2 / height_2]).float()
        
        
         
        head_img_1, head_mask_scene_1 = self.process_head(img_1, head_box_1, head_valid_1, width_1, height_1)
        head_img_2, head_mask_scene_2 = self.process_head(img_2, head_box_2, head_valid_2, width_2, height_2)
        head_coords_1 = torch.tensor([head_box_1[0]/width_1, head_box_1[1]/height_1, head_box_1[2]/width_1, head_box_1[3]/height_1]).float()
        head_coords_2 = torch.tensor([head_box_2[0]/width_2, head_box_2[1]/height_2, head_box_2[2]/width_2, head_box_2[3]/height_2]).float()
        
        
        if eye_loc_v1[0]!=-1.0:
            eye_loc_v1 = np.array([eye_loc_v1[0]/width_1, eye_loc_v1[1]/height_1])
            
        if eye_loc_v2[0]!=-1.0:
            eye_loc_v2 = np.array([eye_loc_v2[0]/width_2, eye_loc_v2[1]/height_2])
            
            
        eye_loc_v1, eye_loc_v2 = torch.tensor(eye_loc_v1).float(), torch.tensor(eye_loc_v2).float()
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1] 
        gaze_heatmap_1, gaze_heatmap_2 = torch.zeros(self.hm_size[1], self.hm_size[0]), torch.zeros(self.hm_size[1], self.hm_size[0])
        if gaze_valid_1 and visib_1!=0:
            gaze_heatmap_1 = draw_labelmap(gaze_heatmap_1, [gaze_x1 * self.hm_size[0], gaze_y1 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        if gaze_valid_2 and visib_2!=0:
            gaze_heatmap_2 = draw_labelmap(gaze_heatmap_2, [gaze_x2 * self.hm_size[0], gaze_y2 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        
        img_1, img_2 = self.img_transform(image=img_1)["image"], self.img_transform(image=img_2)['image']
        if head_valid_1:
            head_img_1 = self.head_transform(image=head_img_1)["image"]
        if head_valid_2:
            head_img_2 = self.head_transform(image=head_img_2)["image"]
        
        img_1, img_2 = torch.tensor(img_1).float(), torch.tensor(img_2).float()
        head_img_1, head_img_2 = torch.tensor(head_img_1).float(), torch.tensor(head_img_2).float()   
        img_1, img_2, head_img_1, head_img_2 = img_1.permute(2, 0, 1), img_2.permute(2, 0, 1), head_img_1.permute(2, 0, 1), head_img_2.permute(2, 0, 1)
        gaze_coord_1, gaze_coord_2 = torch.tensor([gaze_x1, gaze_y1]).float(), torch.tensor([gaze_x2, gaze_y2]).float()
        visib_1, visib_2 = torch.tensor(visib_1).long(), torch.tensor(visib_2).long()
        head_valid_1, head_valid_2 = torch.tensor(head_valid_1).long(), torch.tensor(head_valid_2).long()
        
        
        depth_v1, depth_v2 = torch.tensor(depth_v1).float(), torch.tensor(depth_v2).float()
        if depth_v1.size(0)!=self.img_out_size[1] or depth_v1.size(1)!=self.img_out_size[0]:
            depth_v1 = F.interpolate(depth_v1.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if depth_v2.size(0)!=self.img_out_size[1] or depth_v2.size(1)!=self.img_out_size[0]:
            depth_v2 = F.interpolate(depth_v2.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        
        
        # get normalized depth values  
        # depth_max, depth_min = depth_v1.max(), depth_v1.min()
        # depth_v1_norm = (depth_v1 - depth_min) / (depth_max - depth_min + eps)
        # depth_v1_norm = 1 - depth_v1_norm  # reverse the depth image
        # depth_max, depth_min = depth_v2.max(), depth_v2.min()
        # depth_v2_norm = (depth_v2 - depth_min) / (depth_max - depth_min + eps)
        # depth_v2_norm = 1 - depth_v2_norm  # reverse the depth image
        
        # change original intrinsic matrix due to resizing
        ratio_w, ratio_h = self.img_out_size[0] / width_1, self.img_out_size[1]/height_1
        intri_1[0,:] = intri_1[0,:] * ratio_w
        intri_1[1,:] = intri_1[1,:] * ratio_h
        ratio_w, ratio_h = self.img_out_size[0] / width_2, self.img_out_size[1]/height_2
        intri_2[0,:] = intri_2[0,:] * ratio_w
        intri_2[1,:] = intri_2[1,:] * ratio_h
    
        fund_mat = torch.tensor(self.fundamental_RT_op(intri_1,RT_1, intri_2, RT_2)).float()
        intri_1, intri_2 = torch.from_numpy(intri_1).float(), torch.from_numpy(intri_2).float()
        RT_1, RT_2 = torch.from_numpy(RT_1).float(), torch.from_numpy(RT_2).float()
        R1, R2 = RT_1[:, :3], RT_2[:, :3]
        
        
        gaze_3d = self.gaze_3d_list[index]
        if torch.all(gaze_3d==0):
            gtvec_3d_1, gtvec_3d_2 = torch.zeros(3).float(), torch.zeros(3).float()
        else:
            gaze_3d_1 = R1 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_1 = torch.linalg.norm(gaze_3d_1)
            gtvec_3d_1 = gaze_3d_1 / (gaze_3d_norm_1 + eps)
            gaze_3d_2 = R2 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_2 = torch.linalg.norm(gaze_3d_2)
            gtvec_3d_2 = gaze_3d_2 / (gaze_3d_norm_2 + eps)
        
        img, head_img, head_mask_scene, depth = torch.stack((img_1, img_2)), torch.stack((head_img_1, head_img_2)), torch.stack((head_mask_scene_1, head_mask_scene_2)), torch.stack((depth_v1, depth_v2))
        gaze_heatmap, visib, eye_loc, gaze_coord, head_valid = torch.stack((gaze_heatmap_1, gaze_heatmap_2)), torch.stack((visib_1, visib_2)), torch.stack((eye_loc_v1, eye_loc_v2)), torch.stack((gaze_coord_1, gaze_coord_2)), torch.stack((head_valid_1, head_valid_2))
        gtvec_3d = torch.stack((gtvec_3d_1, gtvec_3d_2))
        head_coords = torch.stack((head_coords_1, head_coords_2))
        intri, RT = torch.stack((intri_1, intri_2)), torch.stack((RT_1, RT_2))
        path_info = os.path.join(scene, subj_folder, cam_1, cam_2, filename)
        
        
        data_dict = {
            "data": (img, head_img, head_mask_scene, depth, gaze_heatmap, visib, gaze_coord, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT),
            "fund_mat": fund_mat,
            "path": path_info,
            'main_id': main_view_index
        }
        
        return data_dict
         
    def __len__(self):
        return len(self.input_list)
    
    def get_transform(self, out_shape):
        transform_list = []
        transform_list.append(A.Resize(out_shape[0], out_shape[1], interpolation=cv2.INTER_AREA))
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return A.Compose(transform_list)
    
    
    def process_head(self, img, head_box, head_valid, width, height):
        head_x_min, head_y_min, head_x_max, head_y_max = head_box
        
        head_img = np.zeros((self.head_out_size[1], self.head_out_size[0], 3))
        head_mask_scene = torch.zeros(1, self.img_out_size[1], self.img_out_size[0])  # head w.r.t scene image
        
        if head_valid:
            head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * self.img_out_size[0]), round(head_y_min / height * self.img_out_size[1]), round(head_x_max / width * self.img_out_size[0]), round(head_y_max / height * self.img_out_size[1])
            head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
            head_img = img[int(head_y_min):int(head_y_max), int(head_x_min):int(head_x_max), :]
            
        return head_img, head_mask_scene
    
    
    def get_eye_keypoint(self, annt, conf_thres=1.2):
        eye_loc = annt['eye']
        if len(annt['head'])==0:
            # eye will be invalid if head is not annotated
            eye_loc = np.array([-1.0,-1.0])
            return eye_loc
        
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
    
    def data_augmentation(self, img, depth_img, head_box, gaze_coord, eye_loc, intri, width, height):
        
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
        
        return img, depth_img, head_box, gaze_coord, eye_loc, intri
    

class Gaze_Dataset_Multiview_CrossView(Dataset):
    def __init__(self, base_dir, img_out_size, head_out_size, hm_size, eval_scene='Lab', test=False, no_aug=False):
        # note img_out_size and hm_size are int as they are squares, body_img_size is tuple as body img is rectangle
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'Data')
        annt_dir = os.path.join(self.base_dir, 'Annotations_facevis')
        self.depth_dir = os.path.join(base_dir, 'Depth_Metric3d')
        self.abs_depth_dir = os.path.join(base_dir, "Reconstruct", "scaleshift_metric3d")
        self.test = test 
        if test:
            self.mode = 'test'
        else:
            self.mode = 'train'
        
        self.eval_scene = eval_scene    
        all_subjs = sorted(glob(os.path.join(self.data_dir, '*', '*')))   # data/scene/subj
        test_subjs = sorted(glob(os.path.join(self.data_dir, eval_scene, '*')))
        train_subjs = [each_subj for each_subj in all_subjs if each_subj not in test_subjs]
            
        if test:
            subj_folders = test_subjs
            #subj_folders = all_subjs
            print(f"{self.mode} on {eval_scene},  {len(test_subjs)} subjects")
        else:
            subj_folders = train_subjs
            print(f"Train on {len(train_subjs)} subjects.")
        
        self.cam_params = {}
        self.annotations_dct = {}
        self.input_list = []
        self.annt_list = []
        self.cam_params_list = []
        self.img_index = []  # indices of the image of the main view, for evaluation
        img_index = 0
        self.gaze_3d_list = []
        self.reproj_err_thres = 35.0
        
        for folder_path in subj_folders:
            scene, subj_folder = folder_path.split('/')[-2], folder_path.split('/')[-1]
            
            cam_params = None
            if os.path.exists(os.path.join(folder_path, "Calibration", 'extri.yml')):
                extri_path = os.path.join(folder_path, 'Calibration', 'extri.yml')
                intri_path = os.path.join(folder_path, 'Calibration', 'intri.yml')
                cam_params = read_camera(intri_path, extri_path)
            else:
                # in this case, one subject has multiple calibrations (due to camera movement, etc)
                all_calibs = []
                all_calib_names = sorted([foldername for foldername in os.listdir(os.path.join(folder_path)) if foldername.startswith('Calib')])
                calib_ranges = []
                for calib_folder in all_calib_names:
                    ranges = calib_folder.split('_')[1].split('to')
                    calib_ranges.append((int(ranges[0]), int(ranges[1])))
                    extri_path = os.path.join(folder_path, calib_folder, 'extri.yml')
                    intri_path = os.path.join(folder_path, calib_folder, 'intri.yml')
                    all_calibs.append(read_camera(intri_path, extri_path))
                
            annt_json = os.path.join(annt_dir, scene, f'{subj_folder}.json')
            with open(annt_json, 'r') as file:
                annt_subj = json.load(file)
            
            # add for triangulated 3d points
            annt3d_path = os.path.join(annt_dir, scene, "triangulate_3d", f"{subj_folder}.json")
            annt_3d = read_json(annt3d_path)
                
            all_cams = sorted([key for key in list(annt_subj[0].keys()) if key.startswith('Cam')])
            for idx, annt_img in enumerate(annt_subj):
                filename = annt_img['filename']
                #if filename!='0031.JPG':
                #    continue
                
                annt_copy = copy.deepcopy(annt_subj[idx])
                annt_copy.pop('filename')
                self.annotations_dct[(scene, subj_folder, filename)] = annt_copy

                filename_noext = os.path.splitext(filename)[0]
                assert annt_3d[idx]['filename'] == filename_noext, print(annt_3d[idx]['filename'], filename_noext, idx)
                eye_3d, tgt_3d = annt_3d[idx]['eye'], annt_3d[idx]['target']
                eye_3d_valid, tgt3d_valid = True, True
                if len(eye_3d)==0 or annt_3d[idx]['eye_err'] > self.reproj_err_thres:
                    eye_3d = torch.zeros(3)
                    eye_3d_valid = False
                else:
                    eye_3d = torch.tensor(eye_3d).float()
                
                if len(tgt_3d)==0 or annt_3d[idx]['target_err'] > self.reproj_err_thres:
                    tgt_3d = torch.zeros(3)
                    tgt3d_valid = False
                else:
                    tgt_3d = torch.tensor(tgt_3d).float()
                
                if eye_3d_valid and tgt3d_valid:
                    gaze_vec = tgt_3d - eye_3d
                else:   # for training gaze estimator: just keep the ones with valid 3d gaze vector
                    gaze_vec = torch.zeros(3)
                
                cam_params_this = None
                if cam_params is not None:
                    cam_params_this = cam_params
                else:
                    if int(os.path.splitext(filename)[0])>calib_ranges[-1][1]:
                        pdb.set_trace()
                        raise NotImplementedError
                    for this_idx, ranges in enumerate(calib_ranges):
                        if int(os.path.splitext(filename)[0])<=ranges[1]:
                            cam_params_this = all_calibs[this_idx]
                            break
                # treat each pair of camera view as one sample of input
                # note: Here each image from a camera is paired with a image from every other camera, we only treat the first image and the main view
                for idx_1 in range(0, len(all_cams)):
                    cam_1 = all_cams[idx_1]
                    annt_cam1 = annt_img[cam_1]
                    #if self.test:
                    # in evaluation, just include the samples that do not have the head (not applicable to single view models)
                    if len(annt_cam1['head'])>0:
                        continue
                    for idx_2 in range(0, len(all_cams)):
                        if idx_2==idx_1:
                            continue
                        cam_2 = all_cams[idx_2]
                        annt_cam2 = annt_img[cam_2]
                        if len(annt_cam2['head'])==0:   
                            continue
                        self.input_list.append((scene, subj_folder, cam_1, cam_2, filename))
                        self.cam_params_list.append(cam_params_this)
                        self.img_index.append(img_index)
                        self.gaze_3d_list.append(gaze_vec)
                    img_index += 1
                        
        self.vis_mapping = {'false':0, 'true':1, 'occlusion':2}
        self.img_out_size = img_out_size
        self.head_out_size = head_out_size
        self.hm_size = hm_size
        self.test = test     
        self.no_aug = no_aug   # if set to True then no data augmentation will be performed   
        
        self.head_transform = self.get_transform((self.head_out_size[1], self.head_out_size[0]))
        self.img_transform = self.get_transform((self.img_out_size[1], self.img_out_size[0]))
        
        def skew_op(x):
            res = np.zeros((3, 3), dtype=x.dtype)
            # 0, -z, y
            res[0, 1] = -x[2, 0]
            res[0, 2] =  x[1, 0]
            # z, 0, -x
            res[1, 0] =  x[2, 0]
            res[1, 2] = -x[0, 0]
            # -y, x, 0
            res[2, 0] = -x[1, 0]
            res[2, 1] =  x[0, 0]
            return res 
        # multi-view geometry
        self.skew_op = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        #self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: np.linalg.inv(K_0).T @ (
            #R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_op = lambda K_1, R_1, T_1, K_0, R_0, T_0: np.linalg.inv(K_0).T @ (
            R_0 @ R_1.T) @ K_1.T @ skew_op(K_1 @ R_1 @ R_0.T @ (T_0 - R_0 @ R_1.T @ T_1))
        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: self.fundamental_op (K_0, RT_0[:, :3], RT_0[:, 3:], K_1,
                                                                          RT_1[:, :3], RT_1[:, 3:] )
        
        self.annt_errors = set()
        
    def __getitem__(self, index):
        scene, subj_folder, cam_1, cam_2, filename = self.input_list[index]
        main_view_index = self.img_index[index]
        
        subj_path = os.path.join(self.data_dir, scene, subj_folder, "Images")
        img_path_1, img_path_2 = os.path.join(subj_path, cam_1, filename), os.path.join(subj_path, cam_2, filename)
        annt = self.annotations_dct[(scene, subj_folder, filename)]
        gaze_coord_1, gaze_coord_2 = annt[cam_1]['coordinate'], annt[cam_2]['coordinate']
        head_box_1, head_box_2 = annt[cam_1]['head'], annt[cam_2]['head']
        

        head_valid_1 = False if len(head_box_1)==0 else True
        head_valid_2 = False if len(head_box_2)==0 else True
        
        facevis_1  = annt[cam_1]['Face_vis'] if head_valid_1 else -2
        facevis_2  = annt[cam_2]['Face_vis'] if head_valid_2 else -2
            
        imgname = os.path.splitext(filename)[0]
        depth_path_1, depth_path_2 = os.path.join(self.depth_dir, scene, subj_folder, cam_1, f"{imgname}.npy"), os.path.join(self.depth_dir, scene, subj_folder, cam_2, f"{imgname}.npy")
        depth_v1, depth_v2 = np.load(depth_path_1), np.load(depth_path_2)
        
        abs_depth_path = os.path.join(self.abs_depth_dir, scene, subj_folder, f"{imgname}_scales.pkl")
        with open(abs_depth_path, 'rb') as file:
            abs_depth_info = pickle.load(file)
        abs_depth_v1, abs_depth_v2 = abs_depth_info[cam_1]['scaled_depth'], abs_depth_info[cam_2]['scaled_depth']
        
        visib_name_1, visib_name_2 = annt[cam_1]['visibility'].lower(), annt[cam_2]['visibility'].lower()
        visib_1, visib_2 = self.vis_mapping[visib_name_1], self.vis_mapping[visib_name_2]

        #self.body_transform = self.get_transform((self.body_img_shape[1], self.body_img_shape[0]))
        img_1, img_2 = cv2.imread(img_path_1), cv2.imread(img_path_2)
        img_1, img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
        height, width = img_1.shape[:2]
        
        head_box_list = [head_box_1, head_box_2]
        for idx, head_box in enumerate(head_box_list):
            head_valid = head_valid_1 if idx==0 else head_valid_2
            if head_valid:
                head_x_min, head_y_min, head_width, head_height = head_box
                head_x_max, head_y_max = head_x_min + head_width, head_y_min + head_height
            else:
                head_x_min, head_y_min, head_x_max, head_y_max = -1, -1, -1, -1
            head_box = np.array([ head_x_min, head_y_min,head_x_max, head_y_max])
            head_box_list[idx] = head_box
        head_box_1, head_box_2 = head_box_list
        
        if len(gaze_coord_1)==0:
            gaze_x1, gaze_y1 = -1, -1
            gaze_valid_1 = False
        else:
            gaze_valid_1 = True
            gaze_x1, gaze_y1 = gaze_coord_1[0], gaze_coord_1[1]
        gaze_coord_1 = np.array([gaze_x1, gaze_y1])
        
        if len(gaze_coord_2)==0:
            #assert visib_2!=1, img_path_2
            gaze_x2, gaze_y2 = -1, -1
            gaze_valid_2 = False
        else:
            #assert visib_2!=0, img_path_2
            gaze_valid_2 = True
            gaze_x2, gaze_y2 = gaze_coord_2[0], gaze_coord_2[1]

        gaze_coord_2 = np.array([gaze_x2, gaze_y2])        
        eye_loc_v1, eye_loc_v2 = self.get_eye_keypoint(annt[cam_1]), self.get_eye_keypoint(annt[cam_2])
        
        cam_params = copy.deepcopy(self.cam_params_list[index])
        RT_1, RT_2 = cam_params[cam_1]['RT'], cam_params[cam_2]['RT']
        intri_1, intri_2 = cam_params[cam_1]['K'], cam_params[cam_2]['K']    
    
        
        if (not self.test) and (not self.no_aug):
            img_1, depth_v1, abs_depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1 = self.data_augmentation_albument(img_1, depth_v1, abs_depth_v1, head_box_1, gaze_coord_1, eye_loc_v1, intri_1, width, height)
            img_2, depth_v2, abs_depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2 = self.data_augmentation_albument(img_2, depth_v2, abs_depth_v2, head_box_2, gaze_coord_2, eye_loc_v2, intri_2, width, height)
        
        height_1, width_1 = img_1.shape[:2]
        height_2, width_2 = img_2.shape[:2] 
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1]
        if gaze_x1>=0 and gaze_y1>=0:
            gaze_coord_1 = torch.tensor([gaze_x1 / width_1, gaze_y1 / height_1]).float()
        if gaze_x2>=0 and gaze_y2>=0:
            gaze_coord_2 = torch.tensor([gaze_x2 / width_2, gaze_y2 / height_2]).float()
         
        head_img_1, head_mask_scene_1 = self.process_head(img_1, head_box_1, head_valid_1, width_1, height_1)
        head_img_2, head_mask_scene_2 = self.process_head(img_2, head_box_2, head_valid_2, width_2, height_2)
        head_coords_1 = torch.tensor([head_box_1[0]/width_1, head_box_1[1]/height_1, head_box_1[2]/width_1, head_box_1[3]/height_1]).float()
        head_coords_2 = torch.tensor([head_box_2[0]/width_2, head_box_2[1]/height_2, head_box_2[2]/width_2, head_box_2[3]/height_2]).float()
        
        if eye_loc_v1[0]!=-1.0:
            eye_loc_v1 = np.array([eye_loc_v1[0]/width_1, eye_loc_v1[1]/height_1])
        if eye_loc_v2[0]!=-1.0:
            eye_loc_v2 = np.array([eye_loc_v2[0]/width_2, eye_loc_v2[1]/height_2])
        eye_loc_v1, eye_loc_v2 = torch.tensor(eye_loc_v1).float(), torch.tensor(eye_loc_v2).float()    
        
    
        gaze_x1, gaze_y1, gaze_x2, gaze_y2 = gaze_coord_1[0], gaze_coord_1[1], gaze_coord_2[0], gaze_coord_2[1] 
        gaze_heatmap_1, gaze_heatmap_2 = torch.zeros(self.hm_size[1], self.hm_size[0]), torch.zeros(self.hm_size[1], self.hm_size[0])
        if gaze_valid_1 and visib_1!=0:
            gaze_heatmap_1 = draw_labelmap(gaze_heatmap_1, [gaze_x1 * self.hm_size[0], gaze_y1 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        if gaze_valid_2 and visib_2!=0:
            gaze_heatmap_2 = draw_labelmap(gaze_heatmap_2, [gaze_x2 * self.hm_size[0], gaze_y2 * self.hm_size[1]],
                                                    3,
                                                    type='Gaussian')
        
        
        img_1, img_2 = self.img_transform(image=img_1)["image"], self.img_transform(image=img_2)['image']
        if head_valid_1:
            head_img_1 = self.head_transform(image=head_img_1)["image"]
        if head_valid_2:
            head_img_2 = self.head_transform(image=head_img_2)["image"]
        
        img_1, img_2 = torch.tensor(img_1).float(), torch.tensor(img_2).float()
        head_img_1, head_img_2 = torch.tensor(head_img_1).float(), torch.tensor(head_img_2).float()   
        img_1, img_2, head_img_1, head_img_2 = img_1.permute(2, 0, 1), img_2.permute(2, 0, 1), head_img_1.permute(2, 0, 1), head_img_2.permute(2, 0, 1)
        gaze_coord_1, gaze_coord_2 = torch.tensor([gaze_x1, gaze_y1]).float(), torch.tensor([gaze_x2, gaze_y2]).float()
        visib_1, visib_2 = torch.tensor(visib_1).long(), torch.tensor(visib_2).long()
        head_valid_1, head_valid_2 = torch.tensor(head_valid_1).long(), torch.tensor(head_valid_2).long()
        
        depth_v1, depth_v2 = torch.tensor(depth_v1).float(), torch.tensor(depth_v2).float()
        abs_depth_v1, abs_depth_v2 = torch.tensor(abs_depth_v1).float(), torch.tensor(abs_depth_v2).float()
        if depth_v1.size(0)!=self.img_out_size[1] or depth_v1.size(1)!=self.img_out_size[0]:
            depth_v1 = F.interpolate(depth_v1.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if depth_v2.size(0)!=self.img_out_size[1] or depth_v2.size(1)!=self.img_out_size[0]:
            depth_v2 = F.interpolate(depth_v2.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if abs_depth_v1.size(0)!=self.img_out_size[1] or abs_depth_v1.size(1)!=self.img_out_size[0]:
            abs_depth_v1 = F.interpolate(abs_depth_v1.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        if abs_depth_v2.size(0)!=self.img_out_size[1] or abs_depth_v2.size(1)!=self.img_out_size[0]:
            abs_depth_v2 = F.interpolate(abs_depth_v2.unsqueeze(0).unsqueeze(0), size=(self.img_out_size[1], self.img_out_size[0]), mode='bilinear', align_corners=True).squeeze()
        
         
        # get normalized depth values  
        depth_max, depth_min = depth_v1.max(), depth_v1.min()
        depth_v1_norm = (depth_v1 - depth_min) / (depth_max - depth_min + eps)
        depth_v1_norm = 1 - depth_v1_norm  # reverse the depth image
        depth_max, depth_min = depth_v2.max(), depth_v2.min()
        depth_v2_norm = (depth_v2 - depth_min) / (depth_max - depth_min + eps)
        depth_v2_norm = 1 - depth_v2_norm  # reverse the depth image
        
        
        # process camera parameters
        ratio_w, ratio_h = self.img_out_size[0] / width_1, self.img_out_size[1]/height_1
        intri_1[0,:] = intri_1[0,:] * ratio_w
        intri_1[1,:] = intri_1[1,:] * ratio_h
        ratio_w, ratio_h = self.img_out_size[0] / width_2, self.img_out_size[1]/height_2
        intri_2[0,:] = intri_2[0,:] * ratio_w
        intri_2[1,:] = intri_2[1,:] * ratio_h
    
        fund_mat = torch.tensor(self.fundamental_RT_op(intri_1,RT_1, intri_2, RT_2)).float()
        intri_1, intri_2 = torch.from_numpy(intri_1).float(), torch.from_numpy(intri_2).float()
        RT_1, RT_2 = torch.from_numpy(RT_1).float(), torch.from_numpy(RT_2).float()
        R1, R2 = RT_1[:, :3], RT_2[:, :3]
        # convert to quaternion
        R_2to1, R_1to2 = R1 @ R2.T, R2 @ R1.T
        Rot_1, Rot_2 = Rotation.from_matrix(R_2to1.numpy()), Rotation.from_matrix(R_1to2.numpy())
        quat_1, quat_2 = torch.tensor(Rot_1.as_quat()).float(), torch.tensor(Rot_2.as_quat()).float()
        
        gaze_3d = self.gaze_3d_list[index]
        if torch.all(gaze_3d==0):
            gtvec_3d_1, gtvec_3d_2 = torch.zeros(3).float(), torch.zeros(3).float()
        else:
            gaze_3d_1 = R1 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_1 = torch.linalg.norm(gaze_3d_1)
            gtvec_3d_1 = gaze_3d_1 / (gaze_3d_norm_1 + eps)
            gaze_3d_2 = R2 @ gaze_3d.unsqueeze(1).view(-1)  # convert to camera coordinate
            gaze_3d_norm_2 = torch.linalg.norm(gaze_3d_2)
            gtvec_3d_2 = gaze_3d_2 / (gaze_3d_norm_2 + eps)
        
        img, head_img, head_mask_scene, depth = torch.stack((img_1, img_2)), torch.stack((head_img_1, head_img_2)), torch.stack((head_mask_scene_1, head_mask_scene_2)), torch.stack((depth_v1, depth_v2))
        gaze_heatmap, visib, eye_loc, gaze_coord, head_valid = torch.stack((gaze_heatmap_1, gaze_heatmap_2)), torch.stack((visib_1, visib_2)), torch.stack((eye_loc_v1, eye_loc_v2)), torch.stack((gaze_coord_1, gaze_coord_2)), torch.stack((head_valid_1, head_valid_2))
        gtvec_3d = torch.stack((gtvec_3d_1, gtvec_3d_2))
        head_coords = torch.stack((head_coords_1, head_coords_2))
        intri, RT = torch.stack((intri_1, intri_2)), torch.stack((RT_1, RT_2))
        path_info = os.path.join(scene, subj_folder, cam_1, cam_2, filename)
        quat = torch.stack((quat_1, quat_2))
        facevis = torch.tensor([facevis_1, facevis_2]).int()
        abs_depth = torch.stack((abs_depth_v1, abs_depth_v2))
        
        
        data_dict = {
            "data": (img, head_img, head_mask_scene, depth, gaze_heatmap, visib, gaze_coord, gtvec_3d, head_valid, eye_loc, head_coords, intri, RT, quat),
            "fund_mat": fund_mat,
            "path": path_info,
            'main_id': main_view_index,
            'face_vis': facevis,
            'abs_depth': abs_depth
        }
        
        return data_dict
         
    def __len__(self):
        return len(self.input_list)
    
    def get_transform(self, out_shape):
        transform_list = []
        transform_list.append(A.Resize(out_shape[0], out_shape[1], interpolation=cv2.INTER_AREA))
        #transform_list.append(A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0)) # divide by 255
        #transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0))
        transform_list.append(A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
        return A.Compose(transform_list)

    def process_head(self, img, head_box, head_valid, width, height):
        head_x_min, head_y_min, head_x_max, head_y_max = head_box
        
        head_img = np.zeros((self.head_out_size[1], self.head_out_size[0], 3))
        head_mask_scene = torch.zeros((1, self.img_out_size[1], self.img_out_size[0]))  # head w.r.t scene image
        
        if head_valid:
            head_x1_scene, head_y1_scene, head_x2_scene, head_y2_scene = round(head_x_min / width * self.img_out_size[0]), round(head_y_min / height * self.img_out_size[1]), round(head_x_max / width * self.img_out_size[0]), round(head_y_max / height * self.img_out_size[1])
            head_mask_scene[:, head_y1_scene:head_y2_scene, head_x1_scene:head_x2_scene] = 1    
            #head_img = img.crop((int(head_x_min), int(head_y_min), int(head_x_max), int(head_y_max)))
            head_img = img[int(head_y_min):int(head_y_max), int(head_x_min):int(head_x_max), :]
            
        return head_img, head_mask_scene
    
    
    def get_eye_keypoint(self, annt, conf_thres=1.2):
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
    
    def data_augmentation_albument(self, img, depth_img, abs_depth_img, head_box, gaze_coord, eye_loc, intri, width, height):
        
        x_min, y_min, x_max, y_max = head_box
        head_width, head_height = x_max - x_min, y_max - y_min
        
        head_valid = x_min>=0
        gaze_valid = gaze_coord[0]>=0
        depth_height, depth_width = depth_img.shape[:2]
        abs_depth_height, abs_depth_width = abs_depth_img.shape[:2]
        
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

            crop_xmin_depth, crop_xmax_depth = int(crop_x_min * abs_depth_width / width), int(crop_x_max * abs_depth_width / width)
            crop_ymin_depth, crop_ymax_depth = int(crop_y_min * abs_depth_height / height), int(crop_y_max * abs_depth_height / height)
            abs_depth_img = A.Crop(crop_xmin_depth, crop_ymin_depth, crop_xmax_depth, crop_ymax_depth)(image=abs_depth_img)['image']

            
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
        
        return img, depth_img, abs_depth_img, head_box, gaze_coord, eye_loc, intri  # return the later 4 to apply consistent color changes

    def get_inout_patch_logits(self, gaze_heatmap):
        patch_size = self.hm_size[0] // 7  # modify here
        steps = 7
        inout_patch = []
        for i in range(steps):
            for j in range(steps):
                inout_patch.append(gaze_heatmap[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].max())
        
        inout_patch = torch.tensor(inout_patch)
        return inout_patch 