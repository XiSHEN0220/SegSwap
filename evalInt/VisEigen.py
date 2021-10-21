import os
import PIL.Image as Image 

import numpy as np 

import torch.nn.functional as F
import torch 
from cv2 import cv2

def interpolate(arr_pred, w, h) : 
    tensor_pred = torch.from_numpy(arr_pred).unsqueeze(0).unsqueeze(0)
    tensor_pred = F.interpolate(tensor_pred, size = (h, w), mode = 'bilinear')
    return tensor_pred.squeeze().numpy()

        

def update_mask(mask1, mask2, count1, count2, pts1_x, pts1_y, pts2_x, pts2_y, eigen_img, nb_feat_w, nb_feat_h, mode = 'sum') : 
    
    pts1_x = pts1_x.astype(np.int64).reshape(-1)
    pts1_y = pts1_y.astype(np.int64).reshape(-1)
    pts2_x = pts2_x.astype(np.int64).reshape(-1)
    pts2_y = pts2_y.astype(np.int64).reshape(-1)
    eigen_img = eigen_img.reshape(-1)
    
    if mode == 'sum' : 
        mask1[pts1_y, pts1_x] = mask1[pts1_y, pts1_x] + eigen_img
        mask2[pts2_y, pts2_x] = mask2[pts2_y, pts2_x] + eigen_img
    elif mode == 'max' :
        mask1[pts1_y, pts1_x] = np.maximum(mask1[pts1_y, pts1_x], eigen_img)
        mask2[pts2_y, pts2_x] = np.maximum(mask2[pts2_y, pts2_x], eigen_img)
    elif mode == 'mean' :
        mask1[pts1_y, pts1_x] = mask1[pts1_y, pts1_x] + eigen_img
        mask2[pts2_y, pts2_x] = mask2[pts2_y, pts2_x] + eigen_img
        count1[pts1_y, pts1_x] = count1[pts1_y, pts1_x] + 1
        count2[pts2_y, pts2_x] = count2[pts2_y, pts2_x] + 1
        
    
    return mask1, mask2, count1, count2

def VisEigen(out_dir, img_idx_start_end, corr, img_dir, nb_feat_w, nb_feat_h, img_list, eigen_vector, mode = 'sum') : 
    
    
    ## nb of images / nb of pairs
    nb_img = len(img_list)
    nb_pair = len(img_idx_start_end)
    

    
    mask_list = [ np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32) for img in img_list ]
    count_list = [ np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32) for img in img_list ]
    
    for pair_id in range(nb_pair) :
            
        start = img_idx_start_end[pair_id, 2]
        end = img_idx_start_end[pair_id, 3]
        eigen_img = eigen_vector[start : end]
        
        i, j = img_idx_start_end[pair_id, :2]
        
        mask_list[i], mask_list[j], count_list[i], count_list[j] = update_mask(mask_list[i], mask_list[j], count_list[i], count_list[j], corr[start : end, 0], corr[start : end, 1], corr[start : end, 2], corr[start : end, 3], eigen_img, nb_feat_w, nb_feat_h, mode )
    
    if mode == 'mean' : 
        for i in range(len(mask_list)) : 
            mask_list[i] = mask_list[i] / (count_list[i] + 1e-10)
            
    if out_dir is not None : 
        for i in range(len(img_list)) : 

            I = cv2.imread(os.path.join(img_dir, img_list[i]))
            h, w = I.shape[:2]
            mask = mask_list[i]
            mask_max = mask.max()
            mask_sum = mask.sum()
            mask_up = interpolate(mask, w, h)

            mask_uint8 = (mask_up / (mask_up.max() + 1e-10) * 255).astype(np.uint8)
            heatmap_img = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap_img, 0.3, I, 0.7, 0)

            out_name = os.path.join(out_dir, 'idx_{:d}_sum_{:.3f}_max_{:.3f}.jpg'.format(i, mask_sum, mask_max))

            cv2.imwrite(out_name, heatmap)
    
    return mask_list



def VisPairEigen(out_dir, img_idx_start_end, corr, img_dir, nb_feat_w, nb_feat_h, img_list, eigen_vector) : 
    
    
    ## nb of images / nb of pairs
    nb_img = len(img_list)
    nb_pair = len(img_idx_start_end)
    
    eig_max = eigen_vector.max()
    print (eig_max)
    mask_dict = {i : [] for i in range(len(img_list))}
    for pair_id in range(nb_pair) :
            
        start = img_idx_start_end[pair_id, 2]
        end = img_idx_start_end[pair_id, 3]
        eigen_img = eigen_vector[start : end]
        
        i, j = img_idx_start_end[pair_id, :2]
        
        mask1 = np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32)
        mask2 = np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32)
        
        mask1, mask2 = update_mask(mask1, mask2, corr[start : end, 0], corr[start : end, 1], corr[start : end, 2], corr[start : end, 3], eigen_img, nb_feat_w, nb_feat_h)
        mask_dict[i].append(mask1)
        mask_dict[j].append(mask2)
        
        
        raw_mask1, raw_mask2 = update_mask(np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32), np.zeros((nb_feat_h, nb_feat_w), dtype=np.float32), corr[start : end, 0], corr[start : end, 1], corr[start : end, 2], corr[start : end, 3], np.ones(eigen_img.shape), nb_feat_w, nb_feat_h)
    
    
        I1 = cv2.imread(os.path.join(img_dir, img_list[i]))
        h, w = I1.shape[:2]
        mask = mask1
        mask_max = mask.max()
        mask_up = interpolate(mask, w, h)
        raw_mask_up = interpolate(raw_mask1, w, h).reshape((h, w, 1))
        raw_mask_up = ( raw_mask_up > 0.0 ).astype(np.uint8)
        
        mask_uint8 = (mask_up / eig_max * 255).astype(np.uint8)
        
        mask_non_zero = mask_uint8 > 0
        mask_uint8[mask_non_zero] = np.clip(mask_uint8[mask_non_zero], a_min = 128, a_max = 255)
        heatmap_img = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap_img, 0.3, I1, 0.7, 0)

        out_name = os.path.join(out_dir, 'pair_{:d}_{:d}_max_{:.3f}_1.jpg'.format(i, j, mask_max * 10000))
        cv2.imwrite(out_name, heatmap)
        out_name_raw = os.path.join(out_dir, 'pair_{:d}_{:d}_max_{:.3f}_1_raw.jpg'.format(i, j, mask_max* 10000))
        cv2.imwrite(out_name_raw, I1 * raw_mask_up)
        
        
        I2 = cv2.imread(os.path.join(img_dir, img_list[j]))
        h, w = I2.shape[:2]
        mask = mask2
        mask_up = interpolate(mask, w, h)
        raw_mask_up = interpolate(raw_mask2, w, h).reshape((h, w, 1))
        raw_mask_up = ( raw_mask_up > 0.0 ).astype(np.uint8)
        
        
        mask_uint8 = (mask_up / eig_max * 255).astype(np.uint8)
        mask_non_zero = mask_uint8 > 0
        mask_uint8[mask_non_zero] = np.clip(mask_uint8[mask_non_zero], a_min = 128, a_max = 255)
        
        heatmap_img = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.addWeighted(heatmap_img, 0.3, I2, 0.7, 0)

        out_name = os.path.join(out_dir, 'pair_{:d}_{:d}_max_{:.3f}_2.jpg'.format(i, j, mask_max* 10000))

        cv2.imwrite(out_name, heatmap)
        
        
        out_name_raw = os.path.join(out_dir, 'pair_{:d}_{:d}_max_{:.3f}_2_raw.jpg'.format(i, j, mask_max* 10000))
        cv2.imwrite(out_name_raw, I2 * raw_mask_up)
        
        
    dict_mean = {}
    dict_max = {}
    mean_norm = 0
    max_norm = 0
    
    for i in range(len(img_list)) : 
        if len(mask_dict[i]) > 0 : 
            mask_list = np.stack(mask_dict[i], axis=0)
            mask_mean = mask_list.mean(axis=0)
            dict_mean[i] = mask_mean
            mean_norm = max(mask_mean.max(), mean_norm)

            mask_max = mask_list.max(axis=0)
            max_norm = max(mask_max.max(), max_norm)
            dict_max[i] = mask_max

    
    
    for i in range(len(img_list)) : 
        if i in dict_mean : 
            
            mask = dict_mean[i]
            mask_max = mask.max()
            mask = mask / mean_norm
            I = cv2.imread(os.path.join(img_dir, img_list[i]))
            h, w = I.shape[:2]
            mask_up = interpolate(mask, w, h)
            mask_uint8 = (mask_up * 255).astype(np.uint8)
            mask_non_zero = mask_uint8 > 0
            mask_uint8[mask_non_zero] = np.clip(mask_uint8[mask_non_zero], a_min = 128, a_max = 255)
            heatmap_img = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap_img, 0.3, I, 0.7, 0)

            out_name = os.path.join(out_dir, 'imageMean_{:d}_max_{:.3f}.jpg'.format(i, mask_max* 10000))
            cv2.imwrite(out_name, heatmap)
            
            
            mask = dict_max[i]
            mask_max = mask.max()
            mask = mask / max_norm
            I = cv2.imread(os.path.join(img_dir, img_list[i]))
            h, w = I.shape[:2]
            mask_up = interpolate(mask, w, h)
            mask_uint8 = (mask_up * 255).astype(np.uint8)
            mask_non_zero = mask_uint8 > 0
            mask_uint8[mask_non_zero] = np.clip(mask_uint8[mask_non_zero], a_min = 128, a_max = 255)
            heatmap_img = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
            heatmap = cv2.addWeighted(heatmap_img, 0.3, I, 0.7, 0)

            out_name = os.path.join(out_dir, 'imageMax_{:d}_max_{:.3f}.jpg'.format(i, mask_max* 10000))
            cv2.imwrite(out_name, heatmap)
        
            
            
            
    
        
                
        