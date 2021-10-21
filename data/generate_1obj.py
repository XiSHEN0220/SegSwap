import os 
import PIL.Image as Image
import torch
import torch.nn.functional as F

import numpy as np
import argparse
from tqdm import tqdm

import utils
import cv2

import sys
sys.path.append('AdaIn/')
import net
import style_transfer

import math
import torchvision.transforms as transforms
from datetime import datetime
import json 
import random


def AspectRatio(I, M, scalew, scaleh, R) : 
    w, h = I.size
    new_w = int(w * scalew)
    new_h = int(h * scaleh)
    Rar = np.array([[float(new_w) / w, 0, 0],
                      [0, float(new_h) / h, 0],
                    [0, 0, 1]])
    
    
    return I.resize((new_w, new_h), resample=2), M.resize((new_w, new_h), resample=2), Rar @ R

def RandomScale(scale_min, scale_max = 1) : 
    return np.random.rand() * (scale_max - scale_min) + scale_min

def Rotate(I, M, angle, R):
    if angle >= 0 :
        angle_radian = angle / 180 * math.pi
        Rrot = np.array([[math.cos(angle_radian), math.sin(angle_radian), 0],
                          [-1 * math.sin(angle_radian), math.cos(angle_radian), I.size[0] * math.sin(angle_radian)],
                        [0, 0, 1]])
    else : 
        angle = -1 * angle / 180 * math.pi
        angle_radian = angle / 180 * math.pi
        Rrot = np.array([[math.cos(angle_radian), -1 * math.sin(angle_radian), I.size[1] * math.sin(angle_radian)],
                          [math.sin(angle_radian), math.cos(angle_radian), 0],
                        [0, 0, 1]])
    return I.rotate(angle, expand=1), M.rotate(angle, expand=1), Rrot @ R

def RandomAngle(angle_min, angle_max) : 
    return np.random.rand() * (angle_max - angle_min) + angle_min

def cropbb(I, M, R) : 
    BBfg_aug = utils.bbox_from_binarymask(np.array(M) > 128)
    Rcrop = np.array([[1, 0, -BBfg_aug[0]],
                      [0, 1, -BBfg_aug[1]],
                    [0, 0, 1]])
    Pfg_aug = I.crop(BBfg_aug)
    Mfg_aug = M.crop(BBfg_aug)
    
    return Pfg_aug, Mfg_aug, Rcrop @ R

def resize_patch_bg(I, M, img_size, R, ratio_list, ratio_prob) : 
    w, h = I.size
    
    ratio_w = img_size[0] / w
    ratio_h = img_size[1] / h

    
    ratio = min(ratio_w, ratio_h) * (np.random.choice(ratio_list, p=ratio_prob))
    
    w_resize, h_resize = int(w * ratio), int(h * ratio)
    
    Rresize = np.array([[float(w_resize) / w, 0, 0],
                      [0, float(h_resize) / h, 0],
                    [0, 0, 1]])
    
    Iresize = I.resize((w_resize, h_resize), resample=2)
    Mresize = M.resize((w_resize, h_resize), resample=2)
    return Iresize, Mresize, Rresize@R


def copypaste_poisson(patch_arr, patch_mask_arr, Ibg_arr, R, img_size) : 
    
    patch_w, patch_h = patch_arr.shape[1], patch_arr.shape[0]
    h_start = np.random.randint(img_size[1] - patch_h + 1)
    w_start = np.random.randint(img_size[0] - patch_w + 1)
    
    Rpe = np.array([[1, 0, w_start],
                    [0, 1, h_start],
                    [0, 0, 1]])
    
    h_center = h_start + patch_h // 2 - 1
    w_center = w_start + patch_w // 2 - 1

    Ibg_blend = cv2.seamlessClone(patch_arr, Ibg_arr, patch_mask_arr, (w_center, h_center), cv2.NORMAL_CLONE)
    
    Ibg_mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
    Ibg_mask[h_start:h_start + patch_h, w_start:w_start + patch_w] =  patch_mask_arr
        
    return Image.fromarray(Ibg_blend), Image.fromarray(Ibg_mask), Rpe@R


def trans_pos(grid_x, grid_y, R, img_size) : 
    
    ## corrdinate on org image 1
    x = grid_x.flatten() * img_size[0]
    y = grid_y.flatten() * img_size[1]

    xy = np.vstack((x, y, np.ones(x.shape)))
    
    ## corrdinate on org image 2
    xy_update = R @ xy
    
    return xy_update[0, :] / img_size[0], xy_update[1, :] / img_size[1]

def sample_one_image (img_list, img2cat) : 
    
    img1= np.random.choice(img_list, 1, replace=False)
    cat1 = img2cat[img1[0]]
    return img1[0], cat1
        
def sample_one_obj(cat2obj, cat_set, cat_exclude) : 
    cat_dispo = cat_set - cat_exclude
    cat = np.random.choice(list(cat_dispo), 1, replace=False)
    
    obj_dispo = cat2obj[cat[0]]
    
    obj1 = np.random.choice(range(len(obj_dispo)), 1)[0]
    obj1 = obj_dispo[obj1]
    
    return obj1


def crop_left(Ifg, Mfg, w, h, left) : 
    return Ifg.crop((left, 0, w, h)), Mfg[:, left:]

def crop_right(Ifg, Mfg, w, h, right) : 
    return Ifg.crop((0, 0, right, h)), Mfg[:, :right]

def crop_top(Ifg, Mfg, w, h, top) : 
    return Ifg.crop((0, top, w, h)), Mfg[top:, :]

def crop_bottom(Ifg, Mfg, w, h, bottom) : 
    return Ifg.crop((0, 0, w, bottom)), Mfg[:bottom, :]

def crop_fg(Ifg, Mfg) : 
    idh, idw = np.where(Mfg)
    left, right = idw.min(), idw.max() + 1
    top, bottom= idh.min(), idh.max() + 1
    w, h = Ifg.size
    
    cropchoices = [(crop_left, left), (crop_right, right), (crop_top, top), (crop_bottom, bottom)]
    idx = np.argmin([left, w-right, top, h-bottom])
    return cropchoices[idx][0](Ifg, Mfg, w, h, cropchoices[idx][1])

def consistent_dilate_mask(mask1, x1, y1, x2, y2) : 
    
    flow12_inside = (x1 > 0) & (x1 < 1) & (y1 > 0) & (y1 < 1)
    flow21_inside = (x2 > 0) & (x2 < 1) & (y2 > 0) & (y2 < 1)

    flow12 = torch.cat((torch.from_numpy(x1.astype(np.float32)).unsqueeze(0),
                      torch.from_numpy(y1.astype(np.float32)).unsqueeze(0)), dim=1)
    flow12 = (flow12 - 0.5)  * 2
    flow12 = flow12.permute(0, 2, 3, 1)

    flow21 = torch.cat((torch.from_numpy(x2.astype(np.float32)).unsqueeze(0),
                      torch.from_numpy(y2.astype(np.float32)).unsqueeze(0)), dim=1)
    flow21 = (flow21 - 0.5)  * 2
    flow21 = flow21.permute(0, 2, 3, 1)

    mask2_grid = F.grid_sample(torch.from_numpy(mask1.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow21, mode='bilinear').numpy().squeeze()
    mask2_grid = mask2_grid * flow21_inside[0]

    mask1_back = F.grid_sample(torch.from_numpy(mask2_grid.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow12, mode='bilinear')
    mask1_final  = (mask1 * mask1_back.squeeze().numpy() * flow12_inside[0])

    mask2_final = F.grid_sample(torch.from_numpy(mask1_final.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow21, mode='bilinear')
    mask2_final = (mask2_final.squeeze().numpy() * flow21_inside[0])
    return mask1_final, mask2_final

def style_transfer_pil(I, style_list, style_dir, vgg, decoder, device) : 

    style_path = np.random.choice(style_list)
    style_pil = Image.open(os.path.join(style_dir, style_path)).convert('RGB')
    I_stylised = style_transfer.style_transfer_pil_input(vgg, decoder, device, I, style_pil, np.random.rand())    
    I_stylised = I_stylised.resize(I.size, resample=2)
    return I_stylised


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--style-dir', type=str, default= 'Brueghel/Image/', help='style image directory (Brueghel)')


parser.add_argument(
    '--coco-dir', type=str, default= 'COCO2017/train2017/', help='coco image directory')

parser.add_argument(
    '--out-dir', type=str, default=None, help='output dir')

parser.add_argument(
    '--start-idx', type=int, default=0, help='training samples: start idx')

parser.add_argument(
    '--end-idx', type=int, default=100000, help='training samples: end idx')

parser.add_argument(
    '--ar-scale-min', type=float, default=0.7, help='scale minimum for shear')


parser.add_argument(
    '--obj-scale-bg-min', type=float, default=0.2, help='obj in bg, small ratio')

parser.add_argument(
    '--obj-scale-bg-max', type=float, default=1, help='obj in bg, max ratio')

parser.add_argument(
    '--angle-min', type=float, default=-15, help='angle minimum of rotation')

parser.add_argument(
    '--angle-max', type=float, default=15, help='angle maxmium of rotation')

parser.add_argument(
    '--info-json', type=str, default='COCO2017_train_img2cat.json', help='coco image information, a dictionary contains: img : [category1, category2, ... ]')

parser.add_argument(
    '--obj-json', type=str, default='COCO2017_train_obj.json', help='coco object information, a dictionary contains: category : object')

parser.add_argument(
    '--gpu', type=str, default='0', help='gpu id')

parser.add_argument(
    '--decoder-pth', type=str, default='AdaIn/Adain_decoder_iter160K.pth.tar', help='decoder path')

parser.add_argument(
    '--vgg-pth', type=str, default='AdaIn/vgg_normalised.pth', help='VGG path')

parser.add_argument(
    '--img-size', type=int, nargs='+', default=[480, 480], help='image size')

parser.add_argument(
    '--iter-dilation', type=int, default=20, help='nb of iteration for dilation before poisson editing')

args = parser.parse_args()

print (args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda")


## load style transfer network
decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder_pth))
vgg.load_state_dict(torch.load(args.vgg_pth))
vgg = torch.nn.Sequential(*list(vgg.children())[:31])

vgg.to(device)
decoder.to(device)

style_list = sorted(os.listdir(args.style_dir))


stride_net = 16
nb_feat_w = int(args.img_size[0] / stride_net)
nb_feat_h = int(args.img_size[1] / stride_net)

x = ( np.arange(nb_feat_w) + 0.5 ) / nb_feat_w
y = ( np.arange(nb_feat_h) + 0.5 ) / nb_feat_h
grid_x, grid_y = np.meshgrid(x, y)

## foreground object will take a sampled ratio (sampled from [args.ratio_min, args.ratio_max]) in the bg image 
ratio_list = np.linspace(args.obj_scale_bg_min, args.obj_scale_bg_max, 10)
ratio_prob = np.exp(-np.arange(10) / 10)
ratio_prob = ratio_prob / ratio_prob.sum()

utils.mkdir_directory(args.out_dir)

## load image to category json
img2cat = utils.load_json(args.info_json)
for img in img2cat : 
    img2cat[img] = set(img2cat[img])
img_list = sorted(list(img2cat.keys()))

## load category to obj json file 
cat2obj = utils.load_json(args.obj_json)
cat_set = set(list(cat2obj.keys()))

count = args.start_idx

while count <args.end_idx :
    if count % 50 == 49 :
        msg = '{} \t generating {:d} samples...'.format(datetime.now().time(), count)
        print (msg)
    
    ## sample one images as bg
    bg, cat_bg = sample_one_image(img_list, img2cat)
    bg_name = utils.coco2017_id2name(bg)
    Ibg = Image.open(os.path.join(args.coco_dir, bg_name)).convert('RGB')
    Ibg = Ibg.resize((args.img_size[0], args.img_size[1]), resample=2) # bilinear resample

    ## sample one object as fg: 
    obj = sample_one_obj(cat2obj, cat_set, cat_bg)
    fg = obj[0]

    ## real image
    fg_name = utils.coco2017_id2name(fg)
    Ifg = Image.open(os.path.join(args.coco_dir, fg_name)).convert('RGB')

    polygon_fg = obj[1]
    Mfg = utils.mask_from_polygon(Ifg.size[0], Ifg.size[1], polygon_fg) 
    
    Mfg = utils.dilation(Mfg, nb_iter = max(args.iter_dilation, 1))
    
    if np.random.rand() > 0.8 :
        Ifg, Mfg = crop_fg(Ifg, Mfg)

    ## foreground image + mask are ready
    Ifg = Ifg.resize((args.img_size[0], args.img_size[1]), resample=2) # bilinear resample
    Mfg = utils.resize_mask(Mfg, args.img_size[0], args.img_size[1])

    # final mask foreground
    mask_fg = F.interpolate(transforms.ToTensor()(Mfg).unsqueeze(0), size = (nb_feat_h, nb_feat_w), mode='bilinear').squeeze().numpy()  # final mask foreground

    ### change aspect ratio
    Ifg_as, Mfg_as, Ras = AspectRatio(Ifg, Mfg, RandomScale(args.ar_scale_min), RandomScale(args.ar_scale_min), np.eye(3))

    ### rotate object
    Ifg_rot, Mfg_rot, Rrot = Rotate(Ifg_as, Mfg_as, RandomAngle(args.angle_min, args.angle_max), Ras)

    ### crop the object
    Ifg_aug, Mfg_aug, Rcrop = cropbb(Ifg_rot, Mfg_rot, Rrot)

    ### resize the object
    Ifg_resize, Mfg_resize, Rresize = resize_patch_bg(Ifg_aug, Mfg_aug, args.img_size, Rcrop, ratio_list, ratio_prob)
    
    ### blending the object 
    Ibg_blend, M_blend, Rpe  = copypaste_poisson(np.array(Ifg_resize), np.array(Mfg_resize), np.array(Ibg), Rresize, args.img_size)

    # final mask background
    mask_bg = np.zeros((nb_feat_h, nb_feat_w), dtype=bool) 
    mask_bg = F.interpolate(transforms.ToTensor()(M_blend).unsqueeze(0), size = (nb_feat_h, nb_feat_w), mode='bilinear').squeeze().numpy()  # final mask foreground

    x1, y1 = trans_pos(grid_x, grid_y, Rpe, args.img_size)
    x1, y1 = x1.reshape((1, int(nb_feat_h), int(nb_feat_w))), y1.reshape((1, int(nb_feat_h), int(nb_feat_w)))
    x2, y2 = trans_pos(grid_x, grid_y, np.linalg.inv(Rpe), args.img_size) 
    x2, y2 = x2.reshape((1, int(nb_feat_h), int(nb_feat_w))), y2.reshape((1, int(nb_feat_h), int(nb_feat_w)))

    mask_fg_dilate, mask_bg_dilate = consistent_dilate_mask(mask_fg, x1, y1, x2, y2)
    
    if mask_fg_dilate.sum() < 9 or mask_bg_dilate.sum() < 9 : 
        continue
        
    ## style transfer
    Ifg_style = style_transfer_pil(Ifg, style_list, args.style_dir, vgg, decoder, device)
    Ibg_style = style_transfer_pil(Ibg_blend, style_list, args.style_dir, vgg, decoder, device)
    

    out1 = os.path.join(args.out_dir, '{:d}_a.jpg'.format(count))
    out2 = os.path.join(args.out_dir, '{:d}_b.jpg'.format(count))
    out4 = os.path.join(args.out_dir, '{:d}_as.jpg'.format(count))
    out5 = os.path.join(args.out_dir, '{:d}_bs.jpg'.format(count))
    
    
    Ifg.save(out1)
    Ibg_blend.save(out2)
    
    Ifg_style.save(out4)
    Ibg_style.save(out5)
    
    ## correspondences
    out1 = os.path.join(args.out_dir, '{:d}_a.npy'.format(count))
    out2 = os.path.join(args.out_dir, '{:d}_b.npy'.format(count))

    np.save(out1, np.concatenate((x1, y1, np.expand_dims(mask_fg_dilate, axis=0)), axis=0).astype(np.float32))
    np.save(out2, np.concatenate((x2, y2, np.expand_dims(mask_bg_dilate, axis=0)), axis=0).astype(np.float32))
    
    count += 1
    

        
        
