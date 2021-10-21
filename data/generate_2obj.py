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

    
def AspectRatio(I, M1, scalew, scaleh, R) : 
    w, h = I.size
    new_w = int(w * scalew)
    new_h = int(h * scaleh)
    Rar = np.array([[float(new_w) / w, 0, 0],
                      [0, float(new_h) / h, 0],
                    [0, 0, 1]])
    
    
    return I.resize((new_w, new_h), resample=2), M1.resize((new_w, new_h), resample=2), Rar @ R

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

def resize_bbox_bg(I, M, bbw, bbh, R, ratio_list, ratio_prob) : 
    w, h = I.size
    
    ratio_w = bbw / w
    ratio_h = bbh / h

    
    ratio = min(ratio_w, ratio_h) * (np.random.choice(ratio_list, p=ratio_prob))
    
    w_resize, h_resize = int(w * ratio), int(h * ratio)
    
    Rresize = np.array([[float(w_resize) / w, 0, 0],
                      [0, float(h_resize) / h, 0],
                    [0, 0, 1]])
    
    Iresize = I.resize((w_resize, h_resize), resample=2)
    Mresize = M.resize((w_resize, h_resize), resample=2)
    return Iresize, Mresize, Rresize@R


def resize_max(I, max_size, stride_net) : 
    w, h = I.size
    ratio = max(w / max_size, h / max_size)
    new_w = int(round(w / ratio / stride_net) * stride_net) 
    new_h = int(round(h / ratio / stride_net) * stride_net) 
    return new_w, new_h



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



def poisson_editing_2obj(patch_arr1, patch_arr2, mask_obj1, mask_obj2, bbox1, bbox2, Ibg_arr, img_size, R1, R2) : 
    patch_w1, patch_h1 = patch_arr1.shape[1], patch_arr1.shape[0]
    patch_w2, patch_h2 = patch_arr2.shape[1], patch_arr2.shape[0]
    
    h_start1 = np.random.randint(bbox1[1], bbox1[3] - patch_arr1.shape[0] + 1)
    w_start1 = np.random.randint(bbox1[0], bbox1[2] - patch_arr1.shape[1] + 1)

    Rpe1 = np.array([[1, 0, w_start1],
                    [0, 1, h_start1],
                    [0, 0, 1]])

    h_center1 = h_start1 + patch_arr1.shape[0] // 2 - 1
    w_center1 = w_start1 + patch_arr1.shape[1] // 2 - 1
    
    Ibg_blend = cv2.seamlessClone(patch_arr1, Ibg_arr, mask_obj1.astype(np.uint8) , (w_center1, h_center1), cv2.NORMAL_CLONE)
    
    Ibg_mask1 = np.zeros((img_size, img_size), dtype=np.uint8)
    Ibg_mask1[h_start1:h_start1 + patch_h1, w_start1:w_start1 + patch_w1] =  mask_obj1
    

    h_start2 = np.random.randint(bbox2[1], bbox2[3] - patch_arr2.shape[0] + 1)
    w_start2 = np.random.randint(bbox2[0], bbox2[2] - patch_arr2.shape[1] + 1)

    Rpe2 = np.array([[1, 0, w_start2],
                    [0, 1, h_start2],
                    [0, 0, 1]])

    h_center2 = h_start2 + patch_arr2.shape[0] // 2 - 1
    w_center2 = w_start2 + patch_arr2.shape[1] // 2 - 1

    Ibg_blend = cv2.seamlessClone(patch_arr2, Ibg_blend, mask_obj2.astype(np.uint8), (w_center2, h_center2), cv2.NORMAL_CLONE)
    
    Ibg_mask2 = np.zeros((img_size, img_size), dtype=np.uint8)
    Ibg_mask2[h_start2:h_start2 + patch_h2, w_start2:w_start2 + patch_w2] =  mask_obj2
    
    return Image.fromarray(Ibg_blend), Image.fromarray(Ibg_mask1), Image.fromarray(Ibg_mask2), Rpe1@R1, Rpe2@R2


def consistent_dilate_mask(mask1, x1, y1, x2, y2) : 
    #mask1_dilate = utils.dilation(mask1)

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

def trans_pos(grid_x, grid_y, R, img_size) : 
    
    ## corrdinate on org image 1
    x = grid_x.flatten() * img_size
    y = grid_y.flatten() * img_size

    xy = np.vstack((x, y, np.ones(x.shape)))
    
    ## corrdinate on org image 2
    xy_update = R @ xy
    
    return xy_update[0, :] / img_size, xy_update[1, :] / img_size


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
    '--ar-scale-max', type=float, default=1.0, help='scale max for shear')

parser.add_argument(
    '--obj-scale-bg-min', type=float, default=0.5, help='obj in bg, small ratio')

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
    '--img2obj-json', type=str, default='COCO2017_train_img2obj.json', help='coco object information, a dictionary contains: img : list of object masks')

parser.add_argument(
    '--gpu', type=str, default='2', help='gpu id')

parser.add_argument(
    '--decoder-pth', type=str, default='AdaIn/Adain_decoder_iter160K.pth.tar', help='decoder path')

parser.add_argument(
    '--vgg-pth', type=str, default='AdaIn/vgg_normalised.pth', help='VGG path')

parser.add_argument(
    '--img-size', type=int, default=480, help='image size')

parser.add_argument(
    '--iter-dilation', type=int, default=20, help='nb of iteration for dilation before poisson editing')

args = parser.parse_args()

print (args)

bb_width = args.img_size // 2
stride_net = 16
nb_feat = int(args.img_size / stride_net)
x = ( np.arange(nb_feat) + 0.5 ) / nb_feat
y = ( np.arange(nb_feat) + 0.5 ) / nb_feat
grid_x, grid_y = np.meshgrid(x, y)
choice_bbox = [[(0, 0, bb_width, bb_width - stride_net), (0, bb_width + stride_net, bb_width, args.img_size)],
               [(0, 0, bb_width- stride_net, bb_width- stride_net), (bb_width + stride_net, bb_width + stride_net, args.img_size, args.img_size)],
               [(0, 0, bb_width - stride_net, bb_width), (bb_width + stride_net, 0, args.img_size, bb_width)],
               [(0, 0, bb_width, bb_width - stride_net), (0, bb_width + stride_net, args.img_size, args.img_size)],
               [(0, 0, bb_width - stride_net, bb_width), (bb_width + stride_net, 0, args.img_size, args.img_size)],
               [(0, bb_width, bb_width - stride_net, args.img_size), (bb_width + stride_net, bb_width, args.img_size, args.img_size)],
               [(0, bb_width + stride_net, bb_width - stride_net, args.img_size), (bb_width + stride_net, 0, args.img_size, bb_width - stride_net)],
               [(0, bb_width + stride_net, bb_width, args.img_size), (0, 0, args.img_size, bb_width - stride_net)],
               [(0, bb_width , bb_width- stride_net, args.img_size), (bb_width + stride_net, 0, args.img_size, args.img_size)],
               [(bb_width, bb_width + stride_net, args.img_size, args.img_size), (bb_width, 0, args.img_size, bb_width - stride_net)],
               [(bb_width, bb_width + stride_net, args.img_size, args.img_size), (0, 0, args.img_size, bb_width - stride_net)],
               [(bb_width + stride_net, bb_width, args.img_size, args.img_size), (0, 0, bb_width - stride_net, args.img_size)],
               [(bb_width + stride_net, 0, args.img_size, bb_width), (0, 0, bb_width - stride_net, args.img_size)],
               [(bb_width, 0, args.img_size, bb_width - stride_net), (0, bb_width + stride_net, args.img_size, args.img_size)],
               [(0, 0, args.img_size, bb_width - stride_net), (0, bb_width+ stride_net, args.img_size, args.img_size)],
               [(0, 0, bb_width - stride_net, args.img_size), (bb_width + stride_net, 0, args.img_size, args.img_size)]]


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

img2obj = utils.load_json(args.img2obj_json)

img_list_2cat = []
for img in img2obj : 
    if len(img2obj[img]) >= 2 :
        img_list_2cat.append(img)

count = args.start_idx

while count <args.end_idx :
    if count % 50 == 49 :
        msg = '{} \t generating {:d} samples...'.format(datetime.now().time(), count)
        print (msg)
    
    
    # sample an image as background, padding the images
    bg = np.random.choice(img_list)
    bg_name = utils.coco2017_id2name(bg)
    Ibg = Image.open(os.path.join(args.coco_dir, bg_name)).convert('RGB')
    Ibg = Ibg.resize((args.img_size, args.img_size), resample=2) # bilinear resample

    # sample an image as foreground
    fg = np.random.choice(img_list_2cat)
    cat_list = list(img2obj[fg])
    nb_cat = len(cat_list)

    # sample 2 catogories
    fg_cat = np.random.choice(nb_cat, 2 ,replace=False)
    cat1, cat2 = fg_cat
    obj1, obj2 = np.random.choice(len(img2obj[fg][cat_list[cat1]])),  np.random.choice(len(img2obj[fg][cat_list[cat2]]))
    fg_name = utils.coco2017_id2name(fg)
    Ifg = Image.open(os.path.join(args.coco_dir, fg_name)).convert('RGB')


    ## obj masks
    polygon1, polygon2 = img2obj[fg][cat_list[cat1]][obj1][1], img2obj[fg][cat_list[cat2]][obj2][1]

    obj_mask1 = utils.mask_from_polygon(Ifg.size[0], Ifg.size[1], polygon1)
    obj_mask2 = utils.mask_from_polygon(Ifg.size[0], Ifg.size[1], polygon2)

    ## dilate the mask
    obj_mask1 = utils.dilation(obj_mask1, nb_iter = max(args.iter_dilation, 1))
    obj_mask2 = utils.dilation(obj_mask2, nb_iter = max(args.iter_dilation, 1))

    ## if two masks overlapped, drop it
    if (obj_mask1 * obj_mask2).sum() > 0 : 
        continue

    ## foreground image + mask are ready
    Ifg = Ifg.resize((args.img_size, args.img_size), resample=2) # bilinear resample
    obj_mask1 = utils.resize_mask(obj_mask1, args.img_size, args.img_size)
    obj_mask2 = utils.resize_mask(obj_mask2, args.img_size, args.img_size)

    # final mask for object 1
    Mfg1 = F.interpolate(transforms.ToTensor()(obj_mask1).unsqueeze(0), size = (nb_feat, nb_feat), mode='bilinear').squeeze().numpy()  

    # final mask for object 1
    Mfg2 = F.interpolate(transforms.ToTensor()(obj_mask2).unsqueeze(0), size = (nb_feat, nb_feat), mode='bilinear').squeeze().numpy()  

    bbox1, bbox2 = choice_bbox[np.random.choice(len(choice_bbox))]

    ### object 1
    ### change aspect ratio
    Ifg_as1, Mfg_as1, Ras1 = AspectRatio(Ifg, obj_mask1, RandomScale(args.ar_scale_min), RandomScale(args.ar_scale_min), np.eye(3))

    ### rotate object
    Ifg_rot1, Mfg_rot1, Rrot1 = Rotate(Ifg_as1, Mfg_as1, RandomAngle(args.angle_min, args.angle_max), Ras1)

    ### crop the object
    Ifg_aug1, Mfg_aug1, Rcrop1 = cropbb(Ifg_rot1, Mfg_rot1, Rrot1)

    ### resize the object
    Ifg_resize1, Mfg_resize1, Rresize1 = resize_bbox_bg(Ifg_aug1, Mfg_aug1, bbox1[2] - bbox1[0], bbox1[3] - bbox1[1], Rcrop1, ratio_list, ratio_prob)

    ## object 2
    ### change aspect ratio
    Ifg_as2, Mfg_as2, Ras2 = AspectRatio(Ifg, obj_mask2, RandomScale(args.ar_scale_min), RandomScale(args.ar_scale_min), np.eye(3))

    ### rotate object
    Ifg_rot2, Mfg_rot2, Rrot2 = Rotate(Ifg_as2, Mfg_as2, RandomAngle(args.angle_min, args.angle_max), Ras2)

    ### crop the object
    Ifg_aug2, Mfg_aug2, Rcrop2 = cropbb(Ifg_rot2, Mfg_rot2, Rrot2)

    ### resize the object
    Ifg_resize2, Mfg_resize2, Rresize2 = resize_bbox_bg(Ifg_aug2, Mfg_aug2, bbox2[2] - bbox2[0], bbox2[3] - bbox2[1], Rcrop2, ratio_list, ratio_prob)


    ## Blending
    Ibg_blend, Mbg1, Mbg2, Rpe1, Rpe2 = poisson_editing_2obj(np.array(Ifg_resize1), np.array(Ifg_resize2), np.array(Mfg_resize1), np.array(Mfg_resize2), bbox1, bbox2, np.array(Ibg), args.img_size, Rresize1, Rresize2)

    x1_obj1, y1_obj1 = trans_pos(grid_x, grid_y, Rpe1, args.img_size)
    x1_obj1, y1_obj1 = x1_obj1.reshape((1, int(nb_feat), int(nb_feat))), y1_obj1.reshape((1, int(nb_feat), int(nb_feat)))
    x2_obj1, y2_obj1 = trans_pos(grid_x, grid_y, np.linalg.inv(Rpe1), args.img_size) 
    x2_obj1, y2_obj1 = x2_obj1.reshape((1, int(nb_feat), int(nb_feat))), y2_obj1.reshape((1, int(nb_feat), int(nb_feat)))

    Mfg1, Mbg1 = consistent_dilate_mask(Mfg1, x1_obj1, y1_obj1, x2_obj1, y2_obj1)

    x1_obj2, y1_obj2 = trans_pos(grid_x, grid_y, Rpe2, args.img_size)
    x1_obj2, y1_obj2 = x1_obj2.reshape((1, int(nb_feat), int(nb_feat))), y1_obj2.reshape((1, int(nb_feat), int(nb_feat)))
    x2_obj2, y2_obj2 = trans_pos(grid_x, grid_y, np.linalg.inv(Rpe2), args.img_size) 
    x2_obj2, y2_obj2 = x2_obj2.reshape((1, int(nb_feat), int(nb_feat))), y2_obj2.reshape((1, int(nb_feat), int(nb_feat)))

    Mfg1, Mbg1 = consistent_dilate_mask(Mfg1, x1_obj1, y1_obj1, x2_obj1, y2_obj1)
    Mfg2, Mbg2 = consistent_dilate_mask(Mfg2, x1_obj2, y1_obj2, x2_obj2, y2_obj2)

    # if two mask touched, drop the images
    if (Mfg1 * Mfg2).sum() > 0 or  (Mbg1 * Mbg2).sum() > 0 :
        continue

    ## if small objs, drop as well
    if Mfg1.sum() < 9 or Mfg2.sum() < 9 or Mbg1.sum() < 9 or Mbg2.sum() < 9: 
        continue


    x1 = x1_obj1 * (Mfg1 > 0) + x1_obj2 * (Mfg2 > 0)
    y1 = y1_obj1 * (Mfg1 > 0.5) + y1_obj2 * (Mfg2 > 0.5) 

    x2 = x2_obj1 * (Mbg1 > 0.5)  + x2_obj2 * (Mbg2 > 0.5)
    y2 = y2_obj1 * (Mbg1 > 0.5) + y2_obj2 * (Mbg2 > 0.5)

    Mfg = Mfg1 + Mfg2
    Mbg = Mbg1 + Mbg2
    
    Ifg_style = style_transfer_pil(Ifg, style_list, args.style_dir, vgg, decoder, device)
    Ibg_style = style_transfer_pil(Ibg, style_list, args.style_dir, vgg, decoder, device)
    

    out1 = os.path.join(args.out_dir, '{:d}_a.jpg'.format(count))
    out2 = os.path.join(args.out_dir, '{:d}_b.jpg'.format(count))

    out3 = os.path.join(args.out_dir, '{:d}_as.jpg'.format(count))
    out4 = os.path.join(args.out_dir, '{:d}_bs.jpg'.format(count))


    Ifg.save(out1)
    Ibg_blend.save(out2)
    Ifg_style.save(out3)
    Ibg_style.save(out4)

    out1 = os.path.join(args.out_dir, '{:d}_a.npy'.format(count))
    out2 = os.path.join(args.out_dir, '{:d}_b.npy'.format(count))

    np.save(out1, np.concatenate((x1, y1, np.expand_dims(Mfg, axis=0)), axis=0).astype(np.float32))
    np.save(out2, np.concatenate((x2, y2, np.expand_dims(Mbg, axis=0)), axis=0).astype(np.float32))
    
    count += 1



        
        
