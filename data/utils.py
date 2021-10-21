from PIL import Image, ImageDraw
import numpy as np 
import os 
import json 
from scipy import ndimage

def mask_from_polygon(w, h, polygon) : 

    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img, dtype=bool)
    return mask 

def dilation(mask, nb_iter = 1) : 
    return ndimage.binary_dilation(mask, iterations = nb_iter).astype(mask.dtype)

def erosion(mask, nb_iter = 1) : 
    return ndimage.binary_erosion(mask, iterations = nb_iter).astype(mask.dtype)

def mask_from_polygon(w, h, polygon) : 

    img = Image.new('L', (w, h), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img, dtype=bool)
    return mask 

def merge_rgb_mask(rgb, mask) : 
    mask = np.expand_dims(mask, axis=2)
    rgba = np.concatenate([rgb, mask*255], axis=2)
    
    return Image.fromarray(rgba.astype(np.uint8))

def coco2017_id2name(image_id) : 
    image_name = str(image_id).zfill(12) + '.jpg'
    return image_name

def mkdir_directory(out_dir) : 
    if not os.path.exists(out_dir) : 
        os.mkdir(out_dir)
        
def bbox_from_binarymask(mask) : 
    idx1, idx2 = np.where(mask)
    
    start_h, end_h = idx1.min(), idx1.max() + 1
    start_w, end_w = idx2.min(), idx2.max() + 1
    return start_w, start_h, end_w, end_h

def load_json(json_pth) : 
    with open(json_pth, 'r') as f : 
        h = json.load(f)
    return h

def resize_mask(mask_binary, target_w, target_h) : 
    
    pil_mask = Image.fromarray(mask_binary.astype(np.uint8) * 255)
    pil_mask = pil_mask.resize((target_w, target_h), resample=2) # bilinear interpolation for the mask
    return pil_mask 