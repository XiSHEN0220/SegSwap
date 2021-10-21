'''
Given a query image with bbox, this script will 
    1. resize the query (keeping aspect ratio) such that the max dimension is defined by qry_size
    2. crop the query such that the annotated the bbox is in the center, the border will be padded 
'''
import json 
from PIL import ImageDraw, Image
import torchvision.transforms as transforms
import torch 


def drawRectanglePil(I, bbox, bboxColor=[0, 255 ,0], lineWidth = 3, alpha = 100):
    """draw a bounding box with bbox color
    """
    assert 'PIL' in str(type(I))
    
    Icopy = I.copy()
    draw = ImageDraw.Draw(Icopy)
    rgba = (bboxColor[0], bboxColor[1], bboxColor[2], alpha)
    draw.rectangle(bbox, outline=rgba)
    for i in range(1, lineWidth) :
        bboxIndex = [(bbox[0] + i, bbox[1] + i),(bbox[2] - i, bbox[3] - i)]
        draw.rectangle(bboxIndex, outline=rgba)

    return Icopy

def get_ratio_resize_max(I, bb, qry_max_size, crop_size, stride_net) : 
    
    bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
    
    ratio = max(bb_w / qry_max_size, bb_h / qry_max_size)
    new_bbw, new_bbh = max(int(round(bb_w / ratio / stride_net)), 1) * stride_net, max(int(round(bb_h / ratio / stride_net)), 1) * stride_net
    
    ratio_w, ratio_h = bb_w / new_bbw,  bb_h / new_bbh
    
    w, h = I.size
    new_w, new_h = int(round(w / ratio_w)), int(round(h / ratio_h))
    new_w, new_h = max(new_w, crop_size), max(new_h, crop_size) # after resize should make sure image is larger than crop_size
    ratio_w, ratio_h = w / new_w, h / new_h
    
    new_bb = [int(round(bb[0] / ratio_w)), int(round(bb[1] / ratio_h)), int(round(bb[2] / ratio_w)), int(round(bb[3] / ratio_h))]
    
    
    return I.resize((new_w, new_h)), new_bb
  
    
def crop(I, bb, crop_size) :
    bb_w, bb_h = bb[2] - bb[0], bb[3] - bb[1]
    
    
    margin_w = (crop_size - bb_w) // 2
    margin_h = (crop_size - bb_h) // 2
    
    large_bb = [bb[0] - margin_w, bb[1] - margin_h, bb[2] + margin_w, bb[3] + margin_h]
    if large_bb[2] > I.size[0] : 
        shift_w = I.size[0] - large_bb[2]
        
    elif large_bb[0] < 0 : 
        shift_w = - large_bb[0]
    else : 
        shift_w = 0
    
    if large_bb[3] > I.size[1] : 
        shift_h = I.size[1] - large_bb[3]
    elif large_bb[1] < 0 : 
        shift_h = - large_bb[1]
    else :
        shift_h = 0
    
    large_bb = [large_bb[0] + shift_w, large_bb[1] + shift_h, large_bb[2] + shift_w, large_bb[3] + shift_h]
    new_bb = [bb[0] - large_bb[0], bb[1] - large_bb[1], bb[2] - large_bb[0], bb[3] - large_bb[1]]
    
    return I.crop(large_bb), new_bb

def preprocess_query(pil_query_org, bbox_org, qry_size, img_size, stride_net) : 
    qry_resize, qry_bbox_resize = get_ratio_resize_max(pil_query_org, bbox_org, qry_size, img_size, stride_net)
    Icrop, qry_bbox_crop = crop(qry_resize, qry_bbox_resize, img_size)
    
    return Icrop, qry_bbox_crop

if __name__ == "__main__":
    import os
    
    label_input = '../data/BrueghelImg/brueghelValCrop20.json'
    img_dir = '../data/BrueghelImg/BrueghelCrop20/'
    img_size = 480
    qry_size = 64
    stride_net = 16
    
    with open(label_input, 'r') as f :
        label = json.load(f)
    
    count = 0
    for cate_id in label : 
        for qry_id in range(len(label[cate_id])) : 
            qry_name = label[cate_id][qry_id]['query'][0]
            qry_bbox = label[cate_id][qry_id]['query'][1]
            
            qry = Image.open(os.path.join(img_dir, qry_name)).convert('RGB')

            qry_crop, qry_bbox_crop = preprocess_query(qry, qry_bbox, qry_size, img_size, stride_net)
            out = 'toto{:d}.jpg'.format(count)
            drawRectanglePil(qry_crop, qry_bbox_crop).save(out)
            count += 1
    