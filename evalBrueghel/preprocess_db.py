'''
Given a database image, this script will 
    1. resize the image (keeping aspect ratio) such that the max dimension is defined by img_size
    2. crop the image to make it square, the border will be padded if necessary
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

def preprocess_db(pil_db, img_size) : 
    pil_db_resize = pil_db.resize((img_size, img_size), resample=2)
    return pil_db_resize


if __name__ == "__main__":
    import os
    
    label_input = '../data/BrueghelImg/brueghelValCrop20.json'
    img_dir = '../data/BrueghelImg/BrueghelCrop20/'
    img_size = 480
    
    with open(label_input, 'r') as f :
        label = json.load(f)
    
    count = 0
    for cate_id in label : 
        for qry_id in range(len(label[cate_id])) : 
            qry_name = label[cate_id][qry_id]['query'][0]
            qry_bbox = label[cate_id][qry_id]['query'][1]
            
            qry = Image.open(os.path.join(img_dir, qry_name)).convert('RGB')

            pil_db_pad = preprocess_db(qry, img_size)
            out = 'toto{:d}.jpg'.format(count)
            pil_db_pad.save(out)
            count += 1
           