import sys
sys.path.append('AdaIn/')
import net, function
import torch
import torch.nn as nn
from torchvision import transforms
import PIL.Image  as Image 
def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0, interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = function.adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = function.adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def style_transfer_pil_input(vgg, decoder, device, pil_content, pil_style, alpha = 0.5) : 
    
    
    content_size = 512
    style_size = 512
    crop = False
    
    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    
    content = content_tf(pil_content)
    style = style_tf(pil_style)

    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, alpha)
    output = output.cpu()
    output = torch.clamp(output, min=0, max=1)
    return transforms.ToPILImage()(output[0])