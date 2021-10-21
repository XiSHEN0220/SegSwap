# coding=utf-8
import torch 
import torchvision.transforms as transforms
from tqdm import tqdm 

from scipy import ndimage
from PIL import ImageDraw, Image

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm 
import os
import torch.nn.functional as F
import torchvision.models as models
from itertools import combinations

import json 

class ImgListLoader(Dataset):

    def __init__(self, img_dir, img_list, img_size):
        
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        
        self.transform = transforms.Compose([
                          transforms.Resize((img_size[1], img_size[0])), # (height, width)
                          transforms.ToTensor(),
                          transforms.Normalize(
                                       mean= [-m/s for m, s in zip(norm_mean, norm_std)],
                                       std= [1/s for s in norm_std])
                          ])
        self.img_dir = img_dir
        self.img_list = img_list
        self.nb_img = len(img_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img = os.path.join(self.img_dir, img_name)
        I = Image.open(img).convert('RGB')
        I_f = I.transpose(method=Image.FLIP_LEFT_RIGHT)
        I = self.transform(I)
        I_f = self.transform(I_f)
        
        return I, I_f, idx
     
    def __len__(self):
        return self.nb_img
    
## Data loader
def ImgLoader(img_dir, batch_size, img_size, img_list):

    dataSet = ImgListLoader(img_dir, img_list, img_size)
    dataLoader = DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=False, num_workers=1, drop_last = False)

    return dataLoader

    

def score_local_feat_match(feat1, x1, y1, feat2, x2, y2, weight_feat) :
    with torch.no_grad() : 
        feat_2_bag = [ feat2[:, y2[i], x2[i]].unsqueeze(0) for i in range(len(x2))]
        feat_2_bag = torch.cat(feat_2_bag, dim=0)
        
        feat_1_bag = [ feat1[:, y1[i], x1[i]].unsqueeze(0) for i in range(len(x1))]
        feat_1_bag = torch.cat(feat_1_bag, dim=0)
        
        ## each local feature is matched to  its feature in an other image, finally the similarity is weighted by the mask prediction 
        score_weight_feat = torch.sum(feat_1_bag * feat_2_bag, dim=1) * weight_feat
        
        ## if (x1i, y1i) --> (x2, y2)
        ## and (x1j, y1j) --> (x2, y2)
        ## pick the best match
        
        dict_score = {(y2[i], x2[i]): [0] for i in range(len(x2))}
        for i in range(len(x2)) : 
            dict_score[(y2[i], x2[i])].append(score_weight_feat[i].item())
        
        return  sum([np.max(dict_score[key]) for key in dict_score])
    
def consistent_mask(o1, o2) : 
    
    o1_tensor, o2_tensor = torch.from_numpy( o1 ), torch.from_numpy( o2 )
    flow12 = (o1_tensor[:, :2].permute(0, 2, 3, 1) - 0.5) * 2 
    flow21 = (o2_tensor[:, :2].permute(0, 2, 3, 1) - 0.5) * 2 
    
    m1, m2 = o1_tensor[:, 2:], o2_tensor[:, 2:]
    
    m1_cycle = F.grid_sample(m2, flow12, mode='bilinear') * m1
    m2_cycle = F.grid_sample(m1, flow21, mode='bilinear') * m2
    
    return m1_cycle, m2_cycle


class PairDiscovery(object):
    def __init__(self,
                 db_dir,
                 feat_dim = 256,
                 img_size = [480, 480],
                 corr_dir = None,
                 useFlip = False):
        
        self.db_dir = db_dir 
        self.img_list = sorted(os.listdir(db_dir))
        self.img_list = [img for img in self.img_list if '.jpg' in img or '.png' in img]
        
        self.nb_img = len(self.img_list)
        self.pair_idx = [[i,j] for i,j in combinations(range(self.nb_img), 2)]
        self.nb_pair = len(self.pair_idx)
        self.useFlip = useFlip
        
        print ('Number of pairs: {:d}'.format(len(self.pair_idx)))
        self.pair_idx = np.array(self.pair_idx)
        self.stride_net = 16 # stride of the backbone net
        self.batch_size = 32 # batch size for the test
        self.feat_dim = feat_dim
        
        self.corr_dir = corr_dir
        
        if self.corr_dir and not os.path.isdir(self.corr_dir):
            os.mkdir(self.corr_dir)
        
        ## random sample 500 rgba colors
        self.start_rgb = np.array([[220, 20, 60, 150]])
        self.end_rgb = np.array([[65,105,225, 150]])
        
        
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.img_size = (img_size[0], img_size[1]) # width = 480, height = 480
        self.nb_feat_w = self.img_size[0] // self.stride_net
        self.nb_feat_h = self.img_size[1] // self.stride_net
        
        mask = np.ones((self.nb_feat_h, self.nb_feat_w), dtype=bool)
        
        self.y_grid, self.x_grid = np.where(mask)
        self.transformINet = transforms.Compose([transforms.Resize((self.img_size[1], self.img_size[0])), # (height, width)
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
        
        
    def extract_db_feat(self, backbone) : 
        torch.cuda.empty_cache()
        backbone.eval()
        
        ## db features
        db_loader = ImgLoader(self.db_dir, self.batch_size, self.img_size, self.img_list)
        db_feat = torch.FloatTensor(self.nb_img, self.feat_dim, self.nb_feat_h, self.nb_feat_w)
        db_feat_flip = torch.FloatTensor(self.nb_img, self.feat_dim, self.nb_feat_h, self.nb_feat_w)

        for (batch, batch_flip, idx) in tqdm(db_loader) : 
            batch = batch.cuda()
            batch_flip = batch_flip.cuda()
            
            db_feat[idx] = backbone(batch).cpu()
            db_feat_flip[idx] = backbone(batch_flip).cpu()
            

        db_feat = F.normalize(db_feat, dim=1)
        db_feat_flip = F.normalize(db_feat_flip, dim=1)
        
        return db_feat, db_feat_flip
            
    def flip_batch(self, netEncoder, db_feat, db_feat_flip, idx1, idx2, isFlip) : 
        
        o1, o2 = netEncoder(db_feat[idx1].cuda(), db_feat[idx2].cuda())
        o1_flip, o2_flip = netEncoder(db_feat[idx1].cuda(), db_feat_flip[idx2].cuda())
        o2_update = []
        o1_update = []

        for i in range(len(o1)) : 
            if o1_flip[i, 2].sum().item() > o1[i, 2].sum().item() : 
                mask = torch.flipud(o2_flip[i, 2])
                flowx = 1 - o2_flip[i, 0]
                flowy = o2_flip[i, 1]
                o2_update.append(torch.stack([flowx, flowy, mask], dim=0))
                o1_update.append(o1_flip[i])
                isFlip.append(True)
            else : 
                o2_update.append(o2[i])
                o1_update.append(o1[i])
                isFlip.append(False)

        o1_update = torch.stack(o1_update, dim=0)
        o2_update = torch.stack(o2_update, dim=0)
        
        return o1_update, o2_update, isFlip
    
    def computePair(self, db_feat, db_feat_flip, netEncoder, start_idx=0, end_idx=1000):
        torch.cuda.empty_cache()
        netEncoder.eval()
        
        
        pair_idx = np.arange(start_idx, end_idx)
        output1 = np.zeros((end_idx - start_idx, 3, self.nb_feat_h, self.nb_feat_w), dtype=np.float32)
        output2 = np.zeros((end_idx - start_idx, 3, self.nb_feat_h, self.nb_feat_w), dtype=np.float32)
        
        nb_batch = (end_idx - start_idx) // self.batch_size
        last_batch = end_idx - start_idx - nb_batch * self.batch_size
        isFlip = []
        for i in tqdm(range(nb_batch)) : 
            start = i * self.batch_size + start_idx
            end = i * self.batch_size + start_idx + self.batch_size
            idx1 = self.pair_idx[start : end, 0]
            idx2 = self.pair_idx[start : end, 1]
            
            if self.useFlip :
                o1, o2, isFlip = self.flip_batch(netEncoder, db_feat, db_feat_flip, idx1, idx2, isFlip)
                        
            else : 
                o1, o2 = netEncoder(db_feat[idx1].cuda(), db_feat[idx2].cuda())
                
            output1[i * self.batch_size : (i+1) * self.batch_size] = o1.cpu().numpy()
            output2[i * self.batch_size : (i+1) * self.batch_size] = o2.cpu().numpy()
            
            
        if last_batch >  0 : 
            start = nb_batch * self.batch_size + start_idx
            end = end_idx
            idx1 = self.pair_idx[start : end, 0]
            idx2 = self.pair_idx[start : end, 1]
            
            if self.useFlip :
                o1, o2, isFlip = self.flip_batch(netEncoder, db_feat, db_feat_flip, idx1, idx2, isFlip)
                        
            else : 
                o1, o2 = netEncoder(db_feat[idx1].cuda(), db_feat[idx2].cuda())
                
            output1[nb_batch * self.batch_size : ] = o1.cpu().numpy()
            output2[nb_batch * self.batch_size : ] = o2.cpu().numpy()
        
        pair_idx, output1, output2, score = self.remove_mask0(db_feat, pair_idx, output1, output2)
        return pair_idx, output1, output2, score
    
    def remove_mask0(self, db_feat, pair_idx, output1, output2) : 
        nb_pair = len(pair_idx)
        score = np.zeros((nb_pair, 2))
        
        m1_cycle, m2_cycle = consistent_mask(output1, output2)
        for i in tqdm(range(nb_pair)) : 
            
            idx1 = self.pair_idx[pair_idx[i], 0]
            idx2 = self.pair_idx[pair_idx[i], 1]
            
            x2_pred =np.round(output1[i, 0, self.y_grid, self.x_grid] * (self.nb_feat_w - 1)).astype(int)
            y2_pred =np.round(output1[i, 1, self.y_grid, self.x_grid] * (self.nb_feat_h - 1)).astype(int)

            x1_pred =np.round(output2[i, 0, self.y_grid, self.x_grid] * (self.nb_feat_w - 1)).astype(int)
            y1_pred =np.round(output2[i, 1, self.y_grid, self.x_grid] * (self.nb_feat_h - 1)).astype(int)
            mask1 = m1_cycle[i, 0, self.y_grid, self.x_grid]
            mask2 = m2_cycle[i, 0, self.y_grid, self.x_grid]
            
            if mask1.sum() < (mask1.shape[0] * 0.01) or mask2.sum() < (mask2.shape[0] * 0.01) :
                continue
            score12 = score_local_feat_match(db_feat[idx1], self.x_grid, self.y_grid, db_feat[idx2], x2_pred, y2_pred, mask1)
            score21 = score_local_feat_match(db_feat[idx2], self.x_grid, self.y_grid, db_feat[idx1], x1_pred, y1_pred, mask2)
            score[i, 0] = score12
            score[i, 1] = score21
        
        matchable = score.max(axis=1) > (mask1.shape[0]  * 0.01)
        
        print ('Mask Non Zeros Pairs: {:d} --> {:d}'.format(nb_pair, np.sum(matchable)))
        return pair_idx[matchable], output1[matchable], output2[matchable], score[matchable]
    
    def main(self, backbone, netEncoder, start_idx, end_idx) : 
        
        ## db features
        db_feat, db_feat_flip = self.extract_db_feat(backbone)

        start = start_idx
        end = min(start + 10000, end_idx)
        while True : 
            pair_idx, output1, output2, score = self.computePair(db_feat, db_feat_flip, netEncoder, start, end)
            dict_out = {'pair_idx' : pair_idx, 'output1' : output1, 'output2' : output2, 'score' : score}
            out_pth = os.path.join(self.corr_dir, 'start{:d}_end{:d}.npy'.format(start, end))
            np.save(out_pth, dict_out)
            start = end
            end = min(start + 10000, end_idx)
            if end - start == 0 :
                break

if __name__ == "__main__":
    import argparse 
    import sys 
    sys.path.append('../model')
    import transformer # model
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Evaluate cross-transformer on Internet dataset')


    parser.add_argument('--gpu', type=str, default='1', help='gpu devices')
    ## input / output dir 
    parser.add_argument('--corr-dir', type=str, default=None, help='output dir')
    
    
    parser.add_argument('--resume-pth', type = str, default='../model/hard_mining_neg5.pth', help='resume path')
    ## paramter transformer
    
    
    parser.add_argument('--mode', type=str, choices=['tiny', 'small', 'base', 'large'], default='small', help='different size of transformer encoder')
    parser.add_argument('--pos-weight', type=float, default=0.1, help='weight for positional encoding')
    parser.add_argument('--feat-weight', type=float, default=1, help='weight for feature component')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout in the transformer layer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='activation in the transformer layer')
    
    parser.add_argument('--layer-type', type=str, nargs='+', default=['I', 'C', 'I', 'C', 'I', 'N'], help='which type of layers: I is for inner image attention, C is for cross image attention, N is None')
    
    parser.add_argument('--drop-feat', type=float, default=0.1, help='drop feature rate')
    
    
    parser.add_argument('--db-dir', type=str, default='../data/Airplane100/', help='db directory')
    
    parser.add_argument('--start-idx', type=int, default=0, help='db directory, start cls index')
    
    parser.add_argument('--end-idx', type=int, default=4950, help='db directory, end cls index')
    
    parser.add_argument('--img-size', type=int, nargs='+', default=[480, 480], help='image size')
    
    
    parser.add_argument('--useFlip', action='store_true', help='whether use flipped image')
    args = parser.parse_args()
    print (args)

    
    ## set gpu

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    ## feature net
    print ('Load MocoV2 pre-trained ResNet-50 feature...')
    resume_path = '../model/moco_v2_800ep_pretrain_torchvision.pth.tar'
    param = torch.load(resume_path)['model']
    new_param = {}
    for key in param.keys() : 
        if 'fc'  in key : 
            continue
        new_param[key] = param[key]

    backbone = models.resnet50(pretrained=False)   
    backbone.load_state_dict(new_param, strict=False)
    resnet_feature_layers = ['conv1','bn1','relu','maxpool','layer1','layer2','layer3']
    resnet_module_list = [getattr(backbone,l) for l in resnet_feature_layers]
    last_layer_idx = resnet_feature_layers.index('layer3')
    backbone = torch.nn.Sequential(*resnet_module_list[:last_layer_idx+1])
    feat_dim=1024

    backbone.cuda()

    ## model
    netEncoder = transformer.TransEncoder(feat_dim,
                                          pos_weight = args.pos_weight,
                                          feat_weight = args.feat_weight,
                                          dropout= args.dropout,
                                          activation=args.activation,
                                          mode=args.mode,
                                          layer_type = args.layer_type,
                                          drop_feat = args.drop_feat) 

    netEncoder.cuda()


    ## resume
    if args.resume_pth : 
        param = torch.load(args.resume_pth)
        backbone.load_state_dict(param['backbone'])
        netEncoder.load_state_dict(param['encoder'])

        print ('Loading net weight from {}'.format(args.resume_pth))
    
    PairCompute = PairDiscovery(db_dir = args.db_dir,
                             feat_dim = feat_dim,
                             img_size = args.img_size,
                             corr_dir = args.corr_dir,
                             useFlip = args.useFlip)
    with torch.no_grad() : 
        PairCompute.main(backbone, netEncoder, args.start_idx, args.end_idx)
    
    

    

                



