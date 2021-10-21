import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import PIL.Image as Image
import os 
import numpy as np 
import torch.nn.functional as F
import kornia

from itertools import combinations

from datetime import datetime
import warnings
warnings.filterwarnings("ignore")



def LoadImg(path) :
    return Image.open(path).convert('RGB')

# 
class ImageFolderNegPool(Dataset):

    def __init__(self, img_dir_list, prob_style, prob_dir, neg_pool_list, transform):
        
        self.img_dir_list = img_dir_list
        
        self.transform = transform
        self.prob_style = prob_style
        self.prob_dir = prob_dir
        
        self.neg_pool_list = neg_pool_list
      
    def __getitem__(self, i):
        
        img_dir_idx = torch.randint(high=len(self.img_dir_list), size=(1,)).item()
        img_dir = self.img_dir_list[img_dir_idx]
        
        idx = self.neg_pool_list[i]
        if torch.rand(1).item() < self.prob_style : 
            if torch.rand(1).item() < 0.5 : 
                pth = os.path.join(img_dir, '{:d}_as.jpg'.format(idx))
            else :
                pth = os.path.join(img_dir, '{:d}_bs.jpg'.format(idx))

        else :
            if torch.rand(1).item() < 0.5 : 
                pth = os.path.join(img_dir, '{:d}_a.jpg'.format(idx))
            else :
                pth = os.path.join(img_dir, '{:d}_b.jpg'.format(idx))

        I = LoadImg( pth )
        T = self.transform(I)
            
        return T 

    def __len__(self):
        return len(self.neg_pool_list)

## Train Data loader
def NegPoolDataLoader(img_dir_list, prob_style, prob_dir, neg_pool_list, transform):
    NegPoolSet = ImageFolderNegPool(img_dir_list, prob_style, prob_dir, neg_pool_list, transform)
    NegPoolLoader = DataLoader(dataset=NegPoolSet, batch_size=64, shuffle=True, num_workers=4, drop_last = False)

    return NegPoolLoader



def getNegPoolDataloader(img_dir_list, prob_style, prob_dir, neg_pool_list) : 
    
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([transforms.RandomResizedCrop(size=480),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    
    
    negLoader = NegPoolDataLoader(img_dir_list, prob_style, prob_dir, neg_pool_list, transformINet)
    
    return  negLoader


class NegaINetGenerator():

    def __init__(self, img_dir_list, prob_style, prob_dir, batch_hard_neg, neg_pool_size, logger):
        
        
        self.img_dir_list = img_dir_list
        
        self.prob_style = prob_style
        self.prob_dir = prob_dir
        
        self.nb_pair =  len(os.listdir(img_dir_list[0])) // 8
        
        self.neg_pool_size = neg_pool_size
        self.idx = torch.from_numpy(np.array([[i,j] for i,j in combinations(range(self.neg_pool_size), 2)], dtype=np.int64))
        self.nb_nega_pair = len(self.idx)
        self.batch_size = 32
        self.batch_hard_neg = batch_hard_neg
        
        self.nb_batch = self.nb_nega_pair // self.batch_size
        self.last_batch = self.nb_nega_pair - self.nb_batch * self.batch_size
        self.logger = logger
       
    def update_negasample(self, backbone) :
        
        self.nega_pool = (torch.randperm(self.nb_pair)[:self.neg_pool_size]).numpy().tolist()
        
        negLoader = getNegPoolDataloader(self.img_dir_list, self.prob_style, self.prob_dir, self.nega_pool)

        with torch.no_grad() : 
            train_sample = []
            for data in negLoader : 
                feat = backbone.eval()(data.cuda())
                train_sample.append(feat)
        self.train_sample = F.normalize(torch.cat(train_sample, dim=0), dim=1)
        
        self.train_sample.cpu()
        torch.cuda.empty_cache()
        
    def get_maskpred(self, encoder) : 
        with torch.no_grad() : 
            mask_pred = []
            
            for i in range(self.nb_batch) : 
                idx = self.idx.narrow(0, self.batch_size * i, self.batch_size)
                tmp1, tmp2 = encoder.eval()(self.train_sample[idx[:, 0]].cuda(), self.train_sample[idx[:, 1]].cuda())
                tmp1 = tmp1.narrow(1, 2, 1).contiguous().view(self.batch_size, -1).mean(dim=1)
                tmp2 = tmp2.narrow(1, 2, 1).contiguous().view(self.batch_size, -1).mean(dim=1)
                tmp = tmp1 + tmp2
                mask_pred.append(tmp)
                if i % 100 == 99: 
                    print (' {} \t {:d} / {:d}'.format(datetime.now().time(), i, self.nb_batch))
            
            idx = self.idx.narrow(0, self.batch_size * self.nb_batch, self.last_batch)
            tmp1, tmp2 = encoder.eval()(self.train_sample[idx[:, 0]], self.train_sample[idx[:, 1]])
            tmp1 = tmp1.narrow(1, 2, 1).contiguous().view(self.last_batch, -1).mean(dim=1)
            tmp2 = tmp2.narrow(1, 2, 1).contiguous().view(self.last_batch, -1).mean(dim=1)
            tmp = tmp1 + tmp2
            mask_pred.append(tmp)
            
            self.mask_pred = torch.cat(mask_pred, dim=0)
            self.mask_pred = self.mask_pred.cpu().numpy()
            self.mask_pred = self.mask_pred * (self.mask_pred > 0.04)
            torch.cuda.empty_cache()
            
    def update(self, backbone, encoder) : 
        while True : 
            msg = 'Sample {:d} images in the training data and extract their features...'.format(self.neg_pool_size)
            self.logger.info(msg)
            
            
            self.update_negasample(backbone)
            
            msg = 'Compute their mask predictions...'
            self.logger.info(msg)
            self.get_maskpred(encoder)
            
            nb_hard_neg = (self.mask_pred > 0.1).sum()
            msg = 'Nb of hard examples {:d} (mask pred > 0.1)...'.format(nb_hard_neg)
            self.logger.info(msg)
            
            self.nb_iter = min(nb_hard_neg, 100)
            self.prob_sample = self.mask_pred / self.mask_pred.sum()
            if self.nb_iter > 10 :
                break
    def get_neg_pool_list(self) : 
        return self.nega_pool
    
    def get_pair_batch(self):
        idx = np.random.choice(self.nb_nega_pair, self.batch_hard_neg, replace=False, p=self.prob_sample)
        
        idx = self.idx[idx]
        return {'T1' : self.train_sample[idx[:, 0]],  'T2' : self.train_sample[idx[:, 1]]}


class ImageFolderTrain(Dataset):

    def __init__(self, img_dir_list, transform, prob_style, prob_dir, tps_grid, neg_pool_list):
        ### each image directory should contain the same number of pairs
        
        self.img_dir_list = img_dir_list
        self.nb_pair = len(os.listdir(img_dir_list[0])) // 8
        self.all_img = [i for i in range(self.nb_pair)]
        self.train_list = list(set(self.all_img) - set(neg_pool_list))
        self.transform = transform
        self.prob_style = prob_style
        self.prob_dir = prob_dir
        
        self.stride_net = 16
        self.nb_feat_w = 480 // self.stride_net
        self.nb_feat_h = 480 // self.stride_net
        
        with torch.no_grad() : 
            self.grid_list = tps_grid
            self.tps_src_pts()

            ## grid list for tps transformation
            ## generate grid for warping
            x = ( np.arange(self.nb_feat_w) + 0.5 ) / self.nb_feat_w
            y = ( np.arange(self.nb_feat_h) + 0.5 ) / self.nb_feat_h
            grid_x, grid_y = np.meshgrid(x, y)
            grid_x, grid_y = grid_x.astype(np.float32), grid_y.astype(np.float32)

            grid_x_tensor = (torch.from_numpy(grid_x).unsqueeze(0).unsqueeze(0) - 0.5) * 2
            grid_y_tensor = (torch.from_numpy(grid_y).unsqueeze(0).unsqueeze(0) - 0.5) * 2

            grid_warp = torch.cat([grid_x_tensor, grid_y_tensor], dim=1)
            self.grid_warp =grid_warp.resize(1, 2, self.nb_feat_w * self.nb_feat_h).permute(0, 2, 1)
        
    
    def __getitem__(self, i):
        
        img_dir_idx = torch.randint(high=len(self.img_dir_list), size=(1,)).item()
        img_dir = self.img_dir_list[img_dir_idx]
        
        idx = self.train_list[i]
        if torch.rand(1).item() < self.prob_style : 
            pth1 = os.path.join(img_dir, '{:d}_as.jpg'.format(idx))

        else :
            pth1 = os.path.join(img_dir, '{:d}_a.jpg'.format(idx))


        if torch.rand(1).item() < self.prob_style : 
            pth2 = os.path.join(img_dir, '{:d}_bs.jpg'.format(idx))

        else :
            pth2 = os.path.join(img_dir, '{:d}_b.jpg'.format(idx))
        
        I1 = LoadImg(pth1)
        I2 = LoadImg(pth2)
        
        M1 = np.load(os.path.join(img_dir, '{:d}_a.npy'.format(idx))).astype(np.float32)
        M2 = np.load(os.path.join(img_dir, '{:d}_b.npy'.format(idx))).astype(np.float32)
        
        mask1 = M1[2]
        mask2 = M2[2]
        
        xy1 = M1[:2]
        xy2 = M2[:2]
        
        
        
        #Tn = self.transform(In)
        xy1 = torch.from_numpy( xy1 ) # 2 * 30 * 30
        xy2 = torch.from_numpy( xy2 ) # 2 * 30 * 30
        
        mask1 = torch.from_numpy( mask1 ).unsqueeze(0) # 1 * 30 * 30
        mask2 = torch.from_numpy( mask2 ).unsqueeze(0) # 1 * 30 * 30
        if torch.rand(1).item() > 0.5: 
            with torch.no_grad() : 
                I2 = transforms.ToTensor()(I2).unsqueeze(0)
                mask2 = mask2.unsqueeze(0)
                I2, mask2, flow_bg2tps, flow_tps2bg = self.tps_trans(I2, mask2) ## note that here the flow are from [-1, 1]

                flow12 = xy1.unsqueeze(0)
                flow12 = (flow12 - 0.5) * 2

                flow21 = xy2.unsqueeze(0)
                flow21 = (flow21 - 0.5) * 2

                flowtps_1 = F.grid_sample(flow21, flow_tps2bg.permute(0,2,3,1))
                flow1_tps = F.grid_sample(flow_bg2tps, flow12.permute(0,2,3,1))

                flowtps_1 = (flowtps_1 + 1) / 2 # renormalise to [0, 1]
                flow1_tps = (flow1_tps + 1) / 2 # renormalise to [0, 1]

                xy1 = flow1_tps[0] # 2 * 30 * 30
                xy2 = flowtps_1[0] # 2 * 30 * 30

            
            
            
        T1 = self.transform(I1)
        T2 = self.transform(I2)
        
        
        random_mask2 = torch.BoolTensor(mask2.size()).fill_(False) 
        random_mask1 = torch.BoolTensor(mask1.size()).fill_(False) 
        
        if  torch.rand(1).item() > 0.5: 
            random_mask2 = mask2 < 0.5
        
        mask1 = mask1.type(torch.FloatTensor)
        mask2 = mask2.type(torch.FloatTensor)
        
            
        return {'T1' : T1,
                'T2' : T2,
                'RM1' :random_mask1,
                'RM2' :random_mask2,
                'M1' : mask1,
                'M2' : mask2,
                'xy1' : xy1,
                'xy2' : xy2
                
                }
                

    def __len__(self):
        return len(self.train_list)
    
    def tps_src_pts(self) : 
        
        self.src_dict = {}
        for grid_size in self.grid_list: 
            axis_coords = np.linspace(-1, 1, grid_size)
            grid_Y, grid_X = np.meshgrid(axis_coords, axis_coords)
            grid_Y, grid_X = torch.from_numpy(grid_Y.reshape(-1).astype(np.float32)), torch.from_numpy(grid_X.reshape(-1).astype(np.float32))
            grid_Y, grid_X = grid_Y.unsqueeze(0).unsqueeze(2), grid_X.unsqueeze(0).unsqueeze(2)

            points_src = torch.cat([grid_X, grid_Y], dim=2)
            self.src_dict[grid_size] = points_src
        
    
    def tps_trans(self, Ibg, mask) : 
        
        grid_size = np.random.choice(self.grid_list)
        points_src = self.src_dict[grid_size]
        move = (torch.rand(points_src.size()) - 0.5) * 1 * 1 / grid_size
        points_dst = torch.clamp(points_src + move, min=-1, max=1)

        # note that we are getting the transform: src -> dst
        kernel_weights, affine_weights = kornia.get_tps_transform(points_src, points_dst)
        kernel_weights_inv, affine_weights_inv = kornia.get_tps_transform(points_dst, points_src)


        Ibg_warp = kornia.warp_image_tps(Ibg, points_dst, kernel_weights, affine_weights)
        mask_warp = kornia.warp_image_tps(mask, points_dst, kernel_weights, affine_weights)

        flow_bg2tps = kornia.warp_points_tps(self.grid_warp, points_src, kernel_weights_inv, affine_weights_inv)
        flow_bg2tps = flow_bg2tps.permute(0, 2, 1).resize(1, 2, self.nb_feat_h, self.nb_feat_w)

        flow_tps2bg = kornia.warp_points_tps(self.grid_warp, points_dst, kernel_weights, affine_weights)
        flow_tps2bg = flow_tps2bg.permute(0, 2, 1).resize(1, 2, self.nb_feat_h, self.nb_feat_w)

        return transforms.ToPILImage()(Ibg_warp[0]), mask_warp[0], flow_bg2tps, flow_tps2bg
    

    

## Train Data loader
def TrainDataLoader(img_dir_list, transform, batch_size, prob_style, prob_dir, tps_grid, neg_pool_list):

    trainSet = ImageFolderTrain(img_dir_list, transform, prob_style, prob_dir, tps_grid, neg_pool_list)
    trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=4, drop_last = True)

    return trainLoader



def getDataloader(train_dir_list, batch_size, prob_style, prob_dir, tps_grid, neg_pool_list) : 
    
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    
    
    trainLoader = TrainDataLoader(train_dir_list, transformINet, batch_size, prob_style, prob_dir, tps_grid, neg_pool_list)
    
    return  trainLoader

