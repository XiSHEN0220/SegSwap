import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision
import PIL.Image as Image
import os 
import numpy as np 
import torch.nn.functional as F
import kornia

import warnings
warnings.filterwarnings("ignore")



def LoadImg(path) :
    return Image.open(path).convert('RGB')




class ImageFolderTrain(Dataset):

    def __init__(self, img_dir_list, transform, prob_style, prob_dir, tps_grid, img_size=(480, 480)):
        ### each image directory should contain the same number of pairs
        
        self.img_dir_list = img_dir_list
        
        self.load_pair = self.load_img_random_style
        self.nb_pair = 100000
        self.transform = transform
        self.prob_style = prob_style
        self.prob_dir = prob_dir
        
        self.stride_net = 16
        self.nb_feat_w = img_size[0] // self.stride_net
        self.nb_feat_h = img_size[1] // self.stride_net
        
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
    
    def load_img_random_style(self, img_dir, idx) :
        
        if torch.rand(1).item() < self.prob_style : 
            pth1 = os.path.join(img_dir, '{:d}_as.jpg'.format(idx))

        else :
            pth1 = os.path.join(img_dir, '{:d}_a.jpg'.format(idx))


        if torch.rand(1).item() < self.prob_style : 
            pth2 = os.path.join(img_dir, '{:d}_bs.jpg'.format(idx))

        else :
            pth2 = os.path.join(img_dir, '{:d}_b.jpg'.format(idx))
        
        return pth1, pth2
    
    def __getitem__(self, idx):
        
        img_dir_idx = torch.randint(high=len(self.img_dir_list), size=(1,)).item()
        img_dir = self.img_dir_list[img_dir_idx]
        
        ## load image
        
        pth1, pth2 = self.load_pair(img_dir, idx)
         
        
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
        return self.nb_pair
    
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
    
class ImageFolderValJpg(Dataset):

    def __init__(self, img_dir_list, transform):
        self.img_dir_list = img_dir_list
        self.nb_pair = len(os.listdir(img_dir_list[0])) // 8
        
        self.transform = transform
        
        
    def __getitem__(self, idx):

        img_dir = self.img_dir_list[idx % len(self.img_dir_list)]
        
        if idx % 4 == 0 :
            I1 = LoadImg(os.path.join(img_dir, '{:d}_a.jpg'.format(idx)))
            I2 = LoadImg(os.path.join(img_dir, '{:d}_b.jpg'.format(idx)))
            
        if idx % 4 == 1 :
            I1 = LoadImg(os.path.join(img_dir, '{:d}_as.jpg'.format(idx)))
            I2 = LoadImg(os.path.join(img_dir, '{:d}_b.jpg'.format(idx)))
            
        if idx % 4 == 2 :
            I1 = LoadImg(os.path.join(img_dir, '{:d}_a.jpg'.format(idx)))
            I2 = LoadImg(os.path.join(img_dir, '{:d}_bs.jpg'.format(idx)))
        if idx % 4 == 3 :
            I1 = LoadImg(os.path.join(img_dir, '{:d}_as.jpg'.format(idx)))
            I2 = LoadImg(os.path.join(img_dir, '{:d}_bs.jpg'.format(idx)))
        
        
        M1 = np.load(os.path.join(img_dir, '{:d}_a.npy'.format(idx))).astype(np.float32)[2]
        M2 = np.load(os.path.join(img_dir, '{:d}_b.npy'.format(idx))).astype(np.float32)[2]
        
        if idx % 2 == 1 : 
            I1, I2 = I2, I1
            M1, M2 = M2, M1
        
        
        T1 = self.transform(I1)
        T2 = self.transform(I2)
        
        M1 = torch.from_numpy( M1 ).unsqueeze(0)
        M2 = torch.from_numpy( M2 ).unsqueeze(0)
        
        
        
        return {'T1' : T1,
                'T2' : T2,
                'M1' : M1.type(torch.FloatTensor),
                'M2' : M2.type(torch.FloatTensor)
                }
                

    def __len__(self):
        return self.nb_pair
    

## Train Data loader
def TrainDataLoader(img_dir_list, transform, batch_size, prob_style, prob_dir, tps_grid,  img_size=(480, 480)):

    trainSet = ImageFolderTrain(img_dir_list, transform, prob_style, prob_dir, tps_grid, img_size)
    trainLoader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True, num_workers=4, drop_last = True)

    return trainLoader



def getDataloader(train_dir_list, batch_size, prob_style, prob_dir, tps_grid, img_size=(480, 480)) : 
    
    norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    transformINet = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    
    
    trainLoader = TrainDataLoader(train_dir_list, transformINet, batch_size, prob_style, prob_dir, tps_grid, img_size)
    
    return  trainLoader
