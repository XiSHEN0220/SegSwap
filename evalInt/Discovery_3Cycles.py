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
import torch
from itertools import combinations
import compute_pair
import VisEigen
import cv2
import warnings
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, square
warnings.filterwarnings("ignore")

def consistent_mask(o1, o2) : 
    
    o1_tensor, o2_tensor = torch.from_numpy( o1 ), torch.from_numpy( o2 )
    flow12 = (o1_tensor[:, :2].permute(0, 2, 3, 1) - 0.5) * 2 
    flow21 = (o2_tensor[:, :2].permute(0, 2, 3, 1) - 0.5) * 2 
    
    m1, m2 = o1_tensor[:, 2:], o2_tensor[:, 2:]
    
    m1_transported = F.grid_sample(m1, flow21, mode='bilinear')
    m1_cycle = F.grid_sample(m1_transported * m2, flow12, mode='bilinear') * m1
    
    m2_cycle = F.grid_sample(m1_cycle, flow21, mode='bilinear')
    return m1_cycle, m2_cycle, flow21

def resize_img(I, max_size = 1000) : 
    w, h = I.size
    ratio = max(w / max_size, h / max_size)
    new_w, new_h = w / ratio, h / ratio
    return I.resize((int(new_w), int(new_h)))

def compute_acc_iou(pred, gt) : 
    '''
    0 : bg
    1 : fg
    '''
    h, w = gt.shape
    inter = (pred & gt).astype(np.float32)
    union = (pred | gt).astype(np.float32)
    
    acc = np.sum((pred == gt).astype(np.float32)) / h / w
    iou = inter.sum() / union.sum() if np.sum(gt.astype(np.float32)) > 0 else None
    return acc, iou

def interpolate(arr_pred, w, h) : 
    tensor_pred = torch.from_numpy(arr_pred).unsqueeze(0).unsqueeze(0)
    tensor_pred = F.interpolate(tensor_pred, size = (h, w), mode = 'bilinear')
    return tensor_pred.squeeze().numpy()

def soft2binarymask(pred, idx, db_dir, img_list, fg_th, dilate) : 
    
    
    
    img = cv2.imread(os.path.join(db_dir,img_list[idx]))
    h, w = img.shape[:2]
    pred= interpolate(pred, w, h)
    pred_max = pred.max()
    fg_th_img = fg_th * pred_max
    
    mask = np.zeros(pred.shape, dtype=np.uint8)
    
    if (pred > 0).sum() != 0 : 
        pred_one = (pred > 0).astype(np.float32)
        if dilate > 0 : 
            idx_neg = (dilation(pred_one, square(dilate)) - pred_one).astype(bool)
        else : 
            idx_neg = np.zeros(pred_one.shape).astype(bool)
        

        
        mask[pred > fg_th_img] = 1 
        mask[idx_neg] = 2 
        mask[(pred > 0) & (pred < fg_th_img)] = 3 


        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
    
        mask2, _, _ =  cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask2==2)|(mask2==0),0,1).astype('uint8')
        
    mask = mask == 1
    
    
    return mask

        
class Discovery(compute_pair.PairDiscovery):
    def __init__(self,
                 db_dir,
                 feat_dim = 256,
                 feat_size = (30, 30),
                 top_k = 3,
                 corr_dir = None):
        compute_pair.PairDiscovery.__init__(self, db_dir, feat_dim, corr_dir = corr_dir)
        self.keep_idx, output1, output2, self.score = self.load_pair_res()
        
        #print ('Nb pairs for discovery: {:d} ...'.format(len(self.keep_idx)))
        
        self.top_k = top_k
        self.feat_size = feat_size
        self.db_dir = db_dir
        
        self.out1_x = np.round(output1[:, 0] * (feat_size[0] - 1)).astype(np.int)
        self.out1_y = np.round(output1[:, 1] * (feat_size[1] - 1)).astype(np.int)
        self.out1_m = output1[:, 2]
        
        self.out2_x = np.round(output2[:, 0] * (feat_size[0] - 1)).astype(np.int)
        self.out2_y = np.round(output2[:, 1] * (feat_size[1] - 1)).astype(np.int)
        self.out2_m = output2[:, 2]
        
        
        x = np.arange(feat_size[0])
        y = np.arange(feat_size[1])
        self.grid_x, self.grid_y = np.meshgrid(x, y)
        
    
    def load_pair_res(self) : 
        pair_list  = sorted(os.listdir(self.corr_dir))
        
        pair_idx = []
        output1 = []
        output2 = []
        score = []
        
        for pair in pair_list : 
            pair_path = os.path.join(self.corr_dir, pair)
            pair_res = np.load(pair_path, allow_pickle=True)
            pair_res = pair_res.tolist()
            
            pair_idx.append( pair_res['pair_idx'] )
            output1.append( pair_res['output1'] )
            output2.append( pair_res['output2'] )
            score.append( pair_res['score'] )
            
        return np.concatenate(pair_idx, axis=0), np.concatenate(output1, axis=0), np.concatenate(output2, axis=0), np.concatenate(score, axis=0)
    
    def get_pair_score(self) : 
        
        self.score_pair = np.zeros((len(self.img_list), len(self.img_list)), dtype=np.float32)
        for i in range(len(self.keep_idx)) : 
            idx1, idx2 = self.pair_idx[self.keep_idx[i], 0], self.pair_idx[self.keep_idx[i], 1]
            score12, score21 = self.score[i]
            self.score_pair[idx1, idx2] = score12
            self.score_pair[idx2, idx1] = score21
            

                    
    def get_topk_pair(self) : 
        
        
        topk_pair = np.argsort(self.score_pair * -1, axis=1)[:, :self.top_k]
        topk_pair_list = []
        for i in range(len(topk_pair)) :
            tmp = [(i, topk_pair[i, j]) for j in range(self.top_k)]
            topk_pair_list = tmp + topk_pair_list
        self.topk_pair = topk_pair_list
        #print ('Nb of pairs : {:d}'.format(len(self.topk_pair)))
        
    def get_nodes(self, conf_th) :     
        start = 0
        pair_count = 0
        max_node = len(self.topk_pair) * int(self.nb_feat_w * self.nb_feat_h) 
        
        self.corr = np.zeros((max_node, 5), dtype=np.float32) # x1, y1, x2, y2
        self.img_idx_start_end = np.zeros((len(self.topk_pair), 4), dtype=np.int64) # img1, img2, start_idx in corr, end_idx in corr


        with torch.no_grad() : 
            for i in range(self.keep_idx.shape[0]) : 
                idx = self.keep_idx[i]
                idx1, idx2 = self.pair_idx[idx, 0], self.pair_idx[idx, 1]
                if i % 10000 == 9999 : 
                    print (i, start)

                if (idx1, idx2) not in self.topk_pair and (idx2, idx1) not in self.topk_pair: 
                    continue
                
                conf1 = self.out1_m[i] 
                conf2 = self.out2_m[i] 

                conf1_warp = conf1 
                conf2_warp = conf2 

                corr1_idx = conf1_warp > conf_th
                corr2_idx = conf2_warp > conf_th

                if corr1_idx.sum() == 0 or corr2_idx.sum() == 0 :
                    continue
                if corr1_idx.sum() > corr2_idx.sum() : 
                    corr1_x1, corr1_y1 = self.grid_x[corr1_idx],  self.grid_y[corr1_idx]
                    corr1_x2, corr1_y2, corr1_conf = self.out1_x[i][corr1_idx], self.out1_y[i][corr1_idx], conf1_warp[corr1_idx]
                    dict_corr = {}
                    for j in range(len(corr1_x1)) : 
                        dict_corr[(corr1_x1[j], corr1_y1[j], corr1_x2[j], corr1_y2[j])] = corr1_conf[j]
                
                else : 
                    corr2_x2, corr2_y2 = self.grid_x[corr2_idx],  self.grid_y[corr2_idx]
                    corr2_x1, corr2_y1, corr2_conf = self.out2_x[i][corr2_idx], self.out2_y[i][corr2_idx], conf2_warp[corr2_idx]

                    dict_corr = {}
                    for j in range(len(corr2_x1)) : 
                        dict_corr[(corr2_x1[j], corr2_y1[j], corr2_x2[j], corr2_y2[j])] = corr2_conf[j]
                
                pair_corr = list(dict_corr.keys())
                conf = np.array([dict_corr[key] for key in pair_corr])
                pair_corr = np.array([list(key) for key in pair_corr])

                
                nb_corr = len(pair_corr)
                
                if nb_corr > 0 : 
                 
                    end = start + nb_corr
                    self.corr[start : end, :4] = pair_corr
                    self.corr[start : end, 4] = conf

                    self.img_idx_start_end[pair_count, 0] = idx1
                    self.img_idx_start_end[pair_count, 1] = idx2

                    self.img_idx_start_end[pair_count, 2] = start
                    self.img_idx_start_end[pair_count, 3] = end


                    pair_count += 1
                    start = end
                    
        self.corr = self.corr[: end]
        self.img_idx_start_end = self.img_idx_start_end[:pair_count]
        #print ('Nb of nodes: {:d}, Nb of pairs {:d}'.format(len(self.corr), len(self.img_idx_start_end)))
        
def mask_color_compose(org, mask, mask_color = [173, 216, 230]) : 
    
    mask_fg = mask == 1
    
    rgb = np.copy(org)
    rgb[mask_fg] = (rgb[mask_fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    
    return Image.fromarray(rgb)
    



        
    

if __name__ == "__main__":
    import argparse 
    import FullGraph3Cycle
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Evaluate cross-transformer on Internet dataset for discovery')
    
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
    ## input / output dir 
    
    parser.add_argument('--db-dir', type=str, default=['../data/Internet/Car100', '../data/Internet/Airplane100', '../data/Internet/Horse100'], nargs='+', help='db directory')
    
    parser.add_argument('--corr-dir', type=str, default=['neg5_car_480p', 'neg5_airplane_480p', 'neg5_horse_480p'], nargs='+', help='dir containing corrspondences from pairs')
    
    parser.add_argument('--order', type=str, default=['Car', 'Airplane', 'Horse'], nargs='+', help='order of the files')
    
    
    parser.add_argument('--feat-size', type=int, nargs='+', default=[30, 30], help='image size')

    parser.add_argument('--top-k', type=int, default=5, help='Keep topk retrieval results')

    parser.add_argument('--conf-th', type=float, default=0.85, help='threshold to filter noisy correspondences')

    parser.add_argument('--space-log', type=float, default=1.1, help='space log')
    
    parser.add_argument('--sigma', type=float, nargs='+', default=[5.0, 0.8, 1.2], help='sigma context for discovery')
    
    
    parser.add_argument('--fg-th', type=float, default=0.6, help='foreground threshold')
    
    parser.add_argument('--dilate', type=int, default=10, help='sigma for blurring')
    
    parser.add_argument('--only3cycle', action='store_true', help='only 3 cycle or not')
    
    parser.add_argument('--mask-in-3cycle', action='store_true', help='mask in 3 cycle connections?')
    
    parser.add_argument('--mode', type=str, choices=['sum', 'max', 'mean'], default='sum', help='mode to merge activation')
    
    args = parser.parse_args()
    print (args)

    
    ## set gpu

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
        
    feat_dim=1024
            
    acc_cat = []
    iou_cat = []
    for i in range(len(args.db_dir)) : 
        
        db_dir = args.db_dir[i]
        corr_dir = args.corr_dir[i]
        
        
        PairCompute = Discovery(db_dir = db_dir,
                                 feat_dim = feat_dim,
                                 feat_size = args.feat_size,
                                 top_k = args.top_k,
                                 corr_dir = corr_dir)
    
        PairCompute.get_pair_score()
    
    
        #print ('Get top retrieved images : ') 
        PairCompute.get_topk_pair()
        PairCompute.get_nodes(conf_th = args.conf_th)

        
    
        gt_dir = os.path.join(db_dir, 'GroundTruth')
        eigen_vector = FullGraph3Cycle.OptimFullGraph(corr = PairCompute.corr,
                                               img_idx_start_end = PairCompute.img_idx_start_end,
                                               sigma = args.sigma[i],
                                               pair_idx = PairCompute.pair_idx,
                                               keep_idx = PairCompute.keep_idx,
                                               out1x = PairCompute.out1_x,
                                               out1y = PairCompute.out1_y, 
                                               out1m = PairCompute.out1_m, 
                                               out2x = PairCompute.out2_x, 
                                               out2y = PairCompute.out2_y, 
                                               out2m = PairCompute.out2_m,
                                               mask_in_3cycle = args.mask_in_3cycle,
                                               only3cycle = args.only3cycle)
        
        mask_list = VisEigen.VisEigen(None, PairCompute.img_idx_start_end, PairCompute.corr, PairCompute.db_dir, args.feat_size[0], args.feat_size[1], PairCompute.img_list, eigen_vector, args.mode)
    
        acc = []
        iou = []
        for idx in range(len(mask_list)) : 
            gt = np.array(Image.open(os.path.join(gt_dir, PairCompute.img_list[idx].replace('.jpg', '.png')))).astype(bool)
            h, w = gt.shape[:2]
            pred = soft2binarymask(mask_list[idx], idx, db_dir, PairCompute.img_list, args.fg_th, args.dilate)
            acc_i, iou_i = compute_acc_iou(pred, gt)
    
            acc.append(acc_i)
            if iou_i is not None : 
                iou.append(iou_i)
            
        
        print ('{}, Acc : {:.3f} \t IoU : {:.3f}'.format(args.order[i], np.mean(acc), np.mean(iou)))
        acc_cat.append(np.mean(acc))
        iou_cat.append(np.mean(iou))
        
        del PairCompute
        
    print ('--- Avg Acc : {:.3} ---\n'.format(np.mean(acc_cat)))
    
    print ('--- Avg IoU : {:.3} ---\n'.format(np.mean(iou_cat)))
    
                
