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

class ImgListLoader(Dataset):

    def __init__(self, img_dir, img_list, img_idx_list, img_size):
        
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
        self.img_idx_list = img_idx_list
        self.nb_img = len(img_idx_list)
    
    def __getitem__(self, idx):
        img_name = self.img_list[self.img_idx_list[idx]]
        img = os.path.join(self.img_dir, img_name)
        I = Image.open(img).convert('RGB')
        I = self.transform(I)
        return I, idx
     
    def __len__(self):
        return self.nb_img
    
## Data loader
def ImgLoader(img_dir, img_list, img_idx_list, batch_size, img_size):

    dataSet = ImgListLoader(img_dir, img_list, img_idx_list, img_size)
    dataLoader = DataLoader(dataset=dataSet, batch_size=batch_size, shuffle=False, num_workers=1, drop_last = False)

    return dataLoader

def resize_arr(input_arr, out_w, out_h) : 
    out_img = Image.fromarray((input_arr * 255).astype(np.uint8)).resize((out_w, out_h), resample= Image.BILINEAR)
    out_arr = np.array(out_img) 
    return out_arr
 





def score_local_feat_match(feat_qry_bag, feat_2, x2, y2, weight_feat) :
    with torch.no_grad() : 
        local_feat = [ feat_2[:, y2[i], x2[i]].unsqueeze(0) for i in range(len(x2))]
        feat_2_bag = torch.cat(local_feat, dim=0)
        ## each local feature is matched to  its feature in an other image, finally the similarity is weighted by the mask prediction 
        score_weight_feat = torch.sum(feat_qry_bag * feat_2_bag, dim=1) * torch.from_numpy(weight_feat).cuda()
        dict_score = {(y2[i], x2[i]): [0] for i in range(len(x2))}
        for i in range(len(x2)) : 
            dict_score[(y2[i], x2[i])].append(score_weight_feat[i].item())
        
        return  sum([np.max(dict_score[key]) for key in dict_score])
    
def consistent_mask(mask_query, mask_target, o1, o2) : 
    
    flow12 = (o1.narrow(0, 0, 2).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2 
    flow21 = (o2.narrow(0, 0, 2).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2 
    

    mask_target_grid = F.grid_sample(torch.from_numpy(mask_query.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow21, mode='bilinear').numpy().squeeze()
    mask_target_grid = mask_target_grid  * mask_target

    mask_query_back = F.grid_sample(torch.from_numpy(mask_target_grid.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow12, mode='bilinear')
    mask_query_final  = (mask_query * mask_query_back.squeeze().numpy() )

    mask_target_final = F.grid_sample(torch.from_numpy(mask_query_final.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow21, mode='bilinear')
    mask_target_final = (mask_target_final.squeeze().numpy() )
    return mask_query_final, mask_target_final

def transported_mask(mask_query, mask_target, o1, o2) : 
    
    flow12 = (o1.narrow(0, 0, 2).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2 
    

    mask_target_transported = F.grid_sample(torch.from_numpy(mask_target.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow12, mode='bilinear').numpy().squeeze()
    
    mask_query_final = mask_query  * mask_target_transported

    return mask_query_final

class EvalTokyo247:
    def __init__(self,
                 qry_dir,
                 db_dir,
                 feat_dim):
        
        self.qry_dir = qry_dir 
        self.db_dir = db_dir 
        self.nms = True
        score = np.load('top100_patchVlad.npy')

        self.qry_list = np.load('qry_list.npy')
        self.qry_utm = np.load('qry_utm.npy')
        self.nb_qry = len(self.qry_list) 
        
        self.db_list = np.load('db_list.npy')
        self.db_utm = np.load('db_utm.npy')

        self.idx_sort = np.argsort(-1 * score, axis=1)
        self.topk = 100


        self.stride_net = 16 # stride of the backbone net
        self.batch_size = 32 # batch size for the test
        self.feat_dim = feat_dim
        
        
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.img_size = (640, 480) # width = 640, height = 480 to make it comparable to NetVlad
        self.nb_feat_w = self.img_size[0] // self.stride_net
        self.nb_feat_h = self.img_size[1] // self.stride_net
        
        self.transformINet = transforms.Compose([transforms.Resize((self.img_size[1], self.img_size[0])), # (height, width)
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(norm_mean, norm_std)])
        
    
        self.nb_batch = self.topk // self.batch_size
        self.last_batch = self.topk - self.nb_batch * self.batch_size
        
        self.o1_list = torch.cuda.FloatTensor(self.topk, 3, self.nb_feat_h, self.nb_feat_w)
        self.o2_list = torch.cuda.FloatTensor(self.topk, 3, self.nb_feat_h, self.nb_feat_w)
        self.db_feat = torch.cuda.FloatTensor(self.topk, feat_dim, self.nb_feat_h, self.nb_feat_w)

    
        
    def evalTokyo247(self, backbone, netEncoder):
        torch.cuda.empty_cache()
        backbone.eval()
        netEncoder.eval()
        
        
        ## compute results
        res = np.copy(self.idx_sort)
        
        for qry_id in tqdm(range(self.nb_qry)) : 
            
            query_name = self.qry_list[qry_id]
            qry = Image.open(os.path.join(self.qry_dir, query_name)).convert('RGB')
            

            mask_qry_dilate = np.ones((self.nb_feat_h, self.nb_feat_w), dtype=bool)
            y_grid, x_grid = np.where(mask_qry_dilate)
            
            
            qry_feat = self.transformINet(qry).unsqueeze(0).cuda()
            qry_feat = F.normalize(backbone(qry_feat))
            qry_feat = qry_feat.expand(self.batch_size, self.feat_dim, self.nb_feat_h, self.nb_feat_w)
            
            o1_qry, o2_qry = netEncoder(qry_feat.narrow(0, 0, 1),
                                        qry_feat.narrow(0, 0, 1))
                
            mask_qry_dilate, _ = consistent_mask(mask_qry_dilate * o1_qry[0, 2].cpu().numpy(), o2_qry[0, 2].cpu().numpy(), o1_qry[0].cpu(), o2_qry[0].cpu())
            
            
            db_loader = ImgLoader(self.db_dir, self.db_list, self.idx_sort[qry_id, : self.topk], self.batch_size, self.img_size)
            
            with torch.no_grad():
                for (batch, idx) in db_loader : 
                    batch = batch.cuda()
                    self.db_feat[idx] = backbone(batch)

                self.db_feat = F.normalize(self.db_feat, dim=1)
            
            for i in range(self.nb_batch) : 
                start = i * self.batch_size
                o1, o2 = netEncoder(qry_feat,
                                    self.db_feat.narrow(0, start, self.batch_size).cuda())
                self.o1_list[start : start + self.batch_size] = o1
                self.o2_list[start : start + self.batch_size] = o2

                if self.last_batch : 
                    o1, o2 = netEncoder(qry_feat.narrow(0, 0, self.last_batch),
                                        self.db_feat.narrow(0, self.nb_batch * self.batch_size, self.last_batch).cuda())

                    self.o1_list[self.nb_batch * self.batch_size : ] = o1
                    self.o2_list[self.nb_batch * self.batch_size : ] = o2


            result_query = np.zeros(len(self.db_feat)) # score
            
            qry_feat_bag = [ qry_feat[0, :, y_grid[i], x_grid[i]].unsqueeze(0) for i in range(len(x_grid))]
            qry_feat_bag = torch.cat(qry_feat_bag, dim=0)

            o1 = self.o1_list.cpu()
            o2 = self.o2_list.cpu()
            x1, y1, mask1, mask2 =o1[:, 0].numpy() , o1[:, 1].numpy(), o1[:, 2].numpy(), o2[:, 2].numpy()

            for i in range(len(self.o1_list)) : 

                x2_by_pred1 = np.round(x1[i, y_grid, x_grid] * (self.nb_feat_w - 1)).astype(int)
                y2_by_pred1 = np.round(y1[i, y_grid, x_grid] * (self.nb_feat_h - 1)).astype(int)
                
                mask1_csit = transported_mask(mask_qry_dilate * mask1[i], mask2[i], o1[i], o2[i])
                
                weight_feat = mask1_csit[y_grid, x_grid]
                score = score_local_feat_match(qry_feat_bag, self.db_feat[i].cuda(), x2_by_pred1, y2_by_pred1, weight_feat)    
                    
                result_query[i] = score
                
            res[qry_id, : self.topk] = self.idx_sort[qry_id, : self.topk][np.argsort(-1 * result_query)]
            
            if qry_id % 30 == 29 : 
                recall1, recall5, recall10, recall100 = self.compute_recall(self.idx_sort)
                print ('{:d} / {:d} \t Raw recall @ 1 : {:.3f}, @ 5 : {:.3f}, @ 10 : {:.3f}, @ 100 : {:.3f}'.format(qry_id, self.nb_qry, recall1, recall5, recall10, recall100))

                recall1, recall5, recall10, recall100 = self.compute_recall(res)
                print ('{:d} / {:d} \t Our recall @ 1 : {:.3f}, @ 5 : {:.3f}, @ 10 : {:.3f}, @ 100 : {:.3f}'.format(qry_id, self.nb_qry, recall1, recall5, recall10, recall100))

        recall1, recall5, recall10, recall100 = self.compute_recall(res)
        print ('Our recall @ 1 : {:.3f}, @ 5 : {:.3f}, @ 10 : {:.3f}, @ 100 : {:.3f}'.format(recall1, recall5, recall10, recall100))        
        if self.nms : 
            print ('After Spatial NMS...')
            idx_nms = self.spatial_nms(self.idx_sort)
            recall1, recall5, recall10, recall100 = self.compute_recall(idx_nms)

            print ('After Spatial NMS Raw recall @ 1 : {:.3f}, @ 5 : {:.3f}, @ 10 : {:.3f}, @ 100 : {:.3f}'.format(recall1, recall5, recall10, recall100))


            res_nms = self.spatial_nms(res)
            recall1, recall5, recall10, recall100 = self.compute_recall(res_nms)

            print ('After Spatial NMS Ours recall @ 1 : {:.3f}, @ 5 : {:.3f}, @ 10 : {:.3f}, @ 100 : {:.3f}'.format(recall1, recall5, recall10, recall100))
        return recall1
        
        
    
    def spatial_nms(self, rank_idx) : 
        
        '''
        in the tokyo247 db directory, every 12 images share the same utm position, make sense to do a spaitial nms
        '''
        rank_idx_update = rank_idx // 12
        rank_idx_nms = np.zeros((rank_idx.shape[0], rank_idx.shape[1] // 12), dtype=np.int32)

        for i in tqdm(range(rank_idx.shape[0])) : 
            idx_update = rank_idx_update[i]
            idx_org = rank_idx[i]
            idx_nms = [j for j in range(len(idx_update)) if idx_update[j] not in idx_update[:j]]
            rank_idx_nms[i] = idx_org[np.array(idx_nms)]
        return rank_idx_nms

    def compute_recall(self, rank_idx) : 
        
        dist_top1 = np.sum((self.db_utm[rank_idx[:, 0]] - self.qry_utm) ** 2, axis=1) **0.5
        
        dist_top5 = np.sum((self.db_utm[rank_idx[:, :5]] - self.qry_utm[:, None]) ** 2, axis=2) **0.5
        
        dist_top10 = np.sum((self.db_utm[rank_idx[:, :10]] - self.qry_utm[:, None]) ** 2, axis=2) **0.5

        dist_top100 = np.sum((self.db_utm[rank_idx[:, :100]] - self.qry_utm[:, None]) ** 2, axis=2) **0.5

        return (dist_top1 <= 25).mean(), (dist_top5.min(axis=1) <= 25).mean(), (dist_top10.min(axis=1) <= 25).mean(), (dist_top100.min(axis=1) <= 25).mean()
    
                    

if __name__ == "__main__":
    import argparse 
    import sys 
    sys.path.append('../model')
    import transformer # model
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Evaluate cross-transformer on tokyo247')

    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
    
    parser.add_argument('--resume-pth', type = str, default='../model/hard_mining_neg5.pth', help='resume path')
    
    ## paramter transformer
    parser.add_argument('--mode', type=str, choices=['tiny', 'small', 'base', 'large'], default='small', help='different size of transformer encoder')
    
    parser.add_argument('--pos-weight', type=float, default=0.1, help='weight for positional encoding')
    
    parser.add_argument('--feat-weight', type=float, default=1, help='weight for feature component')
    
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout in the transformer layer')
    
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='activation in the transformer layer')
    
    parser.add_argument('--layer-type', type=str, nargs='+', default=['I', 'C', 'I', 'C', 'I', 'N'], help='which type of layers: I is for inner image attention, C is for cross image attention, N is None')
    
    parser.add_argument('--drop-feat', type=float, default=0.1, help='drop feature rate')
    
    parser.add_argument('--qry-dir', type=str, default='/space_sdd/PlaceReco/Tokyo247/query/247query_subset_v2', help='query directory')

    parser.add_argument('--db-dir', type=str, default='/space_sdd/PlaceReco/Tokyo247/database/', help='db directory')
    
    parser.add_argument('--score', type=str, default='top100_patchVlad.npy', help='top100 using patchVlad feature')
    
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

        
    TokyoEvaluator = EvalTokyo247(qry_dir = args.qry_dir,
                                 db_dir = args.db_dir,
                                 feat_dim = feat_dim)
    with torch.no_grad() : 
        TokyoEvaluator.evalTokyo247(backbone, netEncoder)
    
    

    

                



