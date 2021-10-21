# coding=utf-8
import json 
import preprocess_qry, preprocess_db
import os 
import torch.nn.functional as F
import pickle 

import numpy as np 
import torch 
import torchvision.transforms as transforms
from tqdm import tqdm 

from scipy import ndimage
from PIL import ImageDraw, Image
import torchvision.models as models

 
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
    

def transported_mask(mask_query, mask_target, o1, o2) : 
    
    flow12 = (o1.narrow(0, 0, 2).unsqueeze(0).permute(0, 2, 3, 1) - 0.5) * 2 
    

    mask_target_transported = F.grid_sample(torch.from_numpy(mask_target.astype(np.float32)).unsqueeze(0).unsqueeze(0), flow12, mode='bilinear').numpy().squeeze()
    
    mask_query_final = mask_query  * mask_target_transported

    return mask_query_final

def coarse_pred(x1, y1) : 
    '''
    x_grid, y_grid : grid points in the query, both need to be a 1-d array
    x1, y1 : flow in db image, 1-d array
    weight_feat: weight of each correspondence
    
    '''
    
    
    coarse_x =x1.mean()
    coarse_y =y1.mean()
    
    return coarse_x, coarse_y

class EvalBrueghel:
    def __init__(self, feat_pth,
                       img_size,
                       qry_size,
                       feat_dim,
                       label_pth,
                       img_dir,
                       out_coarse = None):
        
        self.img_size = img_size
        self.qry_size = qry_size
        self.nb_db_img = 1587
        self.stride_net = 16 # stride of the backbone net
        self.batch_size = 32 # batch size for the test
        self.feat_dim = feat_dim
        self.nb_feat = self.img_size // self.stride_net
        
        self.img_dir = img_dir
        self.feat_pth = feat_pth
        
        
        self.out_coarse = out_coarse
        
        with open(label_pth, 'r') as f : 
            self.label = json.load(f)

        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        self.transformINet = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])

        self.idx2img = sorted(os.listdir(self.img_dir))
        
        self.img2idx = {self.idx2img[i] : i for i in range(len(self.idx2img))}
        self.nb_batch = min(len(self.img2idx), self.nb_db_img) // self.batch_size
        self.last_batch = min(len(self.img2idx), self.nb_db_img) - self.nb_batch * self.batch_size
        self.o1_list = torch.cuda.FloatTensor(self.nb_db_img, 3, self.nb_feat, self.nb_feat)
        self.o2_list = torch.cuda.FloatTensor(self.nb_db_img, 3, self.nb_feat, self.nb_feat)
    
    def compute_db_feat(self, backbone) : 
        
        backbone.eval()
        #### precomputing database features, will take 30s + 4G GPU memory
        if not os.path.exists (self.feat_pth) : 

            featList = torch.FloatTensor(len(self.idx2img), self.feat_dim, self.nb_feat, self.nb_feat) # store feature with high resolution
            wh_org_list = []


            ## pre-compute all features in the database
            with torch.no_grad() :
                for i in tqdm(range(len(self.idx2img))) : 
                    db = Image.open(os.path.join(self.img_dir, self.idx2img[i])).convert('RGB')
                    wh_org_list.append((db.size[0], db.size[1]))

                    db = preprocess_db.preprocess_db(db, self.img_size)
                    tensor = self.transformINet(db).unsqueeze(0).cuda()
                    feat = backbone( tensor )
                    featList[i] = feat[0].cpu()

            
            with open(self.feat_pth, 'wb') as f : 
                pickle.dump({'feat' : featList.cpu().numpy(), 
                             'org' : wh_org_list}, f, protocol=4)

        else : 
            with open(self.feat_pth, 'rb') as f :
                info = pickle.load(f)
            wh_org_list = info['org']
            featList = F.normalize(torch.from_numpy(info['feat']))
        return wh_org_list, featList
            
    def get_qry_bb(self, qry_bbox) : 
        

        qry_bbox_crop_feat = [qry_bbox[0] / self.stride_net,
                              qry_bbox[1] / self.stride_net,
                              qry_bbox[2] / self.stride_net,
                              qry_bbox[3] / self.stride_net]

        qry_bbox_crop_feat = [max(int(round(qry_bbox_crop_feat[0] - 1)), 0),
                              max(int(round(qry_bbox_crop_feat[1] - 1)), 0),
                              min(int(round(qry_bbox_crop_feat[2] + 1)), self.nb_feat),
                              min(int(round(qry_bbox_crop_feat[3] + 1)), self.nb_feat)
                              ]
        
        mask_qry = np.zeros((self.nb_feat, self.nb_feat))

        mask_qry[qry_bbox_crop_feat[1] : qry_bbox_crop_feat[3], qry_bbox_crop_feat[0] : qry_bbox_crop_feat[2]] = 1
        
        
        return mask_qry
            
    def evalBrueghelOneShotMask(self, backbone, netEncoder):
        torch.cuda.empty_cache()
        backbone.eval()
        netEncoder.eval()
        
        ## compute database feature
        wh_org_list, featList = self.compute_db_feat(backbone)
        
        
        ## compute detection results
        res = {}
        res_coarse = {}
        cate_list = sorted(list(self.label.keys()))
        count = 0
        for cat_id in tqdm(range(len(cate_list))) : 

            category = cate_list[cat_id]
            res[category] = []
            res_coarse[category] = []

            for query_id_label in tqdm(range(len(self.label[category]))) :

                query = self.label[category][query_id_label]['query']
                query_name = query[0]
                qry_bbox = query[1]
                query_idx = self.img2idx[query_name]

                qry = Image.open(os.path.join(self.img_dir, query_name))
                
                qry_crop, qry_bbox_crop = preprocess_qry.preprocess_query(qry, qry_bbox, self.qry_size, self.img_size, self.stride_net)
                
                mask_qry_dilate = self.get_qry_bb(qry_bbox_crop)
                y_grid, x_grid = np.where(mask_qry_dilate)
                
                with torch.no_grad() :      
                    qry_feat = self.transformINet(qry_crop).unsqueeze(0).cuda()
                    qry_feat = F.normalize(backbone(qry_feat))
                    qry_feat = qry_feat.expand(self.batch_size, self.feat_dim, self.nb_feat, self.nb_feat)
                    

                gt_dict = {self.img2idx[img] : bbox for (img, bbox) in self.label[category][query_id_label]['gt']}
                
                
                with torch.no_grad() : 
                    
                    o1_qry, o2_qry = netEncoder(qry_feat.narrow(0, 0, 1),
                                            qry_feat.narrow(0, 0, 1))
                    
                    mask_qry_dilate, _ = consistent_mask(mask_qry_dilate * o1_qry[0, 2].cpu().numpy(), o2_qry[0, 2].cpu().numpy(), o1_qry[0].cpu(), o2_qry[0].cpu())
                
                    for i in tqdm(range(self.nb_batch)) : 
                        start = i * self.batch_size
                        o1, o2 = netEncoder(qry_feat.narrow(0, 0, self.batch_size),
                                            featList.narrow(0, start, self.batch_size).cuda())
                        
                        self.o1_list[start : start + self.batch_size] = o1
                        self.o2_list[start : start + self.batch_size] = o2
                        
                    if self.last_batch : 
                        
                        o1, o2 = netEncoder(qry_feat.narrow(0, 0, self.last_batch),
                                            featList.narrow(0, self.nb_batch * self.batch_size, self.last_batch).cuda())
                        
                        self.o1_list[self.nb_batch * self.batch_size : ] = o1
                        self.o2_list[self.nb_batch * self.batch_size : ] = o2




                    result_query = [] # mask, score, iou
                    coarse_xy = []

                    qry_feat_bag = [ qry_feat[0, :, y_grid[i], x_grid[i]].unsqueeze(0) for i in range(len(x_grid))]
                    qry_feat_bag = torch.cat(qry_feat_bag, dim=0)
                    
                    o1 = self.o1_list.cpu()
                    o2 = self.o2_list.cpu()
                    x1, y1, mask1, mask2 =o1[:, 0].numpy() , o1[:, 1].numpy(), o1[:, 2].numpy(), o2[:, 2].numpy()
                    
                    for i in range(len(self.o1_list)) : 
                        if i == query_idx : 
                            continue
                        
                        x2_by_pred1 = np.round(x1[i, y_grid, x_grid] * (self.nb_feat - 1)).astype(int)
                        y2_by_pred1 = np.round(y1[i, y_grid, x_grid] * (self.nb_feat - 1)).astype(int)
                        
                        mask1_csit = transported_mask(mask_qry_dilate * mask1[i], mask2[i], o1[i], o2[i])
                
                        weight_feat = mask1_csit[y_grid, x_grid]
                        coarse_x, coarse_y = coarse_pred(x1[i, y_grid, x_grid].reshape(-1), y1[i, y_grid, x_grid].reshape(-1)) 
                        score = score_local_feat_match(qry_feat_bag, featList[i].cuda(), x2_by_pred1, y2_by_pred1, weight_feat)    
                        
                        result_query.append((score, i in gt_dict, i, x1[i, y_grid, x_grid], y1[i, y_grid, x_grid], o2[i, 2].numpy(), wh_org_list[i][0], wh_org_list[i][1]))
                        coarse_xy.append((float(coarse_x), float(coarse_y), float(score), i in gt_dict, i))
                    
                result_query = sorted(result_query, key = lambda s: s[0], reverse=True)
                coarse_xy = sorted(coarse_xy, key = lambda s: s[2], reverse=True)
                true_pos_rank = [item[1] for item in result_query]

                AP = [np.sum(true_pos_rank[: idx + 1]) / (idx + 1) for idx in range(len(true_pos_rank)) if true_pos_rank[idx]]

                mAP = sum(AP) / len(gt_dict)
                
                res[category].append(mAP)
                res_coarse[category].append(coarse_xy)
                    
        res_mean_category =[np.mean(res[category]) for category in res]
        backbone.train()
        netEncoder.train()
        
        if self.out_coarse is not None : 
            with open(args.out_coarse, 'w') as f: 
                json.dump(res_coarse, f)
                
        return np.mean(res_mean_category)
    
    
    

if __name__ == "__main__":
    import argparse 
    import sys 
    sys.path.append('../model')
    import transformer # model
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='Evaluate cross-transformer on Brueghel dataset')

    ## optimiser
    parser.add_argument('--gpu', type=str, default='0', help='gpu devices')
    
    ## input / output dir 
    
    parser.add_argument('--test-dir', type=str, default='../data/Brueghel/Image/', help='test directory')
    parser.add_argument('--label-pth', type=str, default='../data/Brueghel/brueghelVal.json', choices=['../data/Brueghel/brueghelVal.json', '../data/Brueghel/brueghelTest.json'], help='Brueghel label path')

    parser.add_argument('--resume-pth', type = str, default='../model/hard_mining_neg5.pth', help='resume path')
    
    parser.add_argument('--feat-pth', type = str, default='Moco_resnet50_feat_1Scale_640p.pkl', help='save feature ? for fast computation')

    ## paramter transformer
    parser.add_argument('--mode', type=str, choices=['tiny', 'small', 'base', 'large'], default='small', help='different size of transformer encoder')
    parser.add_argument('--pos-weight', type=float, default=0.1, help='weight for positional encoding')
    parser.add_argument('--feat-weight', type=float, default=1, help='weight for feature component')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout in the transformer layer')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='activation in the transformer layer')
    
    parser.add_argument('--img-size', type=int, default=640, help='image size for evaluation')
    parser.add_argument('--qry-size', type=int, default=64, help='query size (max dimension)')


    parser.add_argument('--nb-db-img', type=int, default=1587, help='nb of database image for evaluation')
        
    parser.add_argument('--layer-type', type=str, nargs='+', default=['I', 'C', 'I', 'C', 'I', 'N'], help='which type of layers: I is for inner image attention, C is for cross image attention, N is None')
    
    parser.add_argument('--out-coarse', type=str, default=None,  help='output coarse json file')
    
    parser.add_argument('--drop-feat', type=float, default=0.1, help='drop feature rate')
    
    
    args = parser.parse_args()
    print (args)

    
    ## set gpu

    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    ## load Moco feature
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

        
    BrueghelEvaluator = EvalBrueghel(feat_pth = args.feat_pth,
                                     img_size=args.img_size,
                                     qry_size=args.qry_size,
                                     feat_dim = feat_dim,
                                     label_pth = args.label_pth,
                                     img_dir = args.test_dir,
                                     out_coarse = args.out_coarse)
    
    mAP = BrueghelEvaluator.evalBrueghelOneShotMask(backbone, netEncoder)
    print ('final mAP is {:.3f}...'.format(mAP))
 
    

    

                


