
from datetime import datetime
import numpy as np 
import torch
import torch.nn.functional as F
from tqdm import tqdm


def stable_clamp(out) : 
    eps = 1e-7
    out = torch.clamp(out, min=eps, max=1-eps)
    return out

def Warmup(trainLoader, backbone, netEncoder, optimizer, Loss, batch_pos, batch_neg, warp_mask, logger, eta_corr, warmup_iter, lr) : 
    
    backbone.eval()
    netEncoder.train()
    
    
    loss_log = []
    loss_mask_log = []
    loss_corr_log = []
    
    acc_log = []
    acc_pos_log = []
    acc_neg_log = []
    
    pos_idx = torch.arange(batch_pos).cuda()
    neg_idx = torch.cat([pos_idx for i in range(batch_pos - 1)], dim=0)
    neg_idx_shuffle = torch.cat([torch.cat([pos_idx[i+1:], pos_idx[0:i+1]], dim=0) for i in range(batch_pos - 1) ], dim=0)
    neg_idx = neg_idx[:batch_neg]
    neg_idx_shuffle = neg_idx_shuffle[:batch_neg]

    trainLoader_iter = iter(trainLoader)
    
    for batch_id in range(warmup_iter):   
        idx = np.random.choice(len(trainLoader))
        
        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter[idx] = iter(trainLoader)
            batch = next(trainLoader_iter)
        
        ## put all into cuda
        T1 = batch['T1'].cuda()
        T2 = batch['T2'].cuda()
        
        
        RM1 = batch['RM1'].cuda()
        RM2 = batch['RM2'].cuda()
        
        M1 = batch['M1'].cuda()
        M2 = batch['M2'].cuda()
        
        xy1 = batch['xy1'].cuda()
        xy2 = batch['xy2'].cuda()
        
        
        optimizer.zero_grad()
        
        with torch.no_grad() : 
            T1 = F.normalize(backbone(T1), dim=1)
            T2 = F.normalize(backbone(T2), dim=1)

                
        if np.random.rand() > 0.5 :
            X, Y =  T1, T2
            MX, MY, RS, RT = M1, M2, RM1, RM2
            PosX, PosY = xy1, xy2
        else : 
            X, Y =  T2, T1
            MX, MY, RS, RT = M2, M1, RM2, RM1
            PosX, PosY = xy2, xy1
            
        O1, O2 = netEncoder(X, Y, RS, RT)
        
        if warp_mask : 
            flow12 = ((O1.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
            flow21 = ((O2.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
        

            O2_warp = F.grid_sample(O1.narrow(1, 2, 1), flow21, mode='bilinear')
            O2_warp = stable_clamp(O2_warp)
            O1_warp = F.grid_sample(O2.narrow(1, 2, 1), flow12, mode='bilinear')
            O1_warp = stable_clamp(O1_warp)
            
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1), O1_warp, O2_warp], dim=0)
            target_pos = torch.cat([MX, MY, MX, MY], dim=0)

        
        else :
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            target_pos = torch.cat([MX, MY], dim=0)
            
        with torch.no_grad() : 
            
            target_pos_binary = torch.cat([MX, MY], dim=0) 
            target_pos_view = target_pos_binary.view(target_pos_binary.size()[0], 1, 1, -1)
            target_pos_view = torch.max( target_pos_view, dim=3, keepdim=True)[0]
            
            target_pos_binary = (target_pos_binary == target_pos_view)
            target_pos_binary_float = target_pos_binary.type(torch.cuda.FloatTensor)

        
        output_xy = torch.cat([O1.narrow(1, 0, 2), O2.narrow(1, 0, 2)], dim=0)
        target_xy = torch.cat([PosX, PosY], dim=0)
        
        loss_corr = (torch.abs(target_xy - output_xy) * target_pos_binary_float ).sum() / target_pos_binary_float.sum() / 2
        
        O1_neg, O3_neg = netEncoder(T1[neg_idx], T2[neg_idx_shuffle])
        output_neg = torch.cat([O1_neg.narrow(1, 2, 1), O3_neg.narrow(1, 2, 1)], dim=0)
        target_neg = torch.zeros_like(output_neg)
        
        output = torch.cat((output_pos, output_neg), dim=0)
        target = torch.cat((target_pos, target_neg), dim=0)
        
        loss_mask = Loss(output, target)
        loss = loss_mask + eta_corr * loss_corr if loss_mask is not None else eta_corr * loss_corr
        
        loss.backward()
        optimizer.step()
        
        loss_log.append(loss.item())
        if loss_mask is not None : 
            loss_mask_log.append(loss_mask.item())
        else : 
            loss_mask_log.append(0)
        
        loss_corr_log.append(loss_corr.item())
        
        
        with torch.no_grad() : 
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            acc_pos = (((output_pos > 0.5) == target_pos_binary).type(torch.cuda.FloatTensor) * target_pos_binary_float).sum() / target_pos_binary_float.sum()
            acc_pos_log.append(acc_pos.item())

            element_neg = 1 - target_pos_binary_float
            acc_neg = ((((output_pos < 0.5) == (~target_pos_binary)).type(torch.cuda.FloatTensor) * element_neg).sum() + (output_neg < 0.5).type(torch.cuda.FloatTensor).sum()) / (element_neg.sum() + (1 - target_neg).sum())

            acc_neg_log.append(acc_neg.item())

            acc_log.append(acc_pos_log[-1] * 0.5 + acc_neg_log[-1] * 0.5)
            
        for g in optimizer.param_groups:
            g['lr'] = lr * batch_id / warmup_iter 

        
        if batch_id % 100 == 99 : 
            for g in optimizer.param_groups:
                lr_print = g['lr']
                break
            msg = '{} Batch id {:d}, Lr {:.6f}; \t | Loss : {:.3f}, Mask : {:.3f}, Corr : {:.3f} |  Acc : {:.3f}%, Pos : {:.3f}%, Neg : {:.3f}%  \t '.format(datetime.now().time(), batch_id + 1, lr_print, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_corr_log), np.mean(acc_log) * 100, np.mean(acc_pos_log) * 100, np.mean(acc_neg_log) * 100)
            
            logger.info(msg)
            
    
    for g in optimizer.param_groups:
        g['lr'] = lr 
    
    return backbone, netEncoder, optimizer, np.mean(acc_pos_log), np.mean(acc_neg_log)





def trainEpoch(trainLoader, backbone, netEncoder, optimizer, history, Loss, batch_pos, batch_neg, warp_mask, logger, eta_corr, iter_epoch) : 
    
    backbone.eval()
    netEncoder.train()
    
    
    loss_log = []
    loss_mask_log = []
    loss_corr_log = []
    
    acc_log = []
    acc_pos_log = []
    acc_neg_log = []
    
    pos_idx = torch.arange(batch_pos).cuda()
        
    neg_idx = torch.cat([pos_idx for i in range(batch_pos - 1)], dim=0)
    neg_idx_shuffle = torch.cat([torch.cat([pos_idx[i+1:], pos_idx[0:i+1]], dim=0) for i in range(batch_pos - 1) ], dim=0)
    neg_idx = neg_idx[:batch_neg]
    neg_idx_shuffle = neg_idx_shuffle[:batch_neg]
        
    trainLoader_iter = iter(trainLoader) 
    
    for batch_id in range(iter_epoch):   
        idx = np.random.choice(len(trainLoader))
        
        try:
            batch = next(trainLoader_iter)
        except:
            trainLoader_iter[idx] = iter(trainLoader)
            batch = next(trainLoader_iter)
        
        ## put all into cuda
        T1 = batch['T1'].cuda()
        T2 = batch['T2'].cuda()
        
        
        RM1 = batch['RM1'].cuda()
        RM2 = batch['RM2'].cuda()
        
        M1 = batch['M1'].cuda()
        M2 = batch['M2'].cuda()
        
        xy1 = batch['xy1'].cuda()
        xy2 = batch['xy2'].cuda()
        
        
        optimizer.zero_grad()
        
        with torch.no_grad() : 
            T1 = F.normalize(backbone(T1), dim=1)
            T2 = F.normalize(backbone(T2), dim=1)

                
        if np.random.rand() > 0.5 :
            X, Y =  T1, T2
            MX, MY, RS, RT = M1, M2, RM1, RM2
            PosX, PosY = xy1, xy2
        else : 
            X, Y =  T2, T1
            MX, MY, RS, RT = M2, M1, RM2, RM1
            PosX, PosY = xy2, xy1
            
        O1, O2 = netEncoder(X, Y, RS, RT)
        
        if warp_mask : 
            flow12 = ((O1.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
            flow21 = ((O2.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
        

            O2_warp = F.grid_sample(O1.narrow(1, 2, 1), flow21, mode='bilinear')
            O2_warp = stable_clamp(O2_warp)
            O1_warp = F.grid_sample(O2.narrow(1, 2, 1), flow12, mode='bilinear')
            O1_warp = stable_clamp(O1_warp)
            
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1), O1_warp, O2_warp], dim=0)
            target_pos = torch.cat([MX, MY, MX, MY], dim=0)

        
        else :
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            target_pos = torch.cat([MX, MY], dim=0)
            
        with torch.no_grad() : 
            
            target_pos_binary = torch.cat([MX, MY], dim=0) 
            target_pos_view = target_pos_binary.view(target_pos_binary.size()[0], 1, 1, -1)
            target_pos_view = torch.max( target_pos_view, dim=3, keepdim=True)[0]
            
            target_pos_binary = (target_pos_binary == target_pos_view)
            target_pos_binary_float = target_pos_binary.type(torch.cuda.FloatTensor)

        
        output_xy = torch.cat([O1.narrow(1, 0, 2), O2.narrow(1, 0, 2)], dim=0)
        target_xy = torch.cat([PosX, PosY], dim=0)
        
        loss_corr = (torch.abs(target_xy - output_xy) * target_pos_binary_float ).sum() / target_pos_binary_float.sum() / 2
        
        O1_neg, O3_neg = netEncoder(T1[neg_idx], T2[neg_idx_shuffle])
        output_neg = torch.cat([O1_neg.narrow(1, 2, 1), O3_neg.narrow(1, 2, 1)], dim=0)
        target_neg = torch.zeros_like(output_neg)
        
        output = torch.cat((output_pos, output_neg), dim=0)
        target = torch.cat((target_pos, target_neg), dim=0)
        
        loss_mask = Loss(output, target)
        loss = loss_mask + eta_corr * loss_corr if loss_mask is not None else eta_corr * loss_corr
        
        loss.backward()
        optimizer.step()
        
        loss_log.append(loss.item())
        if loss_mask is not None : 
            loss_mask_log.append(loss_mask.item())
        else : 
            loss_mask_log.append(0)
        loss_corr_log.append(loss_corr.item())
        
        
        with torch.no_grad() : 
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            acc_pos = (((output_pos > 0.5) == target_pos_binary).type(torch.cuda.FloatTensor) * target_pos_binary_float).sum() / target_pos_binary_float.sum()
            acc_pos_log.append(acc_pos.item())

            element_neg = 1 - target_pos_binary_float
            acc_neg = ((((output_pos < 0.5) == (~target_pos_binary)).type(torch.cuda.FloatTensor) * element_neg).sum() + (output_neg < 0.5).type(torch.cuda.FloatTensor).sum()) / (element_neg.sum() + (1 - target_neg).sum())

            acc_neg_log.append(acc_neg.item())

            acc_log.append(acc_pos_log[-1] * 0.5 + acc_neg_log[-1] * 0.5)

        
        if batch_id % 100 == 99 : 
            for g in optimizer.param_groups:
                lr = g['lr']
                break
            msg = '{} Batch id {:d}, Lr {:.6f}; \t | Loss : {:.3f}, Mask : {:.3f}, Corr : {:.3f} |  Acc : {:.3f}%, Pos : {:.3f}%, Neg : {:.3f}%  \t '.format(datetime.now().time(), batch_id + 1, lr, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_corr_log), np.mean(acc_log) * 100, np.mean(acc_pos_log) * 100, np.mean(acc_neg_log) * 100)
            
            logger.info(msg)
            
    history['trainLoss'].append(np.mean(loss_log))
    history['trainLossMask'].append(np.mean(loss_mask_log))
    history['trainLossCorr'].append(np.mean(loss_corr_log))
    
    
    history['trainAcc'].append(np.mean(acc_log))
    history['trainPosAcc'].append(np.mean(acc_pos_log))
    history['trainNegAcc'].append(np.mean(acc_neg_log))
    
    
    return backbone, netEncoder, optimizer, history



def trainEpochHardMining(trainLoader, backbone, netEncoder, optimizer, history, Loss, batch_pos, batch_neg, warp_mask, logger, eta_corr, iter_epoch, NegaGenerator) : 
    
    backbone.eval()
    netEncoder.train()
    
    loss_log = []
    loss_mask_log = []
    loss_corr_log = []
    
    acc_log = []
    acc_pos_log = []
    acc_neg_log = []
    
    pos_idx = torch.arange(batch_pos).cuda()
    if batch_neg > 0 : 
        
        neg_idx = torch.cat([pos_idx for i in range(batch_pos - 1)], dim=0)
        neg_idx_shuffle = torch.cat([torch.cat([pos_idx[i+1:], pos_idx[0:i+1]], dim=0) for i in range(batch_pos - 1) ], dim=0)
        
        neg_idx = neg_idx[:batch_neg]
        neg_idx_shuffle = neg_idx_shuffle[:batch_neg]
        
    trainLoader_iter = iter(trainLoader)
    for batch_id in range(iter_epoch):   
        
        try:
            batch = next(trainLoader_iter)
            
        except:
            trainLoader_iter = iter(trainLoader)
            batch = next(trainLoader_iter)
        
        ## put all into cuda
        T1 = batch['T1'].cuda()
        T2 = batch['T2'].cuda()
        
        
        RM1 = batch['RM1'].cuda()
        RM2 = batch['RM2'].cuda()
        
        M1 = batch['M1'].cuda()
        M2 = batch['M2'].cuda()
        
        xy1 = batch['xy1'].cuda()
        xy2 = batch['xy2'].cuda()
        
        
        optimizer.zero_grad()
        
        with torch.no_grad() : 
            T1 = F.normalize(backbone(T1), dim=1)
            T2 = F.normalize(backbone(T2), dim=1)
                
        if np.random.rand() > 0.5 :
            X, Y =  T1, T2
            MX, MY, RS, RT = M1, M2, RM1, RM2
            PosX, PosY = xy1, xy2
        else : 
            X, Y =  T2, T1
            MX, MY, RS, RT = M2, M1, RM2, RM1
            PosX, PosY = xy2, xy1
            
        O1, O2 = netEncoder(X, Y, RS, RT)
        
        
        if warp_mask : 
            flow12 = ((O1.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
            flow21 = ((O2.narrow(1, 0, 2) - 0.5) * 2).permute(0, 2, 3, 1)
        
            O2_warp = F.grid_sample(O1.narrow(1, 2, 1), flow21, mode='bilinear')
            O2_warp = stable_clamp(O2_warp)
            O1_warp = F.grid_sample(O2.narrow(1, 2, 1), flow12, mode='bilinear')
            O1_warp = stable_clamp(O1_warp)
            
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1), O1_warp, O2_warp], dim=0)
            target_pos = torch.cat([MX, MY, MX, MY], dim=0)

        
        else :
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            target_pos = torch.cat([MX, MY], dim=0)
            
        with torch.no_grad() : 
            
            target_pos_binary = torch.cat([MX, MY], dim=0) 
            target_pos_view = target_pos_binary.view(target_pos_binary.size()[0], 1, 1, -1)
            target_pos_view = torch.max( target_pos_view, dim=3, keepdim=True)[0]
            
            target_pos_binary = (target_pos_binary == target_pos_view)
            target_pos_binary_float = target_pos_binary.type(torch.cuda.FloatTensor)

        
        output_xy = torch.cat([O1.narrow(1, 0, 2), O2.narrow(1, 0, 2)], dim=0)
        target_xy = torch.cat([PosX, PosY], dim=0)
        
        loss_corr = (torch.abs(target_xy - output_xy) * target_pos_binary_float ).sum() / target_pos_binary_float.sum() / 2
        
        output_neg = []
        if batch_neg > 0 :
            O1_neg, O3_neg = netEncoder(T1[neg_idx], T2[neg_idx_shuffle])
            output_neg.append( torch.cat([O1_neg.narrow(1, 2, 1), O3_neg.narrow(1, 2, 1)], dim=0) )
        
        TR = NegaGenerator.get_pair_batch()
        O1R_neg, O2R_neg = netEncoder(TR['T1'].cuda(), TR['T2'].cuda())
        output_neg.append( torch.cat([O1R_neg.narrow(1, 2, 1), O2R_neg.narrow(1, 2, 1)], dim=0) )
        output_neg = torch.cat(output_neg, dim=0)
        target_neg = torch.zeros_like(output_neg)
        
        output = torch.cat((output_pos, output_neg), dim=0)
        target = torch.cat((target_pos, target_neg), dim=0)
        
        loss_mask = Loss(output, target)
        loss = loss_mask + eta_corr * loss_corr if loss_mask is not None else eta_corr * loss_corr
        
        loss.backward()
        optimizer.step()
        
        loss_log.append(loss.item())
        if loss_mask is not None : 
            loss_mask_log.append(loss_mask.item())
        else : 
            loss_mask_log.append(0)
        loss_corr_log.append(loss_corr.item())
        
        with torch.no_grad() : 
            output_pos = torch.cat([O1.narrow(1, 2, 1), O2.narrow(1, 2, 1)], dim=0)
            acc_pos = (((output_pos > 0.5) == target_pos_binary).type(torch.cuda.FloatTensor) * target_pos_binary_float).sum() / target_pos_binary_float.sum()
            acc_pos_log.append(acc_pos.item())

            element_neg = 1 - target_pos_binary_float
            acc_neg = ((((output_pos < 0.5) == (~target_pos_binary)).type(torch.cuda.FloatTensor) * element_neg).sum() + (output_neg < 0.5).type(torch.cuda.FloatTensor).sum()) / (element_neg.sum() + (1 - target_neg).sum())

            acc_neg_log.append(acc_neg.item())

            acc_log.append(acc_pos_log[-1] * 0.5 + acc_neg_log[-1] * 0.5)
            
        
        if batch_id % 100 == 99 : 
            msg = '{} Batch id {:d}\t | Loss : {:.3f}, Mask : {:.3f}, Corr : {:.3f}|  Acc : {:.1f}%, Pos : {:.1f}%, Neg : {:.1f}%  \t '.format(datetime.now().time(), batch_id + 1, np.mean(loss_log), np.mean(loss_mask_log), np.mean(loss_corr_log), np.mean(acc_log) * 100, np.mean(acc_pos_log) * 100, np.mean(acc_neg_log) * 100)
            
            logger.info(msg)
            
    history['trainLoss'].append(np.mean(loss_log))
    history['trainLossMask'].append(np.mean(loss_mask_log))
    history['trainLossCorr'].append(np.mean(loss_corr_log))
    
    
    history['trainAcc'].append(np.mean(acc_log))
    history['trainPosAcc'].append(np.mean(acc_pos_log))
    history['trainNegAcc'].append(np.mean(acc_neg_log))
    
    
    
    return backbone, netEncoder, optimizer, history


