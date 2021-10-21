import numpy as np 
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix, coo_matrix, diags

import torch 
from itertools import combinations, product
from tqdm import tqdm 
import time 
from scipy.sparse.linalg import eigs, eigsh

import pickle 
import os 
import matplotlib.pyplot as plt

    
def ComputeSimiDecayNeigh(pt1, pt2, sigma_ctxt) : 
    '''
    Parameters : 
        
        pt1      (N, 2): pts in the 1st image
        pt2      (N, 2): pts in the 2nd image
        eta_ctxt   float: context threshold
        
        
    Outputs : 
        coef_ctxt (N, N) : pair_wise similarities
    '''
    
    ## Pairwise distance in pixel level 
    distPt1 = pt1[:, None] - pt1[None, :]
    distPt2 = pt2[:, None] - pt2[None, :]
    
    neigh1_tensor = np.sum(distPt1 ** 2, axis=2) ** 0.5 
    neigh2_tensor = np.sum(distPt2 ** 2, axis=2) ** 0.5 
    
    coef_neigh = np.exp(- (neigh1_tensor + neigh2_tensor) / 2 / sigma_ctxt)
    
    return coef_neigh


def ComputeTripletDecayNeigh(pt1, pt2, sigma_ctxt) : 
    '''
    Parameters : 
        
        pt1      (N1, 2): correspondences in the 1st image
        pt2      (N2, 2): correspondences in the same image
        
    Outputs : 
        coef_ctxt (N1, N2) : pair_wise similarities
    '''
    
    ## Pairwise distance in pixel level 
    distPt = pt1[:, None] - pt2[None, :]
    
    neigh_tensor = np.sum(distPt ** 2, axis=2) ** 0.5 
    
    coef_neigh = np.exp(- neigh_tensor / sigma_ctxt ) 
    
    return coef_neigh


def candidate_L2norm_select(sol, index) : 
    candidate = np.zeros_like(sol)
    candidate[index] = 1
    candidate = candidate / candidate.sum() ** 0.5
    candidate = candidate.reshape(1, -1)
    return candidate


def energy_normL2_eta_sparse_grid(A_sparse, sol, sol_sort_neg, sol_sort_pos, grid) : 
    
    ## positive part
    
    nb_sample = len(grid)
    pos = np.zeros(nb_sample)
    neg = np.zeros(nb_sample)
    
    for i in range(nb_sample - 1) : 
        
        idx = grid[i]
        candidate = candidate_L2norm_select(sol, sol_sort_neg[:idx+1])
        neg[i] = candidate @ (A_sparse.dot(candidate.T))
            
        candidate = candidate_L2norm_select(sol, sol_sort_pos[:idx+1])
        pos[i] = candidate @ (A_sparse.dot(candidate.T))
        
    candidate = candidate_L2norm_select(sol, sol_sort_neg)
    neg[-1] = candidate @ (A_sparse.dot(candidate.T))
        
    pos[-1] = neg[-1]
        
    return pos, neg   


def TopEigen(graph, space_log) : 

    value_p, vec_p = eigsh(graph, k=1, which='LM', v0 = np.ones(graph.shape[0]))
        
    sol = np.real(vec_p[:, 0])
    
    sol_sort_neg = np.argsort(sol)
    sol_sort_pos = sol_sort_neg[::-1]
    
    nb_sample = graph.shape[0]
    sample_grid = []

    pts = 1
    count = 1
    while pts < nb_sample : 
        if pts - 1 not in sample_grid : 
            sample_grid.append(pts - 1)
        pts = int(round(space_log ** count))
        count += 1

    sample_grid.append(nb_sample - 1)
    pos_score, neg_score = energy_normL2_eta_sparse_grid(graph, sol, sol_sort_neg, sol_sort_pos, sample_grid)
    
    if pos_score.max() >= neg_score.max() : 
        return sol
    else : 
        return -sol
    

def OptimFullGraph(corr,
                   img_idx_start_end,
                   sigma,
                   pair_idx,
                   keep_idx,
                   out1x, 
                   out1y, 
                   out1m, 
                   out2x, 
                   out2y, 
                   out2m,
                   mask_in_3cycle=False,
                   only3cycle = False) : 
    
    
    nb_node = len(corr)
    #print ('Nb of nodes: {:d}'.format(nb_node))
    nb_pair  = len(img_idx_start_end)
    pair_list = [(img_idx_start_end[i, 0], img_idx_start_end[i, 1]) for i in range(nb_pair)]
    #print ('Nb of pairs: {:d}'.format(nb_pair))
    
        
    #print ('Building the graph...')
    data = []
    row = []
    column = []
    #print ('\t 2-cycle...')
    if not only3cycle: 
        for i in range(nb_pair) : 
            start, end = img_idx_start_end[i, 2], img_idx_start_end[i, 3]
            A = ComputeSimiDecayNeigh(corr[start : end, :2], corr[start : end, 2:4], sigma)
            conf = corr[start : end, 4]
            A = A * conf[:, None] * conf[None, :]
            idx_row, idx_column = np.where(A > 0.01)
            data.append(A[idx_row, idx_column])
            row.append(idx_row + start)
            column.append(idx_column + start)

            
    #print ('\t 3-cycle...')
    all_pairs = {(pair_idx[keep_idx[i], 0], pair_idx[keep_idx[i], 1]) : i for i in range(len(keep_idx))}
    
    count = 0
    for key1, key2 in combinations(range(nb_pair), 2) : 

        i,j = img_idx_start_end[key1, :2]
        m,n = img_idx_start_end[key2, :2]

        set_key1 = set([i,j])
        set_key2 = set([m,n])

        if len(set_key1.intersection(set_key2)) == 0 :
            continue

        start1 = img_idx_start_end[key1, 2]
        end1 = img_idx_start_end[key1, 3]

        start2 = img_idx_start_end[key2, 2]
        end2 = img_idx_start_end[key2, 3]

        ## (i,j), (i,n)
        if i == m :#and ((j, n) in pair_list or (n, j) in pair_list): 
             
            A = ComputeTripletDecayNeigh(corr[start1 : end1, :2], corr[start2 : end2, :2], sigma)

            if (j,n) not in all_pairs and (n, j) not in all_pairs : 
                continue
            else :
                pts_j = corr[start1 : end1, 2:4].astype(np.int32)
                pts_n = corr[start2 : end2, 2:4].astype(np.int32)

                if (j,n) in all_pairs : 
                    idx_jn = all_pairs[(j,n)]
                    match_j_n_x = out1x[idx_jn][pts_j[:, 1], pts_j[:, 0]]
                    match_j_n_y = out1y[idx_jn][pts_j[:, 1], pts_j[:, 0]]
                    mask_j = out1m[idx_jn][pts_j[:, 1], pts_j[:, 0]].reshape((-1, 1))
                    
                    match_n_j_x = out2x[idx_jn][pts_n[:, 1], pts_n[:, 0]]
                    match_n_j_y = out2y[idx_jn][pts_n[:, 1], pts_n[:, 0]]
                    mask_n = out2m[idx_jn][pts_n[:, 1], pts_n[:, 0]].reshape((1, -1))
                    

                elif (n, j) in all_pairs : 
                    idx_nj = all_pairs[(n,j)]
                    match_j_n_x = out2x[idx_nj][pts_j[:, 1], pts_j[:, 0]]
                    match_j_n_y = out2y[idx_nj][pts_j[:, 1], pts_j[:, 0]]
                    mask_j = out2m[idx_nj][pts_j[:, 1], pts_j[:, 0]].reshape((-1, 1))
                    
                    match_n_j_x = out1x[idx_nj][pts_n[:, 1], pts_n[:, 0]]
                    match_n_j_y = out1y[idx_nj][pts_n[:, 1], pts_n[:, 0]]
                    mask_n = out1m[idx_nj][pts_n[:, 1], pts_n[:, 0]].reshape((1, -1))

                match_j_n = np.stack([match_j_n_x, match_j_n_y], axis=1)
                A3cycle_n = ComputeTripletDecayNeigh(match_j_n, pts_n, sigma)
                match_n_j = np.stack([match_n_j_x, match_n_j_y], axis=1)
                A3cycle_j = ComputeTripletDecayNeigh(pts_j, match_n_j, sigma)
                if mask_in_3cycle : 
                    
                    A = A * (A3cycle_n * mask_j + A3cycle_j * mask_n) * 0.5
                    
                else :
                    A = A * (A3cycle_n + A3cycle_j) * 0.5
                    
            
        elif j == m :#and ((i, n) in pair_list or (n, i) in pair_list): 
            
                A = ComputeTripletDecayNeigh(corr[start1 : end1, 2:4], corr[start2 : end2, :2], sigma)
                
                if (i,n) not in all_pairs and (n,i) not in all_pairs : 
                    continue
                else :
                    pts_i = corr[start1 : end1, 0:2].astype(np.int32)
                    pts_n = corr[start2 : end2, 2:4].astype(np.int32)

                    if (i,n) in all_pairs : 
                        idx_in = all_pairs[(i,n)]
                        match_i_n_x = out1x[idx_in][pts_i[:, 1], pts_i[:, 0]]
                        match_i_n_y = out1y[idx_in][pts_i[:, 1], pts_i[:, 0]]
                        mask_i = out1m[idx_in][pts_i[:, 1], pts_i[:, 0]].reshape((-1, 1))
                        
                        match_n_i_x = out2x[idx_in][pts_n[:, 1], pts_n[:, 0]]
                        match_n_i_y = out2y[idx_in][pts_n[:, 1], pts_n[:, 0]]
                        mask_n = out2m[idx_in][pts_n[:, 1], pts_n[:, 0]].reshape((1, -1))

                    elif (n,i) in all_pairs : 
                        idx_ni = all_pairs[(n,i)]
                        match_i_n_x = out2x[idx_ni][pts_i[:, 1], pts_i[:, 0]]
                        match_i_n_y = out2y[idx_ni][pts_i[:, 1], pts_i[:, 0]]
                        mask_i = out2m[idx_ni][pts_i[:, 1], pts_i[:, 0]].reshape((-1, 1))
                        
                        match_n_i_x = out1x[idx_ni][pts_n[:, 1], pts_n[:, 0]]
                        match_n_i_y = out1y[idx_ni][pts_n[:, 1], pts_n[:, 0]]
                        mask_n = out1m[idx_ni][pts_n[:, 1], pts_n[:, 0]].reshape((1, -1))

                    match_i_n = np.stack([match_i_n_x, match_i_n_y], axis=1)
                    A3cycle_n = ComputeTripletDecayNeigh(match_i_n, pts_n, sigma)
                    match_n_i = np.stack([match_n_i_x, match_n_i_y], axis=1)
                    A3cycle_i = ComputeTripletDecayNeigh(pts_i, match_n_i, sigma)
                    
                    if mask_in_3cycle : 
                    
                        A = A * (A3cycle_n * mask_i + A3cycle_i * mask_n) * 0.5
                    
                    else :
                        A = A * (A3cycle_n + A3cycle_i) * 0.5
                    
                
            
        elif j == n :#and ((i, m) in pair_list or (m, i) in pair_list): 
            
                A = ComputeTripletDecayNeigh(corr[start1 : end1, 2:4], corr[start2 : end2, 2:4], sigma)
                
                if (i,m) not in all_pairs and (m,i) not in all_pairs : 
                    continue
                else :
                    pts_i = corr[start1 : end1, 0:2].astype(np.int32)
                    pts_m = corr[start2 : end2, 0:2].astype(np.int32)

                    if (i,m) in all_pairs : 
                        idx_im = all_pairs[(i,m)]
                        match_i_m_x = out1x[idx_im][pts_i[:, 1], pts_i[:, 0]]
                        match_i_m_y = out1y[idx_im][pts_i[:, 1], pts_i[:, 0]]
                        mask_i = out1m[idx_im][pts_i[:, 1], pts_i[:, 0]].reshape((-1, 1))
                        
                        match_m_i_x = out2x[idx_im][pts_m[:, 1], pts_m[:, 0]]
                        match_m_i_y = out2y[idx_im][pts_m[:, 1], pts_m[:, 0]]
                        mask_m = out2m[idx_im][pts_m[:, 1], pts_m[:, 0]].reshape((1, -1))
                        

                    elif (m,i) in all_pairs : 
                        idx_mi = all_pairs[(m,i)]
                        match_i_m_x = out2x[idx_mi][pts_i[:, 1], pts_i[:, 0]]
                        match_i_m_y = out2y[idx_mi][pts_i[:, 1], pts_i[:, 0]]
                        mask_i = out2m[idx_mi][pts_i[:, 1], pts_i[:, 0]].reshape((-1, 1))
                        
                        match_m_i_x = out1x[idx_mi][pts_m[:, 1], pts_m[:, 0]]
                        match_m_i_y = out1y[idx_mi][pts_m[:, 1], pts_m[:, 0]]
                        mask_m = out1m[idx_mi][pts_m[:, 1], pts_m[:, 0]].reshape((1, -1))
                        

                    match_i_m = np.stack([match_i_m_x, match_i_m_y], axis=1)
                    A3cycle_m = ComputeTripletDecayNeigh(match_i_m, pts_m, sigma)
                    match_m_i = np.stack([match_m_i_x, match_m_i_y], axis=1)
                    A3cycle_i = ComputeTripletDecayNeigh(pts_i, match_m_i, sigma)
                    
                    if mask_in_3cycle : 
                    
                        A = A * (A3cycle_m * mask_i + A3cycle_i * mask_m) * 0.5
                    
                    else :
                        A = A * (A3cycle_m + A3cycle_i) * 0.5
                    
            
        else : 
            continue
            
        
        
        conf1 = corr[start1 : end1, 4]
        conf2 = corr[start2 : end2, 4]
        
        A = A * conf1[:, None] * conf2[None, :]
         
        idx_row, idx_column = np.where(A > 0.01)
        data.append(A[idx_row, idx_column])
        row.append(idx_row + start1)
        column.append(idx_column + start2)

        data.append(A[idx_row, idx_column])
        row.append(idx_column + start2)
        column.append(idx_row + start1)
                            
        
    cluster_id = 0
    graph = coo_matrix((np.concatenate(data, axis=0).astype(np.float32), (np.concatenate(row, axis=0).astype(np.int64), np.concatenate(column, axis=0).astype(np.int64))), shape=(nb_node, nb_node))

    subgraph = graph.tocsr().tocsc()
    eigen_vector  = TopEigen(subgraph, space_log=1.1)
    
    
    del graph
    return eigen_vector