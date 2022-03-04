# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import logging
import os
import torch
import numpy as np
import torch.nn.functional as F
from data.datasets.eval_reid import evaluate
from data.prefetcher import data_prefetcher, data_prefetcher_mask
import random
import cv2 

# Changed by Xinchen Liu

def L_Matrix(adj_npy, adj_size):

    D =np.zeros((adj_size, adj_size))
    for i in range(adj_size):
        tmp = adj_npy[i,:]
        count = np.sum(tmp==1)
        if count>0:
            number = count ** (-1/2)
            D[i,i] = number

    x = np.matmul(D,adj_npy)
    L = np.matmul(x,D)
    return L


coarse_adj_list = [
    # 1  2  3  4  5  6  7  8  9
    [ 1, 1, 0, 1, 0, 1, 0, 1, 0], #1
    [ 1, 1, 1, 1, 0, 1, 0, 0, 0], #2
    [ 0, 1, 1, 0, 1, 0, 1, 0, 0], #3
    [ 1, 1, 0, 1, 1, 0, 0, 1, 0], #4
    [ 0, 0, 1, 1, 1, 0, 0, 0, 1], #5
    [ 1, 1, 0, 0, 0, 1, 1, 1, 0], #6
    [ 0, 0, 1, 0, 0, 1, 1, 0, 1], #7
    [ 1, 0, 0, 1, 0, 1, 0, 1, 1], #8
    [ 0, 0, 0, 0, 1, 0, 1, 1, 1]  #9
]
coarse_adj_npy = np.array(coarse_adj_list)
coarse_adj_npy = L_Matrix(coarse_adj_npy, len(coarse_adj_npy))
obj_adj_list = [
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,0,1,0,0],
    [1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,0,0,1],
    [1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1],
    [1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,0,1,0,1,0,1,0,1],
    [1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,1],
    [0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]
]
obj_adj_npy = np.array(obj_adj_list)
obj_adj_npy = L_Matrix(obj_adj_npy, len(obj_adj_npy))

def inference(
        cfg,
        model,
        test_dataloader_collection,
        num_query_collection,
        is_vis=False,
        test_collection=None,
        use_mask=True,
        num_parts=10,
        mask_image=False
):
    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Start inferencing")

    model.eval()

    adj = torch.from_numpy(coarse_adj_npy).float()
    obj = torch.from_numpy(obj_adj_npy).float()
    idx = -1
    for test_dataset_name, test_dataloader, num_query in zip(cfg.DATASETS.TEST_NAMES, test_dataloader_collection, num_query_collection):
        idx += 1
        feats1,feats2, pids, camids = [],[], [], []
        if use_mask:
            test_prefetcher = data_prefetcher_mask(test_dataloader)
        else:
            test_prefetcher = data_prefetcher(test_dataloader)
        batch = test_prefetcher.next()
        while batch[0] is not None:
            if use_mask:
                img, mask,xml, pid, camid = batch
                adj_batch = adj.repeat(img.size(0), 1, 1)
                obj_batch = obj.repeat(img.size(0), 1, 1)
            #print(xml.sum())
            #print(mask.sum())
            #print(pid)
            with torch.no_grad():
                output = model(img, img, mask,xml,adj_batch,obj_batch)
#                 feat = output[1]
#                 feat = output[3]
                feat1 = torch.cat([output[1], output[3]], dim=1)
                feat2 = output[9]
            feats1.append(feat1)
            feats2.append(feat2)
            pids.extend(pid.cpu().numpy())
            camids.extend(np.asarray(camid))

            batch = test_prefetcher.next()

        feats1 = torch.cat(feats1, dim=0)
        feats2 = torch.cat(feats2, dim=0)
        if cfg.TEST.NORM:
            feats1 = F.normalize(feats1, p=2, dim=1)
            feats2 = F.normalize(feats2, p=2, dim=1)
        # query
        qf1 = feats1[:num_query]
        qf2 = feats2[:num_query]
        q_pids = np.asarray(pids[:num_query])
        q_camids = np.asarray(camids[:num_query])
        # gallery
        gf1 = feats1[num_query:]
        gf2 = feats2[num_query:]
        g_pids = np.asarray(pids[num_query:])
        g_camids = np.asarray(camids[num_query:])
        distmat1 = torch.mm(qf1, gf1.t()).cpu().numpy()
        distmat2 = torch.mm(qf2, gf2.t()).cpu().numpy()
        for k in np.arange(0, 1, 0.1):
            logger.info(f"{k}")
            distmat = distmat1 +k* distmat2
                    # distmat = distmat1+k*distmat2
            cmc, mAP = evaluate(-distmat, q_pids, g_pids, q_camids, g_camids)
            logger.info(f"Results on {test_dataset_name} : ")
            logger.info(f"mAP: {mAP:.1%}")
            for r in [1, 5, 10]:
                logger.info(f"CMC curve, Rank-{r:<3}:{cmc[r - 1]:.1%}")
                #self.writer.add_scalar('rank1', cmc[0], self.global_step)
                #self.writer.add_scalar('mAP', mAP, self.global_step)
                #metric_dict.append({'rank1': cmc[0], 'mAP': mAP})
        if is_vis:
            query_rand = 10
            topK = 10
            is_save_all = True
            query_rand_idx = range(0, num_query) if is_save_all else random.sample(range(0, num_query), query_rand)
            print(f'|-------- Randomly saving top-{topK} results of {len(query_rand_idx)} queries for {test_dataset_name} --------|')
            qf_rand = qf[query_rand_idx]
            q_pids_rand = q_pids[query_rand_idx]
            q_camids_rand = q_camids[query_rand_idx]
            
            q_items = test_collection[idx][:num_query]
            q_items_rand = list()
            for i in query_rand_idx:
                q_items_rand.append(q_items[i])
            g_items = test_collection[idx][num_query:]
            
            distmat_rand = torch.mm(qf_rand, gf.t()).cpu().numpy()
            distmat_rand = -distmat_rand
            
            indices = np.argsort(distmat_rand, axis=1)
            matches = (g_pids[indices] == q_pids_rand[:, np.newaxis]).astype(np.int32)
            
            save_img_size = (256, 256)
            
            for q_idx in range(len(query_rand_idx)):
                savefilename = ''
                # get query pid and camid
                q_path = q_items_rand[q_idx][0]
                q_pid = q_items_rand[q_idx][1]
                q_camid = q_items_rand[q_idx][2]
                
                savefilename += 'q-'+q_path.split('/')[-1]+'_g'

                # remove gallery samples that have the same pid and camid with query
                order = indices[q_idx]
                remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
                keep = np.invert(remove)

                print('Query Path : ', q_path)
                print('Result idx : ', order[:topK])
                
                img_list = list()
                q_img = cv2.imread(q_path)
                q_img = cv2.resize(q_img, save_img_size)
                cv2.rectangle(q_img, (0,0), save_img_size, (255,0,0), 4)
                img_list.append(q_img)
                
                for g_idx in order[:topK]:
                    g_img = cv2.imread(g_items[g_idx][0])
                    g_img = cv2.resize(g_img, save_img_size)
                    if q_pid == g_items[g_idx][1] and q_camid == g_items[g_idx][2]:
                        cv2.rectangle(g_img, (0,0), save_img_size, (255,255,0), 4)
                    elif q_pid == g_items[g_idx][1] and q_camid != g_items[g_idx][2]:
                        cv2.rectangle(g_img, (0,0), save_img_size, (0,255,0), 4)
                    else:
                        cv2.rectangle(g_img, (0,0), save_img_size, (0,0,255), 4)
                    img_list.append(g_img)
                    savefilename += '-'+str(g_items[g_idx][3])
                
                pic = np.concatenate(img_list, 1)
                picsavedir = os.path.join(cfg.OUTPUT_DIR, '-'.join(cfg.DATASETS.TEST_NAMES), 'examples', test_dataset_name)
                if not os.path.exists(picsavedir): os.makedirs(picsavedir)
                savefilepath = os.path.join(picsavedir, savefilename+'.jpg')
                cv2.imwrite(savefilepath, pic)
                print('Save example picture to ', savefilepath)
