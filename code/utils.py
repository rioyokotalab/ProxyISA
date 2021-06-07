import numpy as np
import torch
import logging
import losses
import json
from tqdm import tqdm
import torch.nn.functional as F
import math

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))


def predict_batchwise(model, dataloader):
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.cuda())

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def proxy_init_calc(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest K neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    recall = []
    for k in [2**x for x in range(6)]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recall.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recall

def calc_map_at_r(T, Y, r):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    n_samples = len(T)
    total = [0] * n_samples
    count = [0] * n_samples
    for i in range(r):
        for j in range(n_samples):
            if T[j] == torch.Tensor(Y[j]).long()[i]:
                count[j] += 1
                total[j] += count[j] / (i + 1)
    return sum(total) / (r * n_samples)

def evaluate_cos_map(model, dataloader):
    nb_classes = dataloader.dataset.nb_classes()

    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest K neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y = Y.float().cpu()
    
    map_list = []
    for r in [2**x for x in range(6)]:
        map_at_r = calc_map_at_r(T, Y, r)
        map_list.append(map_at_r)
        print("MAP@{} : {:.3f}".format(r, 100 * map_at_r))

    return map_list

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader, isMap=False):
    nb_classes = query_dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 50
    Y = []
    xs = []
    
    cos_sim = F.linear(query_X, gallery_X)

    def recall_k(cos_sim, query_T, gallery_T, k):
        m = len(cos_sim)
        match_counter = 0

        for i in range(m):
            pos_sim = cos_sim[i][gallery_T == query_T[i]]
            neg_sim = cos_sim[i][gallery_T != query_T[i]]

            thresh = torch.max(pos_sim).item()

            if torch.sum(neg_sim > thresh) < k:
                match_counter += 1
            
        return match_counter / m
    
    def map_r(cos_sim, query_T, gallery_T, r):
        m = len(cos_sim)
        total = [0] * m
        count = [0] * m

        for i in range(r):
            for j in range(m):
                _, index = torch.sort(cos_sim[j], descending=True)

                if gallery_T[index[i]] == query_T[j]:
                    count[j] += 1
                    total[j] += count[j] / (i + 1)
            
        return sum(total) / (r * m)

    # calculate recall@k / map@r
    result = []
    if isMap:
        for r in [1, 10, 20, 30, 40, 50]:
            m_at_r = map_r(cos_sim, query_T, gallery_T, r)
            result.append(m_at_r)
            print("MAP@{} : {:.3f}".format(r, 100 * m_at_r))
    else:
        for k in [1, 10, 20, 30, 40, 50]:
            r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
            result.append(r_at_k)
            print("R@{} : {:.3f}".format(k, 100 * r_at_k))
                
    return result

def evaluate_cos_SOP(model, dataloader, isMap=False):
    nb_classes = dataloader.dataset.nb_classes()
    
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)
    
    # get predictions by assigning nearest 8 neighbors with cosine
    K = 1000
    Y = []
    xs = []
    for x in X:
        if len(xs)<10000:
            xs.append(x)
        else:
            xs.append(x)            
            xs = torch.stack(xs,dim=0)
            cos_sim = F.linear(xs,X)
            y = T[cos_sim.topk(1 + K)[1][:,1:]]
            Y.append(y.float().cpu())
            xs = []
            
    # Last Loop
    xs = torch.stack(xs,dim=0)
    cos_sim = F.linear(xs,X)
    y = T[cos_sim.topk(1 + K)[1][:,1:]]
    Y.append(y.float().cpu())
    Y = torch.cat(Y, dim=0)

    # calculate recall@k / map@r
    result = []
    if isMap:
        for r in [10**x for x in range(4)]:
            map_at_r = calc_map_at_r(T, Y, r)
            result.append(map_at_r)
            print("MAP@{} : {:.3f}".format(r, 100 * map_at_r))
    else:
        for k in [10**x for x in range(4)]:
            r_at_k = calc_recall_at_k(T, Y, k)
            result.append(r_at_k)
            print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return result

def calcSVD(X_grad):
    u, s, vh = torch.linalg.svd(X_grad, full_matrices=False)
    return s