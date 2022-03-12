from collections import defaultdict
import numpy as np
import torch, os, logging, losses, json
from tqdm import tqdm
import torch.nn.functional as F
import math
from sklearn.manifold import TSNE
import plotly.graph_objects as go

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
    T : [num_samples] (target labels)
    Y : [num_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t, y in zip(T, Y):
        if t in y[:k]:
            s += 1
    return s / (1. * len(T))

def calc_MAP_at_r(T, Y):
    """
    T : [num_samples] (target labels)
    Y : [num_samples x r] (r predicted lables of query results)
    """

    n_queries = len(T)
    total = [0] * n_queries
    count = [0] * n_queries
    for i in range(n_queries):
        r = len(Y[i])
        for j in range(r):
            if T[i] == Y[i][j]:
                count[i] += 1
                total[i] += count[i] / (j + 1)
        total[i] /= r
    return sum(total) / n_queries

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
    num_classes = dataloader.dataset.nb_classes()
    X, T, *_ = predict_batchwise(model, dataloader)

    proxy_mean = torch.stack([X[T==class_idx].mean(dim=0) for class_idx in range(num_classes)])

    return proxy_mean

def evaluate_cos(model, dataloader, calcMAP=False):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest K neighbors with cosine
    K = 32
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]].cpu()
    
    result = []
    for k in [2**x for x in range(6)]:
        r_at_k = calc_recall_at_k(T, Y, k)
        result.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if calcMAP:
        Y = []
        t_cnt = defaultdict(int)
        for t in T:
            t_cnt[t.item()] += 1
        for i in range(len(T)):
            Y.append(T[cos_sim[i].topk(t_cnt[T[i].item()])[1][1:]])
    
        MAP_at_r = calc_MAP_at_r(T, Y)
        result.append(MAP_at_r)
        print("MAP@R : {:.3f}".format(100 * MAP_at_r))

    return result

def evaluate_cos_Inshop(model, query_dataloader, gallery_dataloader, calcMAP=False):
    # calculate embeddings with model and get targets
    query_X, query_T = predict_batchwise(model, query_dataloader)
    gallery_X, gallery_T = predict_batchwise(model, gallery_dataloader)
    
    query_X = l2_norm(query_X)
    gallery_X = l2_norm(gallery_X)
    
    # get predictions by assigning nearest K neighbors with cosine
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

    # calculate recall@k / MAP@r
    result = []
    for k in [1, 10, 20, 30, 40, 50]:
        r_at_k = recall_k(cos_sim, query_T, gallery_T, k)
        result.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if calcMAP:
        Y = []
        t_cnt = defaultdict(int)
        for t in gallery_T:
            t_cnt[t.item()] += 1
        for i in range(len(query_T)):
            Y.append(gallery_T[cos_sim[i].topk(t_cnt[query_T[i].item()])[1]])

        MAP_at_r = calc_MAP_at_r(query_T, Y)
        result.append(MAP_at_r)
        print("MAP@R : {:.3f}".format(100 * MAP_at_r))
        
    return result

def evaluate_cos_SOP(model, dataloader, calcMAP=False):
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader)
    X = l2_norm(X)

    # get predictions by assigning nearest K neighbors with cosine
    K = 1000
    Y = []
    if calcMAP:
        Y_MAP = []
        t_cnt = defaultdict(int)
        for t in T:
            t_cnt[t.item()] += 1

    step = 10000
    for i in range(0, len(T), step):
        end = min(i + step, len(T))
        cos_sim = F.linear(X[i:end], X)
        Y.append(T[cos_sim.topk(1 + K)[1][:, 1:]].cpu())
        if calcMAP:
            m = len(cos_sim)
            for j in range(m):
                Y_MAP.append(T[cos_sim[j].topk(t_cnt[T[end - m + j].item()])[1][1:]])

    # calculate recall@k / MAP@r
    Y = torch.cat(Y, dim=0)
    result = []
    for k in [10**x for x in range(4)]:
        r_at_k = calc_recall_at_k(T, Y, k)
        result.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    if calcMAP:
        MAP_at_r = calc_MAP_at_r(T, Y_MAP)
        result.append(MAP_at_r)
        print("MAP@R : {:.3f}".format(100 * MAP_at_r))
    
    return result

def tSNEPlot(model, dataloader, n_components=2, proxy=None, showProxyId=False, savePath=None, epoch=None):
    if n_components < 2 or n_components > 3:
        raise ValueError('n_components=%s is not supported' % n_components)

    X, T = predict_batchwise(model, dataloader)
    maxId = T.max().item()
    if proxy is not None:
        X = torch.cat((X, l2_norm(proxy)))
        T = torch.cat((T, torch.ones(proxy.shape[0], dtype=T.dtype) * (maxId + 1)))

    if savePath is not None:
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        torch.save({'embedding_vectors': X, 'labels': T}, '{}/embeddings_at_epoch{}.pth'.format(savePath, epoch))

    dist_mat = torch.cdist(X, X)
    projections = TSNE(n_components=n_components, random_state=0, metric='precomputed').fit_transform(dist_mat.cpu())
    fig = go.Figure()

    for t in range(T.min().item(), maxId + 1):
        if n_components == 2:
            fig.add_trace(go.Scatter(x=projections[T == t, 0],
                                     y=projections[T == t, 1],
                                     mode='markers',
                                     marker=dict(line_width=0, symbol='circle'),
                                     name='class{}'.format(t)))
        else:
            fig.add_trace(go.Scatter3d(x=projections[T == t, 0],
                                       y=projections[T == t, 1],
                                       z=projections[T == t, 2],
                                       mode='markers',
                                       marker=dict(line_width=0, symbol='circle'),
                                       name='class{}'.format(t)))

    if proxy is not None:
        if showProxyId:
            for id in range(proxy.shape[0]):
                if n_components == 2:
                    fig.add_trace(go.Scatter(x=[projections[T == maxId + 1, 0][id]],
                                             y=[projections[T == maxId + 1, 1][id]],
                                             mode='markers',
                                             marker=dict(line_width=2, symbol='star-diamond-open'),
                                             name='proxy{}'.format(id),
                                             marker_color='rgba(0, 0, 100, .8)'))
                else:
                    fig.add_trace(go.Scatter3d(x=[projections[T == maxId + 1, 0][id]],
                                               y=[projections[T == maxId + 1, 1][id]],
                                               z=[projections[T == maxId + 1, 2][id]],
                                               mode='markers',
                                               marker=dict(line_width=2, symbol='diamond-open'),
                                               name='proxy{}'.format(id),
                                               marker_color='rgba(0, 0, 100, .8)'))
        else:
            if n_components == 2:
                fig.add_trace(go.Scatter(x=projections[T == maxId + 1, 0],
                                         y=projections[T == maxId + 1, 1],
                                         mode='markers',
                                         marker=dict(line_width=2, symbol='star-diamond-open'),
                                         name='proxies',
                                         marker_color='rgba(0, 0, 100, .8)'))
            else:
                fig.add_trace(go.Scatter3d(x=projections[T == maxId + 1, 0],
                                           y=projections[T == maxId + 1, 1],
                                           z=projections[T == maxId + 1, 2],
                                           mode='markers',
                                           marker=dict(line_width=2, symbol='diamond-open'),
                                           name='proxies',
                                           marker_color='rgba(0, 0, 100, .8)'))

    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=140, r=40, b=50, t=80),
        legend=dict(font=dict(family='Helvetica', size=16, color='black')),
        width=800,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig