import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from pytorch_metric_learning import miners, losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def get_distance(x, p):
    """Helper function for proxy-anchor loss. Return a distance matrix given 2 matrices."""
    m = p.shape[0]
    n = x.shape[0]

    p_square = torch.sum(p ** 2.0, dim=1).reshape(m, -1)
    tmp = p_square
    for i in range(n-1):
        p_square = torch.hstack((p_square, tmp))

    x_square = torch.sum(x ** 2.0, dim=1).reshape(n, -1)
    tmp = x_square
    for i in range(m-1):
        x_square = torch.hstack((x_square, tmp))

    distance_square = x_square + p_square.t() - (2.0 * torch.matmul(x, p.t()))

    return torch.sqrt(distance_square)

class ProxyAnchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Initialization
        self.proxies = torch.nn.Parameter(torch.Tensor(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        proxy_l2 = l2_norm(self.proxies)

        n, d = X.shape

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        cos = F.linear(l2_norm(X), proxy_l2)  # Calcluate cosine similarity
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class ProxyNCA(torch.nn.Module):
    def __init__(self, numClasses, sizeEmbed, scale=12.0, init_type='normal'):
        super(ProxyNCA, self).__init__()
        self.proxies = nn.Parameter(torch.Tensor(numClasses, sizeEmbed).cuda())
        self.n_classes = numClasses
        self.scale = scale
        if init_type == 'normal':
            nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        elif init_type == 'uniform':
            nn.init.kaiming_uniform_(self.proxies, a=math.sqrt(5))
        else:
            raise ValueError('%s not supported' % init_type)

    def forward(self, input, target):
        P = self.proxies

        # input already l2_normalized
        proxy_l2 = F.normalize(P, p=2, dim=1)

        # N, dim, cls
        sim_mat = F.linear(input, proxy_l2) * self.scale

        pos_target = F.one_hot(target, self.n_classes).float()
        neg_target = 1 - pos_target

        loss = torch.mean(-sim_mat[torch.arange(0, input.shape[0]), target] + torch.log(torch.sum(neg_target * torch.exp(sim_mat), -1)))

        # return loss if loss >= 0 else torch.zeros([], requires_grad=True).cuda()
        return loss
    
class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_neg=50.0, hard_mining=True):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = 0.1

        self.scale_pos = 2.0
        self.scale_neg = scale_neg
        self.hard_mining = hard_mining

    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        batch_size = inputs_col.size(0)
        sim_mat = F.linear(l2_norm(inputs_col), l2_norm(inputs_row))

        epsilon = 1e-5
        loss = list()
        neg_count = 0
        for i in range(batch_size):
            pos_pair_ = torch.masked_select(sim_mat[i], target_row == targets_col[i])
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], target_row != targets_col[i])

            # sampling step
            if self.hard_mining and len(pos_pair_) >= 1:
                neg_pair = neg_pair_[neg_pair_ + self.margin > torch.min(pos_pair_)]
                pos_pair = pos_pair_[pos_pair_ - self.margin < torch.max(neg_pair_)]
            else:
                pos_pair = pos_pair_
                neg_pair = neg_pair_

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = (
                1.0
                / self.scale_pos
                * torch.log(
                    1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh)))
                )
            )
            neg_loss = (
                1.0
                / self.scale_neg
                * torch.log(
                    1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh)))
                )
            )
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).cuda()
        loss = sum(loss) / batch_size
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = F.linear(l2_norm(inputs_col), l2_norm(inputs_row))
        epsilon = 1e-5
        loss = list()

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets_col[i] == target_row)
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            if len(pos_pair_) < 1:
                continue
            neg_pair_ = torch.masked_select(sim_mat[i], targets_col[i] != target_row)

            neg_pair = torch.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = torch.sum(1 - pos_pair_)
            if len(neg_pair) > 0:
                neg_loss = torch.sum(neg_pair)
                # neg_count.append(len(neg_pair))
            else:
                neg_loss = 0

            loss.append(pos_loss + neg_loss)
        
        loss = sum(loss) / n
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ProxyISA(nn.Module):
    def __init__(self, numClasses, sizeEmbed, mrg=0.1, alpha=32, V=100, k=1.0, lam=0.1, h=0.15, tau=1.5):
        super(ProxyISA, self).__init__()
        self.numClasses = numClasses
        # Proxy Initialization
        self.proxies = nn.Parameter(torch.Tensor(numClasses, sizeEmbed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.counter = [0] * self.numClasses
        self.learnedSim = torch.ones(self.numClasses, dtype=float).cuda()
        self.mrg = mrg
        self.alpha = alpha
        self.V = V
        self.beta = (V - 1) / V
        self.effectiveNum = torch.zeros(self.numClasses, dtype=float).cuda()
        self.k = k
        self.lam = lam
        self.hardnessScaler = h
        self.tau = tau
        self.enableMemory = False
        self.enableFilter = False

    def forward(self, inputsBatch, targetsBatch, memory=None):
        batchSize = len(targetsBatch)
        P_one_hot = binarize(T=targetsBatch, nb_classes=self.numClasses)
        N_one_hot = 1 - P_one_hot

        proxy_l2 = l2_norm(self.proxies)
        # Calcluate cosine similarity
        cos = F.linear(inputsBatch, proxy_l2)
        
        classCnt = P_one_hot.sum(dim=0)
        classInBatch = torch.nonzero(classCnt != 0, as_tuple=False).squeeze(dim=1)

        w_base = 1.0 / (1.0 + torch.log(1.0 + self.effectiveNum))
        w = (1.0 + torch.exp(-self.tau)) * (w_base - 1) / (1.0 + torch.exp(self.V - self.tau - self.effectiveNum)) + 1.0
        eta = (1.0 + self.k * (1.0 - self.learnedSim)) * w_base + self.lam
        outlierSim = self.learnedSim - eta

        pos_cos = self.mrg - cos
        neg_cos = cos + self.mrg

        pos_weight_sum = 0.0
        neg_weight_sum = 0.0

        if self.enableFilter:
            filtered = torch.tensor([]).cuda()

        mask = torch.ones((batchSize, self.numClasses), dtype=cos.dtype).cuda()

        for i in classInBatch:
            P_idx = torch.nonzero(P_one_hot[:, i], as_tuple=False).squeeze(dim=1)
            P_idy = torch.tensor([i] * len(P_idx))

            if self.enableFilter:
                # Count filtered sample
                filt_sample_id = torch.nonzero(cos[P_idx, P_idy] < outlierSim[i], as_tuple=False).squeeze(dim=1)
                if len(filt_sample_id) > 0:
                    classCnt[i] -= len(filt_sample_id)
                    filtered = torch.hstack((filtered, P_idx[filt_sample_id]))

                P_i_mask = torch.where(cos[P_idx, P_idy] > self.learnedSim[i],
                                       torch.ones_like(cos[P_idx, P_idy]) * w[i],
                                       torch.ones_like(cos[P_idx, P_idy]) + w[i]).float()
                P_i_mask = torch.where(cos[P_idx, P_idy] < outlierSim[i], torch.ones_like(P_i_mask) * w[i], P_i_mask)
                mask[P_idx, P_idy] = P_i_mask

            pos_weight_sum += mask[P_idx, P_idy].mean()

        for i in range(self.numClasses):
            N_idx = torch.nonzero(N_one_hot[:, i], as_tuple=False).squeeze(dim=1)
            if len(N_idx) == 0:
                continue
            N_idy = torch.tensor([i] * len(N_idx))

            N_i_mask = torch.where(cos[N_idx, N_idy] < outlierSim[i], #self.learnedSim[i] - eta[i],
                                   torch.ones_like(cos[N_idx, N_idy]) / max(1.0, self.effectiveNum[i]),
                                   torch.ones_like(cos[N_idx, N_idy])).float()
            mask[N_idx, N_idy] = N_i_mask

            neg_weight_sum += N_i_mask.mean()
        
        
        pos_term = torch.log(1.0 + torch.sum(P_one_hot * torch.exp(self.alpha * pos_cos * mask), dim=0)).sum()
        neg_term = torch.log(1.0 + torch.sum(N_one_hot * torch.exp(self.alpha * neg_cos * mask), dim=0)).sum()

        if self.enableMemory:
            if self.enableFilter:
                clean_idx = torch.tensor([x for x in range(batchSize) if x not in filtered])
                newFeats = inputsBatch[clean_idx]
                newTargets = targetsBatch[clean_idx]
            else:
                newFeats = inputsBatch
                newTargets = targetsBatch

            memory.enqueue_dequeue(newFeats.detach(), newTargets.detach())
            featsMem, targetsMem = memory.get()

        for i in classInBatch:
            if classCnt[i].item() > 0:
                self.counter[i] += classCnt[i].item()
                self.effectiveNum[i] = (1.0 - self.beta ** self.counter[i]) / (1.0 - self.beta)
            if self.enableMemory:
                targetIdx = torch.nonzero(targetsMem == i, as_tuple=False).squeeze(dim=1)
                if targetIdx.shape[0] > 0:
                    self.learnedSim[i] = F.linear(proxy_l2[i].detach(), featsMem[targetIdx]).mean() * self.hardnessScaler

        loss = pos_term / pos_weight_sum + neg_term / neg_weight_sum

        return loss