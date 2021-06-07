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

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32, resample = False, cutoff = 0.5, nonzero_loss_cutoff = 1.4, NPR = 0.5):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.resample = resample
        self.cutoff = cutoff

        # We sample only from negatives that induce a non-zero loss
        # These are negative proxies with a distance < nonzero_loss_cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff
        self.nb_n_proxies = int(NPR * nb_classes)
        
    def forward(self, X, T):
        P = self.proxies

        n, d = X.shape

        distance = get_distance(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)

        if self.resample: # Distance weighted sampling
            # Cut off to avoid high variance
            distance_clip = torch.maximum(distance, distance.new_full(distance.shape, self.cutoff))
            # Subtract max(log(distance)) for stability
            log_weights = ((2.0 - float(d)) * torch.log(distance_clip)
                          - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (distance_clip ** 2.0)))
            weights = torch.exp(log_weights - torch.max(log_weights))
            # Sample only negative examples
            weights = weights * (distance_clip < self.nonzero_loss_cutoff)
            weights = (weights / torch.sum(weights, dim=0, keepdim=True))

            n_indices = []
            for i in range(n):
                n_index = []
                try:
                    n_index += np.random.choice(self.nb_classes, self.nb_n_proxies, p=weights[i], replace=False).tolist()
                except:
                    n_index += np.random.choice(self.nb_classes, self.nb_n_proxies, replace=False).tolist()
                n_indices.append(n_index)

            N_one_hot = [[0 for i in range(self.nb_classes)] for j in range(n)]
            for i in range(len(n_indices)):
                for j in range(len(n_indices[0])):
                    N_one_hot[i][int(n_indices[i][j])] = 1
            N_one_hot = torch.FloatTensor(N_one_hot).cuda()
        else:
            N_one_hot = 1 - P_one_hot

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(nn.Module):
    def __init__(self, scale_neg = 50.0, hard_mining = True):
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
            # neg_count += len(neg_pair)

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
        # log_info["neg_count"] = neg_count / batch_size
        loss = sum(loss) / batch_size
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, inputs_col, targets_col, inputs_row, target_row):
        n = inputs_col.size(0)
        # Compute similarity matrix
        sim_mat = F.linear(l2_norm(inputs_col), l2_norm(inputs_row))
        epsilon = 1e-5
        loss = list()

        # neg_count = list()
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
        # if inputs_col.shape[0] == inputs_row.shape[0]:
        #     prefix = "batch_"
        # else:
        #     prefix = "memory_"
        loss = sum(loss) / n  # / all_targets.shape[1]
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, ):
        super(NPairLoss, self).__init__()
       #self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(distance = CosineSimilarity(),
                                           reducer = ThresholdReducer(high=0.3),
                                           embedding_regularizer = LpRegularizer())
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
