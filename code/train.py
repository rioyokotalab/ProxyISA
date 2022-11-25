import torch, math, time, argparse, os
import random, dataset, utils, losses, net
import numpy as np

from dataset.Inshop import Inshop_Dataset
from net.resnet import *
from net.googlenet import *
from net.bn_inception import *
from dataset import sampler
from memory import CrossBatchMemory
from torch.utils.data.sampler import BatchSampler
from torch.utils.data.dataloader import default_collate

from tqdm import *
import wandb

seed = 67
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # set random seed for all gpus

parser = argparse.ArgumentParser(description='Official implementation of `Informative Sample-Aware Proxies for Deep Metric Learning`')
# Export directory, training and val datasets, test datasets
parser.add_argument('--LOG_DIR', 
                    default='../logs',
                    help='Path to log folder'
)
parser.add_argument('--dataset', 
                    default='cub',
                    help='Training dataset, e.g. cub, cars, SOP, Inshop'
)
parser.add_argument('--embedding_size', default=512, type=int,
                    dest='sz_embedding',
                    help='Size of embedding that is appended to backbone model'
)
parser.add_argument('--batch_size', default=128, type=int,
                    dest='sz_batch',
                    help='Size of mini-batch'
)
parser.add_argument('--epochs', default=100, type=int,
                    dest='num_epochs',
                    help='Number of training epochs'
)
parser.add_argument('--gpu_id', default=0, type=int,
                    help='ID of GPU that is used for training'
)
parser.add_argument('--workers', default=4, type=int,
                    dest='num_workers',
                    help='Number of workers for dataloader'
)
parser.add_argument('--model', default='bn_inception',
                    help='Model for training'
)
parser.add_argument('--loss', default='ProxyISA',
                    help='Criterion for training'
)
parser.add_argument('--optimizer', default='adam',
                    help='Optimizer for training'
)
parser.add_argument('--lr', default=1e-4, type=float,
                    help='Learning rate for training'
)
parser.add_argument('--scheduler', default='step',
                    help='Learning rate scheduler, e.g. step, exp, cyclic'
)
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='Weight decay parameter for optimizer'
)
parser.add_argument('--lr_decay_step', default=10, type=int,
                    help='Decay step of learning rate'
)
parser.add_argument('--lr_decay_gamma', default=0.5, type=float,
                    help='Margin parameter gamma for scheduler'
)
parser.add_argument('--warm', default=1, type=int,
                    help='Warmup epochs for training'
)
parser.add_argument('--bn_freeze', default=True, type=bool,
                    help='Batch normalization parameter freeze'
)
parser.add_argument('--l2_norm', default=True, type=bool,
                    help='L2 normlization for model outputs'
)
parser.add_argument('--init_type', default='normal',
                    help='Set initialize type for proxies, e.g. uniform, normal, center'
)

# Memory queue settings
parser.add_argument('--enableMemory', default=False, type=bool,
                    help='Use memory queue'
)
parser.add_argument('--T', default=55000, type=int,
                    help='Size of memory queue'
)
parser.add_argument('--start_epoch', default=2, type=int,
                    help='Num. of epoch to introduce memory queue for training'
)

# ProxyISA settings
parser.add_argument('--k', default=0.9, type=float,
                    help='Scaling factor for eta'
)
parser.add_argument('--V', default=100, type=float,
                    help='Total volumetric unit of feature space for a class'
)
parser.add_argument('--lam', default=0.1, type=float,
                    help='margin parameter for eta'
)

# ProxyAnchor settings
parser.add_argument('--alpha', default=32.0, type=float,
                    help='Scaling parameter for Proxy-Anchor loss'
)
parser.add_argument('--mrg', default=0.1, type=float,
                    help='Margin parameter for Proxy-Anchor loss'
)

# ProxyNCA settings
parser.add_argument('--ncaScale', default=12.0, type=float,
                    help='Scaling parameter for Proxy-NCA loss'
)

# Balanced Sampling settings
parser.add_argument('--IPC', default=None, type=int,
                    help='Balanced sampling, images per class'
)
parser.add_argument('--p', default=0.2, type=float,
                    help='Hyper-parameter of balanced sampling'
)

parser.add_argument('--eval_metric', default='MLRC',
                    help='Evaluation method'
)
parser.add_argument('--visualize', default=False, type=bool,
                    help='If True, visualize embedding space with t-SNE'
)
parser.add_argument('--remark', default='',
                    help='Any reamrk'
)

args = parser.parse_args()

if args.gpu_id != -1:
    torch.cuda.set_device(args.gpu_id)

# Directory for Log
LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embSize{}_k{}_lam{}_{}_lr{}_batchSize{}{}'.format(args.dataset, args.model, args.loss, args.sz_embedding, args.k, 
                                                                                            args.lam, args.optimizer, args.lr, args.sz_batch, args.remark)
# Wandb Initialization
wandb.init(project=args.dataset + '_ProxyISA', notes=LOG_DIR)
wandb.config.update(args)

if args.enableMemory:
    memory_q = CrossBatchMemory(args.T, args.sz_embedding)
    print('Use Memory Queue')

os.chdir('../data/')
data_root = os.getcwd()
# Dataset Loader and Sampler
if args.dataset != 'Inshop':
    trn_dataset = dataset.load(name=args.dataset,
                               root=data_root,
                               mode='train',
                               transform=dataset.utils.make_transform(is_train=True, is_inception=(args.model == 'bn_inception')))
else:
    trn_dataset = Inshop_Dataset(root=data_root,
                                 mode='train',
                                 transform=dataset.utils.make_transform(is_train=True, is_inception=(args.model == 'bn_inception')))

if args.IPC:
    balanced_sampler = sampler.PowerBalancedSampler(trn_dataset, batch_size=args.sz_batch, images_per_class=args.IPC, p=args.p)
    batch_sampler = BatchSampler(balanced_sampler, batch_size=args.sz_batch, drop_last=True)
    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        batch_sampler=batch_sampler)
    print('Balanced Sampling')
else:
    dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                        batch_size=args.sz_batch,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        drop_last=True,
                                        pin_memory=True)
    print('Random Sampling')

if args.dataset != 'Inshop':
    ev_dataset = dataset.load(name=args.dataset,
                              root=data_root,
                              mode='eval',
                              transform=dataset.utils.make_transform(is_train=False, is_inception=(args.model == 'bn_inception')))

    dl_ev = torch.utils.data.DataLoader(ev_dataset,
                                        batch_size=args.sz_batch,
                                        shuffle=False,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
else:
    query_dataset = Inshop_Dataset(root=data_root,
                                   mode='query',
                                   transform=dataset.utils.make_transform(is_train=False, is_inception=(args.model == 'bn_inception')))

    dl_query = torch.utils.data.DataLoader(query_dataset,
                                           batch_size=args.sz_batch,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True)

    gallery_dataset = Inshop_Dataset(root=data_root,
                                     mode='gallery',
                                     transform=dataset.utils.make_transform(is_train=False, is_inception=(args.model == 'bn_inception')))

    dl_gallery = torch.utils.data.DataLoader(gallery_dataset,
                                             batch_size=args.sz_batch,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True)

num_classes = trn_dataset.nb_classes()

# Backbone Model
if args.model.find('googlenet')+1:
    model = googlenet(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet18')+1:
    model = Resnet18(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet50')+1:
    model = Resnet50(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
elif args.model.find('resnet101')+1:
    model = Resnet101(embedding_size=args.sz_embedding, pretrained=True, is_norm=args.l2_norm, bn_freeze=args.bn_freeze)
model = model.cuda()

if args.gpu_id == -1:
    model = torch.nn.DataParallel(model)

# DML Losses
if args.loss == 'ProxyISA':
    if not args.enableMemory:
        raise NotImplementedError('Memory queue is not enabled.')
    criterion = losses.ProxyISA(numClasses=num_classes, sizeEmbed=args.sz_embedding, V=args.V, k=args.k, lam=args.lam).cuda()
elif args.loss == 'ProxyAnchor':
    criterion = losses.ProxyAnchor(nb_classes=num_classes, sz_embed=args.sz_embedding, mrg=args.mrg, alpha=args.alpha).cuda()
elif args.loss == 'ProxyNCA':
    criterion = losses.ProxyNCA(numclasses=num_classes, sizeEmbed=args.sz_embedding, scale=args.ncaScale).cuda()
elif args.loss == 'MS':
    criterion = losses.MultiSimilarityLoss().cuda()
elif args.loss == 'Contrastive':
    criterion = losses.ContrastiveLoss().cuda()
else:
    raise NotImplementedError('{} is currently not supported.'.format(args.loss))

if args.init_type == 'center' and (args.loss == 'ProxyISA' or args.loss == 'ProxyAnchor' or args.loss == 'ProxyNCA'):
    criterion.proxies = torch.nn.Parameter(utils.proxy_init_calc(model, dl_tr))

# Train Parameters
param_groups = [
    {'params': list(set(model.parameters()).difference(set(model.embedding.parameters()))) if args.gpu_id != -1 else 
                 list(set(model.module.parameters()).difference(set(model.module.embedding.parameters())))},
    {'params': model.embedding.parameters() if args.gpu_id != -1 else model.module.embedding.parameters(), 'lr':float(args.lr) * 1},
]
if args.loss == 'ProxyISA' or args.loss == 'ProxyAnchor' or args.loss == 'ProxyNCA':
    param_groups.append({'params': criterion.proxies, 'lr': args.lr * 100})

# Optimizer Setting
if args.optimizer == 'sgd': 
    opt = torch.optim.SGD(param_groups, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
elif args.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=args.lr, alpha=0.9, weight_decay=args.weight_decay, momentum=0.9)
elif args.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    
# Scheduler Setting
if args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
elif args.scheduler == 'cyclic':
    scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr * 0.25, max_lr=args.lr, step_size_up=500,
                                                  cycle_momentum=args.optimizer == 'sgd' or args.optimizer == 'rmsprop')
elif args.scheduler == 'exp':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=args.lr_decay_gamma)

print("Training parameters: {}".format(vars(args)))
print("Training for {} epochs.".format(args.num_epochs))
losses_list = []
best_recall=[0]
best_map = 0
best_epoch = 0

for epoch in range(args.num_epochs):
    model.train()
    if args.bn_freeze:
        modules = model.model.modules() if args.gpu_id != -1 else model.module.model.modules()
        for m in modules:
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []
    
    # Warmup: Train only new params, helps stabilize learning
    if args.warm > 0:
        if args.gpu_id != -1:
            unfreeze_model_param = list(model.embedding.parameters()) + list(criterion.parameters())
        else:
            unfreeze_model_param = list(model.module.embedding.parameters()) + list(criterion.parameters())

        if epoch == 0:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = False
        if epoch == args.warm:
            for param in list(set(model.parameters()).difference(set(unfreeze_model_param))):
                param.requires_grad = True

    if epoch == 0 and args.visualize:
        with torch.no_grad():
            tsneFig = utils.tSNEPlot(model, dl_tr, n_components=2, proxy=criterion.proxies)
            wandb.log({"t-SNE": tsneFig}, step=epoch)

    pbar = tqdm(enumerate(dl_tr))

    for batch_idx, (imgs, targets) in pbar:                         
        feats = model(imgs.squeeze().cuda())

        if args.loss == 'ProxyISA':
            if epoch == args.start_epoch:
                criterion.enableMemory = True
            if epoch == args.start_epoch + 1:
                criterion.enableFilter = True
            loss = criterion(feats, targets.squeeze().cuda(), memory_q)
        elif args.enableMemory and epoch >= args.start_epoch:
            memory_q.enqueue_dequeue(feats.detach(), targets.squeeze().detach().cuda())
            mem_feats, mem_targets = memory_q.get()
            loss = criterion(feats, targets.squeeze().cuda(), mem_feats, mem_targets)
        elif args.loss == 'Contrastive' or args.loss == 'MS':
            loss = criterion(feats, targets.squeeze().cuda(), feats, targets.squeeze().cuda())
        else:
            loss = criterion(feats, targets.squeeze().cuda())
        
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if args.loss == 'ProxyISA' or args.loss == 'ProxyAnchor' or args.loss == 'ProxyNCA':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()
        if args.scheduler == 'cyclic':
            scheduler.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * batch_idx / len(dl_tr),
                loss.item()))
        
    losses_list.append(np.mean(losses_per_epoch))
    wandb.log({'loss': losses_list[-1]}, step=epoch)
    if args.scheduler != 'cyclic':
        scheduler.step()

    # Evaluation
    includeMAP = args.eval_metric == 'MLRC'
    with torch.no_grad():
        print("**Evaluating...**")
        if args.dataset == 'Inshop':
            res = utils.evaluate_cos_Inshop(model, dl_query, dl_gallery, calcMAP=includeMAP)
        elif args.dataset != 'SOP':
            res = utils.evaluate_cos(model, dl_ev, calcMAP=includeMAP)
        else:
            res = utils.evaluate_cos_SOP(model, dl_ev, calcMAP=includeMAP)
        if args.visualize and (epoch + 1) % 10 == 0 and (epoch < 100 or (epoch - 99) % 90 == 0):
            tsneFig = utils.tSNEPlot(model, dl_tr, n_components=2, proxy=criterion.proxies)
            wandb.log({"t-SNE": tsneFig}, step=epoch)

    # Logging Evaluation Score
    if includeMAP:
        mean_average_precision = res[-1]
        wandb.log({"MAP@R": mean_average_precision}, step=epoch)
    if args.dataset == 'Inshop':
        for i, K in enumerate([1,10,20,30,40,50]):    
            wandb.log({"R@{}".format(K): res[i]}, step=epoch)
    elif args.dataset != 'SOP':
        for i in range(6):
            wandb.log({"R@{}".format(2**i): res[i]}, step=epoch)
    else:
        for i in range(4):
            wandb.log({"R@{}".format(10**i): res[i]}, step=epoch)

    # Best model save
    if args.eval_metric == 'MLRC' and best_map < mean_average_precision:
        best_map = mean_average_precision
        best_epoch = epoch
        if not os.path.exists('{}'.format(LOG_DIR)):
            os.makedirs('{}'.format(LOG_DIR))
        torch.save({'model_state_dict': model.state_dict()}, '{}/{}_{}_best.pth'.format(LOG_DIR, args.dataset, args.model))
        with open('{}/{}_{}_best_results.txt'.format(LOG_DIR, args.dataset, args.model), 'w') as f:
            f.write('Best Epoch: {}\n'.format(best_epoch))
            f.write('Best MAP@R: {:.4f}\n'.format(best_map * 100))
            if args.dataset == 'Inshop':
                for i, K in enumerate([1,10,20,30,40,50]):    
                    f.write('Best Recall@{}: {:.4f}\n'.format(K, res[i] * 100))
            elif args.dataset != 'SOP':
                for i in range(6):
                    f.write('Best Recall@{}: {:.4f}\n'.format(2**i, res[i] * 100))
            else:
                for i in range(4):
                    f.write('Best Recall@{}: {:.4f}\n'.format(10**i, res[i] * 100))