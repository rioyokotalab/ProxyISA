# Informative Sample-Aware Proxy for Deep Metric Learning

Official PyTorch implementation of MMAsia 2022 paper [**Informative Sample-Aware Proxy for Deep Metric Learning**]. 

<!-- This repository provides source code of experiments on four datasets (CUB-200-2011, Cars-196, Stanford Online Products and In-shop) and pretrained models. -->

## Requirements

- Python3
- PyTorch (> 1.6)
- NumPy
- tqdm
- [wandb](https://wandb.ai/quickstart/pytorch)
- [plotly](https://plotly.com/python/getting-started/) (needed if you want to visualize with t-SNE)
- [Pytorch-Metric-Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
   - In-shop Clothes Retrieval ([Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Training Embedding Network

<!-- You can download the trained model through the hyperlink in the table. -->

### CUB-200-2011

- Train a embedding network of Inception-BN using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model bn_inception \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 1e-4 \
                --dataset cub \
                --lr_decay_step 10 \
                --enableMemory True
```

- Train a embedding network of ResNet-50 using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model resnet50 \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 1e-4 \
                --dataset cub \
                --warm 0 \
                --lr_decay_step 5 \
                --enableMemory True
```

<!-- | Method | Backbone | Recall@1 | MAP@R |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Proxy-ISA | Inception-BN | 68.1 | 26.97 | -->
<!-- | [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1s-cRSEL2PhPFL9S7bavkrD_c59bJXL_u/view?usp=sharing) | ResNet-50 | 69.9 | 79.6 | 86.6 | 91.4 | -->

### Cars-196

- Train a embedding network of Inception-BN using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model bn_inception \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 1e-4 \
                --dataset cars \
                --lr_decay_step 20 \
                --enableMemory True \
                --k 0.4
```

- Train a embedding network of ResNet-50 using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model resnet50 \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 1e-4 \
                --dataset cars \
                --warm 0 \
                --lr_decay_step 10 \
                --enableMemory True \
                --k 0.4
```

<!-- | Method | Backbone | R@1 | R@2 | R@4 | R@8 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1wwN4ojmOCEAOaSYQHArzJbNdJQNvo4E1/view?usp=sharing) | Inception-BN | 86.4 | 91.9 | 95.0 | 97.0 |
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1_4P90jZcDr0xolRduNpgJ9tX9HZ1Ih7n/view?usp=sharing) | ResNet-50 | 87.7 | 92.7 | 95.5 | 97.3 | -->

### Stanford Online Products

- Train a embedding network of Inception-BN using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model bn_inception \
                --optimizer adamw \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 6e-4 \
                --dataset SOP \
                --warm 1 \
                --bn_freeze False \
                --lr_decay_step 20 \
                --lr_decay_gamma 0.25 \
                --enableMemory True
```

<!-- | Method | Backbone | R@1 | R@10 | R@100 | R@1000 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1hBdWhLP2J83JlOMRgZ4LLZY45L-9Gj2X/view?usp=sharing) | Inception-BN | 79.2 | 90.7 | 96.2 | 98.6 | -->

### In-Shop Clothes Retrieval

- Train a embedding network of Inception-BN using **Proxy-ISA**

```bash
python train.py --gpu_id 0 \
                --loss ProxyISA \
                --model bn_inception \
                --optimizer adamw \
                --embedding_size 512 \
                --batch_size 128 \
                --lr 6e-4 \
                --dataset Inshop \
                --warm 1 \
                --bn_freeze False \
                --lr_decay_step 20 \
                --lr_decay_gamma 0.25 \
                --enableMemory True \
                --k 0.1
```

<!-- | Method | Backbone | R@1 | R@10 | R@20 | R@30 | R@40 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| [Proxy-Anchor<sup>512</sup>](https://drive.google.com/file/d/1VE7psay7dblDyod8di72Sv7Z2xGtUGra/view?usp=sharing) | Inception-BN | 91.9 | 98.1 | 98.7 | 99.0 | 99.1 | -->

## Evaluating Image Retrieval

Follow the below steps to evaluate the trained model. 

Trained best model will be saved in `./logs/folder_name`.

```bash
# The parameters should be changed according to the model to be evaluated.
python evaluate.py --gpu_id 0 \
                   --batch_size 128 \
                   --model bn_inception \
                   --embedding_size 512 \
                   --dataset cub \
                   --resume /PATH/TO/YOUR/Model.pth
```

## Embedding Space Visualization

t-SNE visualization of 512-dimensional embedding space for the Cars-196 dataset (during training).

Left: Proxy-Anchor loss (Kim et al. CVPR 2022);
Right: Proxy-ISA (Ours)

<p align="left"><img src="images/tSNE_ProxyAnchor.png" alt="graph" width="47%">&nbsp<img src="images/tSNE_ProxyISA.png" alt="graph" width="48%"></p>

## Citation
    
    @InProceedings{Li_2022_MMAsia,
      title = {Informative Sample-Aware Proxy for Deep Metric Learning},
      author = {Li, Aoyu and Sato, Ikuro and Ishikawa, Kohta and Kawakami, Rei and Yokota, Rio},
      booktitle = {ACM Multimedia Asia (MMAsia '22)},
      year = {2022},
      doi = {10.1145/3551626.3564942}
    }