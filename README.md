# MSViT-ReID
The official repository for Multiple Sparse Vision Transformer for Occluded Person Re-Identification.

## Prepare Datasets
Download the person datasets, vehicle datasets, and fine-grained Visual Categorization/Retrieval datasets.

Then unzip them and rename them under your "dataset_root" directory like
```bash
dataset_root
├── Occluded_Duke
├── P-DukeMTMC-reid
├── Market-1501-v15.09.15
├── DukeMTMC-reID
├── MSMT17
├── cuhk03-np
├── VeRi
├── VehicleID_V1.0
├── CARS
├── CUB_200_2011
└── Stanford_Online_Products
```

## Training
We prepared the ImageNet Pretrained backbone in "./pretrain".

### occluded_duke
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset occluded_duke --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.697738 top5:0.819457 top10:0.857014 mAP:0.611958
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset occluded_duke --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.713765 top5:0.825792 top10:0.855656 mAP:0.624352
```

### p_dukemtmc
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset p_dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.919556 top5:0.950069 top10:0.963477 mAP:0.842074
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset p_dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.918632 top5:0.950994 top10:0.960703 mAP:0.844234
```

### market1501
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.961698 top5:0.986045 top10:0.991983 mAP:0.909755
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset market1501 --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.961105 top5:0.984264 top10:0.989905 mAP:0.905942
```

### dukemtmc
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 1 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.924147 top5:0.960054 top10:0.968582 mAP:0.847815
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset dukemtmc --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.929533 top5:0.961400 top10:0.971275 mAP:0.854566
```

### npdetected
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset npdetected --gpus 0 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.848571 top5:0.932857 top10:0.965000 mAP:0.816581
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset npdetected --gpus 4 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.861429 top5:0.939286 top10:0.963571 mAP:0.830185
```

### nplabeled
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset nplabeled --gpus 4 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.874286 top5:0.955000 top10:0.970714 mAP:0.850546
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset nplabeled --gpus 6 --epochs 5,75 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.886429 top5:0.956429 top10:0.976429 mAP:0.866022
```

### msmt17
```bash
python train.py --net msvit_s --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset msmt17 --gpus 0 --epochs 5,45 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.10
# top1:0.877005 top5:0.934300 top10:0.950510 mAP:0.705762
```
```bash
python train.py --net msvit_l --img-height 384 --img-width 128 --batch-size 24 --lr 5.0e-2 --dataset msmt17 --gpus 7 --epochs 5,45 --instance-num 4 --erasing 0.40 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 8,8 --drop-path 0.15
# top1:0.876833 top5:0.939189 top10:0.952140 mAP:0.719252
```

### car196
```bash
python train.py --net msvit_s --img-height 224 --img-width 224 --batch-size 24 --lr 5.0e-2 --dataset car196 --gpus 5 --epochs 5,45 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 7,7 --drop-path 0.10
# Recall@1:0.938507 Recall@2:0.963719 Recall@4:0.976510 Recall@8:0.985365 NMI:0.815310
```
```bash
python train.py --net msvit_l --img-height 224 --img-width 224 --batch-size 24 --lr 5.0e-2 --dataset car196 --gpus 6 --epochs 5,45 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 7,7 --drop-path 0.15
# Recall@1:0.935924 Recall@2:0.960644 Recall@4:0.974665 Recall@8:0.982659 NMI:0.798309
```

### cub200
```bash
python train.py --net msvit_s --img-height 224 --img-width 224 --batch-size 24 --lr 1.0e-3 --dataset cub200 --gpus 6 --epochs 5,45 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 7,7 --drop-path 0.10
# Recall@1:0.714889 Recall@2:0.804018 Recall@4:0.872890 Recall@8:0.921843 NMI:0.725097
```
```bash
python train.py --net msvit_l --img-height 224 --img-width 224 --batch-size 24 --lr 1.0e-3 --dataset cub200 --gpus 7 --epochs 5,45 --instance-num 6 --erasing 0.10 --num-part 2 --triplet-weight 1.0 --freeze stem --dataset-root /dev/shm --ema-ratio 0.80 --ema-extra 25 --sparse-ratio 0.3 --split-size 7,7 --drop-path 0.15
# Recall@1:0.726367 Recall@2:0.816678 Recall@4:0.880317 Recall@8:0.926739 NMI:0.737156
```