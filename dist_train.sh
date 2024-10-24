#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")
port=$(shuf -i 1024-49151 -n 1)

current_time=$(date "+%Y-%m-%d_%H")

epoch=5
bs=8
gpus=2
lr=1.e-6

encoder=vitl
dataset="hypersim,vkitti,nyu,syns"
img_size=(462,616)
min_depth=0.01
max_depth=100 #meters
pretrained_from=./checkpoint/latest.pth
save_path=exp/${current_time}

mkdir -p $save_path

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$port \
    train.py --epoch $epoch --encoder $encoder --bs $bs --lr $lr --save-path $save_path --dataset $dataset \
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port $port 2>&1 | tee -a $save_path/$now.log 
