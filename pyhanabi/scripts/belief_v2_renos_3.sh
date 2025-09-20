# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_belief_v2.py \
       --save_dir /home/zaboreno/offline-zsc-br/pyhanabi/exps/br_medium_data_seed_777_31337_1e9+7/iql_cp_bc_0.4_finetune/agent_2/belief\
       --num_thread 80 \
       --num_game_per_thread 80 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --hid_dim 512 \
       --burn_in_frames 10000 \
       --replay_buffer_size 100000 \
       --epoch_len 1000 \
       --num_epoch 2101 \
       --train_device cuda:0 \
       --act_device cuda:1 \
       --explore 1 \
       --policy /home/zaboreno/offline-zsc-br/pyhanabi/exps/br_medium_data_seed_777_31337_1e9+7/iql_cp_bc_0.4_finetune/agent_2/model_seed_1000000007_epoch_0.pthw \
       --seed 1000000007 \
       --num_player 2 \
       --shuffle_color 0 \
       --save_model_after 50 \
       --update_agent 50 \
