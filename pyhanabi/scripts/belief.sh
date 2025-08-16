# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_belief.py \
       --save_dir exps/belief_9_42_1234_finetune/agent_1 \
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
       --num_epoch 501 \
       --train_device cuda:7 \
       --act_device cuda:6 \
       --explore 1 \
       --policy exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35_finetune/agent_1/model_seed_9_agent_1_epoch_150.pthw \
       --seed 2254257 \
       --num_player 2 \
       --shuffle_color 0 \
       --save_model_after 100 \
