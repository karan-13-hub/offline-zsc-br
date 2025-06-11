# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python replay_load.py \
       --save_dir exps/vdn_offline_2p \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 2 \
       --method vdn \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 5e-04 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 9 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 1500 \
       --num_lstm_layer 2 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_eval_after 1 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --prefetch 3 \
       --act_device cuda:2,cuda:3 \
       --train_device cuda:0\
       --dataset_path /data/kmirakho/vdn-offline-data-seed1234