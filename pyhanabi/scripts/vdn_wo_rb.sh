# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_wo_rb.py \
       --save_dir exps/train_data_80_seed1234/vdn_offline_medium_replay_data80_seed9 \
       --num_thread 40 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
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
       --num_epoch 150 \
       --num_lstm_layer 2 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --visit_weight 0.5 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --load_after 5 \
       --num_eval_after 1 \
       --save_model_after 25 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --prefetch 50 \
       --bc True \
       --bc_weight 1 \
       --act_device cuda:2,cuda:3 \
       --train_device cuda:2 \
       --dataset_path /data/kmirakho/vdn-offline-data-seed1234/data_vdn_80