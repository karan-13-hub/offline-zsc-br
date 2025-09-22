# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_agent_belief.py \
       --save_dir /home/zaboreno/hanabi_2p/exps/br_expert_data_seed_9_42_111_777_1234_31337/iql_finetune_cp_bc_0.4_coop/agent_5 \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 2500\
       --method iql \
       --seed 31337 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --batchsize 128 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 10 \
       --rnn_hid_dim 512 \
       --hid_dim 512 \
       --num_lstm_layer 2 \
       --gamma 0.999 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 2001 \
       --multi_step 3 \
       --explore 1 \
       --num_eval_after 1 \
       --policy /home/zaboreno/hanabi_2p/exps/br_expert_data_seed_9_42_111_777_1234_31337/agent_5/model_seed_31337_epoch_200.pthw \
       --num_player 2 \
       --shuffle_color 0 \
       --train_device cuda:2 \
       --act_device cuda:0,cuda:1\
       --save_model_after 50 \
       # --off_belief 1 \