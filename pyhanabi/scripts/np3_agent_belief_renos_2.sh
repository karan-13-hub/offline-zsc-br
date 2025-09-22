# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python train_agent_belief.py \
       --save_dir /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/iql_cp_bc_1.0_finetune/agent_1 \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 2500\
       --method iql \
       --seed 42 \
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
       --policy /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_1/model_seed_42_epoch_200.pthw \
       --num_player 3 \
       --shuffle_color 0 \
       --save_model_after 50 \
       --train_device cuda:0 \
       --act_device cuda:1,cuda:2\
       # --off_belief 1 \