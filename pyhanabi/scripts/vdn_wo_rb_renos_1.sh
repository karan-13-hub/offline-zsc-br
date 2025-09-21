# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_wo_rb.py \
       --save_dir /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_0\
       --num_thread 80 \
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
       --num_epoch 501 \
       --num_lstm_layer 2 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --visit_weight 0.5 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --load_after 5 \
       --num_eval_after 1 \
       --save_model_after 100 \
       --num_player 3 \
       --rnn_hid_dim 512 \
       --multi_step 3 \
       --prefetch 50 \
       --bc True \
       --bc_weight 1.0 \
       --train_device cuda:0 \
       --dataset_path /home/zaboreno/hanabi_3p/data/np3_vdn_offline_data_seed-9/data_640 /home/zaboreno/hanabi_3p/data/np3_vdn_offline_data_seed-9/data_1280 \
       --data_sample 0.8\
       # --cql True \
       # --cql_weight 1.0 \