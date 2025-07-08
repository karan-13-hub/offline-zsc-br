# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_mi_sim_br.py \
       --save_dir exps/br_medium_data_seed_42_1e9+7/vdn_cp_bc_0.4\
       --models_dir ../models/mi_cluster_medium/vdn_seed42_bc_0.4 \
       --clu_mod_dir exps/br_medium_data_seed_42_1e9+7/ \
       --num_thread 40 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 1000\
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
       --num_agents 3\
       --multi_step 3 \
       --prefetch 50 \
       --div_weight 0.0\
       --div_type jsd\
       --start_div 0\
       --bc True \
       --bc_weight 0.4\
       --cp_weight 1.0\
       --cp_bc_weight 0.4\
       --cp_bc_decay_factor 0\
       --cp_bc_decay_start 0\
       --sp_weight 1.0\
       --act_device cuda:6,cuda:7 \
       --train_device cuda:6 \
       --dataset_path /data/kmirakho/vdn-offline-data/data_80\
       --wgt_thr 0.25\
       --include 150\
       --exclude 100\
       --div True\
       # --cp True\