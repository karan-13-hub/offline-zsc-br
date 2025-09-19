# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_mi_sim_br.py \
       --save_dir /data/kmirakho/hanabi_3p/exps/br_medium_data_seed_1337_31415_271828_380843_7777777_1e9+7/coop_agents_sp_1.0_bc_0.4_div_0.05\
       --clu_mod_dir /data/kmirakho/hanabi_3p/exps/br_medium_data_seed_1337_31415_271828_380843_7777777_1e9+7\
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 1500\
       --method vdn \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 5e-04 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 1337 \
       --batchsize 512 \
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
       --num_agents 6\
       --multi_step 3\
       --prefetch 10\
       --div True\
       --div_weight 0.05\
       --div_type jsd\
       --start_div 30\
       --bc True \
       --bc_weight 0.4\
       --coop_train True\
       --sp_weight 1.0\
       --act_device cuda:0,cuda:2 \
       --train_device cuda:1 \
       --dataset_path /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-1337/data_80 /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-31415/data_80 /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-271828/data_80 /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-380843/data_80 /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-7777777/data_80 /data/kmirakho/hanabi_3p/data/np3_vdn_offline_data_seed-1e9+7/data_80\
       --data_sample 0.75\
       --wgt_thr 0.25\
       # --load_coop_model /data/kmirakho/hanabi_2p/exps/br_medium_data_seed_9_42_111_777_1234_31337/vdn_cp_bc_0.4_wo_div_coop/\
       # --include epoch_200\
       # --exclude agent_br\
       # --br_train True\
       # --load_br_model exps/br_medium_data_seed_9_42_111_777_1234_31337/vdn_cp_bc_0.4/model_seed_9_agent_br_epoch_0.pthw\
       # --cp_weight 1.0\
       # --cp_bc_weight 0.4\
       # --cp_bc_decay_factor 0\
       # --cp_bc_decay_start 0\
       # --cp True\
       # --cp_cql_weight 0.4\
       # --exclude 100\
