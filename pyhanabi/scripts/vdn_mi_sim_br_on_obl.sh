# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_mi_sim_br_on_obl_with_agent_belief_finetune.py \
       --save_dir exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune_contd \
       --clu_mod_dir exps/br_medium_data_seed_777_31337_1e9+7/vdn_cp_bc_0.4_finetune\
       --load_br_model exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune/BR_agent_seed_9_epoch_400.pthw\
       --load_model exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune/coop_agent_1_seed_9_epoch_400.pthw exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune/coop_agent_2_seed_9_epoch_400.pthw exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune/coop_agent_3_seed_9_epoch_400.pthw\
       --belief_model exps/br_medium_data_seed_777_31337_1e9+7/vdn_on_cp_obl_finetune_TRY_with_sp_0.0_agent_belief_finetune/ \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 2500\
       --actor_sync_freq 10\
       --coop_agent_belief_sync_freq 5000\
       --method iql \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 10 \
       --fc_only 0 \
       --gamma 0.999 \
       --seed 9 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 1501 \
       --num_lstm_layer 2 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --load_after 5 \
       --num_eval_after 1 \
       --coeff_selfplay 0.0 \
       --sp_weight 0.0 \
       --save_model_after 100 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --num_agents 3\
       --multi_step 1 \
       --start_epoch 400 \
       --act_device cuda:2,cuda:3 \
       --train_device cuda:1 \
       --finetune_coop_agents 1 \
       --off_belief 1 \
       --finetune_coop_agents_belief 1 \
       # --dataset_path /data/kmirakho/vdn-offline-data-seed-777/data_80\
       # --cp True\
       # --burn_in_frames 10000 \