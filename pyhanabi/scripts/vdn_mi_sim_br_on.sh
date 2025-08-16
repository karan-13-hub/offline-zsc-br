# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_mi_sim_br_on_with_finetune.py \
       --save_dir exps/br_medium_data_seed_9_42_1234/vdn_on_cp_wo_belief_try\
       --load_br_model exps/br_medium_data_seed_9_42_1234/vdn_on_cp_wo_belief_try/BR_agent_9_epoch_100.pthw\
       --load_model exps/br_medium_data_seed_9_42_1234/vdn_on_cp_wo_belief_try/coop_agent_1_9_epoch_100.pthw exps/br_medium_data_seed_9_42_1234/vdn_on_cp_wo_belief_try/coop_agent_2_9_epoch_100.pthw exps/br_medium_data_seed_9_42_1234/vdn_on_cp_wo_belief_try/coop_agent_3_9_epoch_100.pthw\
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 2500\
       --method iql \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-05 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 9 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 1501 \
       --num_lstm_layer 2 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --visit_weight 0.5 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --load_after 5 \
       --num_eval_after 1 \
       --save_model_after 100 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --num_agents 3\
       --multi_step 3 \
       --prefetch 50 \
       --act_device cuda:2,cuda:3 \
       --train_device cuda:1 \
       # --dataset_path /data/kmirakho/vdn-offline-data-seed-777/data_80\
       # --cp True\
       # --burn_in_frames 10000 \