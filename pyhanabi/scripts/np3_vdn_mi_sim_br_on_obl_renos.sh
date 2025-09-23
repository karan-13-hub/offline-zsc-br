# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay_mi_sim_br_on_obl_with_agent_belief_finetune.py \
       --save_dir /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/final_vdn_on_cp_obl_finetune\
       --load_br_model /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111/BR_cp_bc_1.0/model_seed_9_epoch_100.pthw\
       --load_model /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_0/model_seed_9_epoch_0.pthw /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_1/model_seed_42_epoch_0.pthw /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_2/model_seed_111_epoch_0.pthw\
       --belief_model /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_0/belief/belief_seed_9_epoch_50.pthw /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_1/belief/belief_seed_42_epoch_50.pthw /home/zaboreno/hanabi_3p/exps/br_expert_data_seed_9_42_111_777_1234_31337/coop_agents_sp_1.0_bc_0.4/agent_2/belief/belief_seed_111_epoch_50.pthw\
       --num_thread 80 \
       --num_game_per_thread 80 \
       --num_data_thread 4 \
       --num_update_between_sync 2500\
       --actor_sync_freq 10\
       --coop_agent_belief_sync_freq 5000\
       --update_coop_agents True\
       --update_coop_agents_belief True\
       --update_coop_agents_freq 50\
       --update_coop_agents_belief_freq 50\
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
       --replay_buffer_size 262144 \
       --epoch_len 1000 \
       --num_epoch 2101 \
       --num_lstm_layer 2 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --load_after 5 \
       --num_eval_after 1 \
       --coeff_selfplay 0.0 \
       --sp_weight 0.0 \
       --save_model_after 100 \
       --num_player 3 \
       --rnn_hid_dim 512 \
       --num_agents 3\
       --multi_step 1 \
       --coop_multi_step 1 \
       --act_device cuda:1,cuda:2 \
       --train_device cuda:0 \
       --off_belief 1
       # --dataset_path /data/kmirakho/vdn-offline-data-seed-777/data_80\
       # --cp True\
       # --burn_in_frames 10000 \
       # --finetune_coop_agents_belief 1 \
       # --finetune_coop_agents 1 \
