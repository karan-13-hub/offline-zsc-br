# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python best_response.py \
       --save_dir /data/kmirakho/br_vdn/exps/train_-offline-data-seed-9-42-1234/br_vdn_lstm\
       --coop_agents pyhanabi/exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35/model_seed_9_agent_0_epoch_150.pthw pyhanabi/exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35/model_seed_9_agent_1_epoch_150.pthw pyhanabi/exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35/model_seed_9_agent_2_epoch_150.pthw pyhanabi/exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35/model_seed_9_agent_3_epoch_150.pthw\
       --method vdn \
       --mode br \
       --num_thread 40 \
       --num_game_per_thread 80 \
       --method vdn \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 5e-04 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 777 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 1001 \
       --num_lstm_layer 2 \
       --priority_exponent 0.9 \
       --priority_weight 0.6 \
       --train_bomb 0 \
       --eval_bomb 0 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --prefetch 50 \
       --act_device cuda:2,cuda:3 \
       --train_device cuda:1 \