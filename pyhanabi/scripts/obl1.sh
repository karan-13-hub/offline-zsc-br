# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python selfplay.py \
       --save_dir exps/vanilla-obl-seed-9 \
       --num_thread 80 \
       --num_game_per_thread 80 \
       --sad 0 \
       --act_base_eps 0.1 \
       --act_eps_alpha 7 \
       --lr 6.25e-04 \
       --eps 1.5e-05 \
       --grad_clip 5 \
       --gamma 0.999 \
       --seed 9 \
       --batchsize 128 \
       --burn_in_frames 10000 \
       --replay_buffer_size 131072 \
       --epoch_len 1000 \
       --num_epoch 501 \
       --num_player 2 \
       --rnn_hid_dim 512 \
       --multi_step 1 \
       --train_device cuda:2 \
       --act_device cuda:3,cuda:4 \
       --num_lstm_layer 2 \
       --boltzmann_act 0 \
       --min_t 0.01 \
       --max_t 0.1 \
       --off_belief 1 \
       --num_fict_sample 10 \
       --belief_device cuda:3,cuda:4 \
       --belief_model exps/belief_obl0/model0.pthw \
       --load_model None \
       --net publ-lstm \
       # --load_model exps/br_medium_data_seed_9_42_1234/vdn_cp_bc_0.35/model_seed_9_agent_br_epoch_150.pthw\


# python selfplay.py \
#        --save_dir exps/obl-offline-1 \
#        --num_thread 80 \
#        --num_game_per_thread 80 \
#        --sad 0 \
#        --act_base_eps 0.1 \
#        --act_eps_alpha 7 \
#        --lr 6.25e-05 \
#        --eps 1.5e-05 \
#        --grad_clip 5 \
#        --gamma 0.999 \
#        --seed 2254257 \
#        --batchsize 128 \
#        --burn_in_frames 10000 \
#        --replay_buffer_size 100000 \
#        --epoch_len 1000 \
#        --num_epoch 1500 \
#        --num_player 2 \
#        --rnn_hid_dim 512 \
#        --multi_step 1 \
#        --train_device cuda:4 \
#        --act_device cuda:5,cuda:6 \
#        --num_lstm_layer 2 \
#        --boltzmann_act 0 \
#        --min_t 0.01 \
#        --max_t 0.1 \
#        --off_belief 1 \
#        --num_fict_sample 10 \
#        --belief_device cuda:7,cuda:8 \
#        --belief_model exps/belief_obl0/model0.pthw \
#        --load_model None \
#        --net publ-lstm \
