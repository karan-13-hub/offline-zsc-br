# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash
python tools/convert_model.py \
       --save_path /data/kmirakho/hanabi_2p/models\
       --model /data/kmirakho/hanabi_2p/exps/br_expert_data_seed_42_1337_31337/final_vdn_on_cp_obl_finetune/seed_1337/BR_agent_seed_1337_epoch_1900.pthw
       # --dataset_path /data/kmirakho/vdn-offline-data-seed-777/data_80\
       # --cp True\
       # --burn_in_frames 10000 \
       # --finetune_coop_agents_belief 1 \
       # --finetune_coop_agents 1 \
