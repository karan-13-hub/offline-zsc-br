# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint
from sklearn.cluster import KMeans
import numpy as np
import torch
from torch import nn

from act_group import ActGroup, BRActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils
from process_data import save_replay_buffer
from replay_load import PrioritizedReplayBuffer
from replay_load import LoadDataset, DataLoader, process_batch
from replay_load import ChunkedDataLoader
import concurrent.futures as futures
import datetime
import copy
from tqdm import tqdm
from losses import train_br_agent, diversity_loss
import gc   
import common_utils
import utils
from train_mi import *

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--models_dir", type=str, default="../models/mi_cluster_medium/vdn_seed9_bc_0.4_div_0.05")
    parser.add_argument("--clu_mod_dir", type=str, default="exps/mi_cluster_data_80_seed9")
    parser.add_argument("--method", type=str, default="vdn")
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--aux_weight", type=float, default=0)
    parser.add_argument("--boltzmann_act", type=int, default=0)
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--off_belief", type=int, default=0)
    parser.add_argument("--belief_model", type=str, default="None")
    parser.add_argument("--num_fict_sample", type=int, default=10)
    parser.add_argument("--belief_device", type=str, default="cuda:1")
    parser.add_argument("--load_br_model", type=str, default="")
    parser.add_argument("--load_model", type=str, nargs="+", default="")
    parser.add_argument("--load_bc_model", type=str, default="", help="path to pre-trained BC model")
    parser.add_argument("--clone_bot", type=str, default="", help="behavior clone loss")
    parser.add_argument("--clone_weight", type=float, default=0.0)
    parser.add_argument("--clone_t", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)

    # DQN settings
    parser.add_argument("--multi_step", type=int, default=3)

    # replay buffer settings
    parser.add_argument("--load_after", type=int, default=20)
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)

    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument("--agent_finetune", type=int, default=0)

    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )

    parser.add_argument(
        "--visit_weight", type=float, default=0.5, help="visit weight"
    )

    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)
    parser.add_argument("--num_data_thread", type=int, default=4)

    # actor setting
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--num_eval_after", type=int, default=100)
    parser.add_argument("--save_model_after", type=int, default=100)
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="/data/kmirakho/offline-model/dataset_rl_1040640_sml.npz")

    #behaviour policy
    parser.add_argument("--bc", type=bool, default=False)
    parser.add_argument("--bc_weight", type=float, default=0.0)
    parser.add_argument("--cp_bc_weight", type=float, default=0.0)
    parser.add_argument("--cp_bc_decay_factor", type=float, default=0.0)
    parser.add_argument("--cp_bc_decay_start", type=int, default=0, help="Number of epochs to decay bc_weight from initial value to eps")

    #sp, cp, cp weights
    parser.add_argument("--cp", type=bool, default=False)
    parser.add_argument("--sp_weight", type=float, default=1.0)
    parser.add_argument("--cp_weight", type=float, default=1.0)
    parser.add_argument("--wgt_thr", type=float, default=0.0)

    #diversity loss
    parser.add_argument("--div", type=bool, default=False)
    parser.add_argument("--div_weight", type=float, default=1.0)
    parser.add_argument("--div_type", type=str, default='jsd')
    parser.add_argument("--start_div", type=int, default=0, help="Epoch to start using div_weight (before this, div_weight is 0)")
    parser.add_argument("--start_epoch", type=int, default=0, help="Number of epochs to start training")

    #loading trained diverse agents
    parser.add_argument("--include", type=str, nargs="+", default=None)
    parser.add_argument("--exclude", type=str, nargs="+", default=None)

    parser.add_argument("--coop_sync_freq", type=int, default=0)
    parser.add_argument("--mode", type=str, default="selfplay")
    parser.add_argument("--update_coop_agents", type=bool, default=False)
    parser.add_argument("--update_coop_agents_freq", type=int, default=0)

    args = parser.parse_args()
    return args

def filter_include(entries, includes):
    if not isinstance(includes, list):
        includes = [includes]
    keep = []
    for entry in entries:
        for include in includes:
            if include not in entry:
                break
        else:
            keep.append(entry)
    return keep


def filter_exclude(entries, excludes):
    if not isinstance(excludes, list):
        excludes = [excludes]
    keep = []
    for entry in entries:
        for exclude in excludes:
            if exclude in entry:
                break
        else:
            keep.append(entry)
    return keep

def load_models(models):
    agents = []
    overwrite = {}
    overwrite["device"] = args.train_device
    for model in models:
        agent, _ = utils.load_agent(model, overwrite)
        agents.append(agent)
    return agents
 
#compute the kmeans clusters
def cluster_data(batch_loader, encoder, args):
    all_lstm_o = []
    epoch_bar = tqdm(batch_loader, desc=f'Finding lstm_o', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
    with torch.no_grad():
        encoder.eval()
        for _, sample in enumerate(epoch_bar):
            batch, _ = process_batch(sample, args)
            z = encode_batch(batch, encoder)
            all_lstm_o.append(z)
    all_lstm_o = torch.cat(all_lstm_o, dim=0)
    lstm_o_np = all_lstm_o.cpu().numpy()
    n_samples = lstm_o_np.shape[0]
    lstm_o_reshaped = lstm_o_np.reshape(n_samples, -1)
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=args.num_agents, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(lstm_o_reshaped)

    # Print cluster statistics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster Statistics:") 
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/n_samples*100:.2f}%)")
    
    return kmeans

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

    if args.boltzmann_act:
        boltzmann_beta = utils.generate_log_uniform(
            1 / args.max_t, 1 / args.min_t, args.num_t
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))
    else:
        boltzmann_t = []
        print("no boltzmann")

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )

    agent_br = r2d2.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        games[0].feature_size(args.sad),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.boltzmann_act,
        False,
        False,
        )
    
    agent_br.sync_target_with_online()
    
    if args.load_br_model and args.load_br_model != "None":
        print("*****loading pretrained model*****")
        print(args.load_br_model)
        utils.load_weight(agent_br.online_net, args.load_br_model, args.train_device)
        print("*****done*****")
    else:
        print("*****no pretrained model*****")
        
    
    agent_br = agent_br.to(args.train_device)
    optim = torch.optim.Adam(agent_br.online_net.parameters(), lr=args.lr, eps=args.eps)
    print('Best Response agent: ', agent_br)
    eval_agent = agent_br.clone(args.train_device, {"vdn": False, "boltzmann_act": False})

    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    if args.coop_sync_freq:
        sync_pool = futures.ThreadPoolExecutor(max_workers=1)
        save_future, coop_future = None, None
        save_ckpt = common_utils.ModelCkpt(args.save_dir)
        print(
            f"{datetime.datetime.now()}, save the initial model to ckpt: {save_ckpt.prefix}"
        )
        utils.save_intermediate_model(agent_br.online_net.state_dict(), save_ckpt)    
    
    # coop_agents = None
    # coop_eval_agents = []
    # if args.load_model and args.load_model != "None":
    #     model_paths = args.load_model
    #     coop_ckpts = []
    #     for i, model_path in enumerate(model_paths):
    #         print(f"Loading model from {model_path}")
    #         coop_ckpts.append(common_utils.ModelCkpt(model_path))
    #         print("*****done*****")    
    #     coop_agents = utils.load_coop_agents(coop_ckpts, device="cpu", vdn=(args.method == "vdn"))

    if args.load_model and args.load_model != "None":
        coop_agents = []
        optim_agents = []
        eval_coop_agents = []
        act_group_agents = []
        replay_buffer_agents = []
        for i in range(len(args.load_model)):
            coop_agents.append(r2d2.R2D2Agent(
                (args.method == "vdn"),
                args.multi_step,
                args.gamma,
                args.eta,
                args.train_device,
                games[0].feature_size(args.sad),
                args.rnn_hid_dim,
                games[0].num_action(),
                args.net,
                args.num_lstm_layer,
                args.boltzmann_act,
                False,
                args.off_belief,
            ))
            coop_agents[i].sync_target_with_online()
            coop_agents[i] = coop_agents[i].to(args.train_device)
            optim_agents.append(torch.optim.Adam(coop_agents[i].online_net.parameters(), lr=args.lr, eps=args.eps))
            print(f"\n\n*****loading pretrained model {i} for training*****")
            print(args.load_model[i])
            utils.load_weight(coop_agents[i].online_net, args.load_model[i], args.train_device)
            eval_coop_agents.append(coop_agents[i].clone(args.train_device, {"vdn": False, "boltzmann_act": False}))
            eval_seed = (9917 + 0 * 999999) % 7777777
            eval_coop_agents[i].load_state_dict(coop_agents[i].state_dict())
            score, perfect, *_ = evaluate(
                [eval_coop_agents[i] for _ in range(args.num_player)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
                device=args.train_device,
            )
            print("================")
            print(f"Initial eval score for pretrained model {i}: {score}, perfect: {perfect}")
            print("================")
            print(f"*****done*****\n\n")
            if args.agent_finetune:
                replay_buffer_agents.append(rela.RNNPrioritizedReplay(
                    args.replay_buffer_size,
                    args.seed,
                    args.priority_exponent,
                    args.priority_weight,
                    args.prefetch,
                ))
                act_group_agents.append(ActGroup(
                    args.act_device,
                    coop_agents[i],
                    args.seed,
                    args.num_thread,
                    args.num_game_per_thread,
                    args.num_player,
                    explore_eps,
                    boltzmann_t,
                    args.method,
                    args.sad,
                    args.shuffle_color,
                    args.hide_action,
                    True,  # trinary, 3 bits for aux task
                    replay_buffer_agents[i],
                    args.multi_step,
                    args.max_len,
                    args.gamma,
                    args.off_belief,
                    None,
                ))

    belief_model = None
    
    act_group_args = {
        "devices": args.act_device,
        "agent": agent_br,
        "seed": args.seed,
        "num_thread": args.num_thread,
        "num_game_per_thread": args.num_game_per_thread,
        "num_player": args.num_player,
        "explore_eps": explore_eps,
        "boltzmann_t": boltzmann_t,
        "method": args.method,
        "sad": args.sad,
        "shuffle_color": args.shuffle_color,
        "hide_action": args.hide_action,
        "trinary": True,  # trinary, 3 bits for aux task
        "replay_buffer": replay_buffer,
        "multi_step": args.multi_step,
        "max_len": args.max_len,
        "gamma": args.gamma
    }

    act_group_cls = BRActGroup    
    if coop_agents is not None:
        if args.mode == "klr" and coop_agents is None:
            print("Going to make BR act group for KLR level 1")
        act_group_args["coop_agents"] = coop_agents

    act_group = act_group_cls(**act_group_args)
    
    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )

    act_group.start()
    context.start()

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up BR agent replay buffer:", replay_buffer.size())
        time.sleep(1)
    print("warming up BR agent replay buffer:", replay_buffer.size(), "\n")
    
    if args.agent_finetune:
        games_agents = []
        context_agents = []
        for i in range(len(args.load_model)):
            games_agents.append(create_envs(
            args.num_thread * args.num_game_per_thread,
            args.seed,
            args.num_player,
            args.train_bomb,
            args.max_len,
            ))
            context_agents.append(create_threads(
                args.num_thread,
                args.num_game_per_thread,
                act_group_agents[i].actors,
                games_agents[i],
            ))
            act_group_agents[i].start()
            context_agents[i][0].start()
            time.sleep(1)
            while replay_buffer_agents[i].size() < args.burn_in_frames:
                print(f"warming up agent {i} replay buffer:", replay_buffer_agents[i].size())
                time.sleep(1)
            print(f"warming up agent {i} replay buffer:", replay_buffer_agents[i].size(), "\n")    

    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    if args.agent_finetune:
        stat_agents = []
        tachometer_agents = []
        stopwatch_agents = []
        for i in range(len(args.load_model)):
            model_dir = os.path.dirname(args.load_model[i])
            stat_agents.append(common_utils.MultiCounter(model_dir))
            tachometer_agents.append(utils.Tachometer())
            stopwatch_agents.append(common_utils.Stopwatch())

    for epoch in range(args.start_epoch, args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()
        if args.agent_finetune:
            for i in range(len(args.load_model)):
                stat_agents[i].reset()
                tachometer_agents[i].start()
                stopwatch_agents[i].reset()
        
        if args.update_coop_agents and epoch % args.update_coop_agents_freq == 0:
            for i in range(len(args.load_model)):
                #load new coop agents
                model_path = args.load_model[i].split("epoch_")[0] + "epoch_" + str(epoch) + ".pthw"
                if os.path.exists(model_path):
                    utils.load_weight(coop_agents[i].online_net, model_path, args.train_device)
                    print(f"Updated model for coop agent {i}: {model_path}")
            act_group.update_coop_models(coop_agents)
            print(f"*****done*****\n\n")

        for batch_idx in tqdm(range(args.epoch_len), desc=f'Training', bar_format='{l_bar}{bar:20}{r_bar}', leave=True):
            num_update = batch_idx + epoch * args.epoch_len
            
            if num_update % args.num_update_between_sync == 0:
                agent_br.sync_target_with_online()
                print(f"\nSynced target with online for BR agent")
                if args.agent_finetune:
                    for i in range(len(args.load_model)):
                        coop_agents[i].sync_target_with_online()
                    print(f"\nSynced target with online for {len(args.load_model)} coop agents")

            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent_br)
                if args.agent_finetune:
                    act_group.update_coop_models(coop_agents)
                    for i in range(len(args.load_model)):
                        act_group_agents[i].update_model(coop_agents[i])

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch.time("sample data br")

            loss, priority, online_q = agent_br.loss(batch, args.aux_weight, stat)
    
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent_br.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            replay_buffer.update_priority(priority)
            stopwatch.time("updating priority")

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

            if args.agent_finetune:
                for i in range(len(args.load_model)):
                    print("LANJAAAAAAAAAA")
                    batch, weight = replay_buffer_agents[i].sample(args.batchsize, args.train_device)
                    stopwatch_agents[i].time(f"sample data agent {i}")    

                    loss, priority, online_q = coop_agents[i].loss(batch, args.aux_weight, stat_agents[i])
                    
                    loss = (loss * weight).mean()
                    loss.backward()
                    torch.cuda.synchronize()
                    stopwatch_agents[i].time(f"forward & backward agent {i}")

                    g_norm = torch.nn.utils.clip_grad_norm_(
                        coop_agents[i].online_net.parameters(), args.grad_clip
                    )
                    
                    optim_agents[i].step()
                    optim_agents[i].zero_grad()
                    torch.cuda.synchronize()
                    stopwatch_agents[i].time(f"update model agent {i}")

                    replay_buffer_agents[i].update_priority(priority)
                    stopwatch_agents[i].time(f"updating priority agent {i}")
                    stat_agents[i]["loss"].feed(loss.detach().item())
                    stat_agents[i]["grad_norm"].feed(g_norm)
                    stat_agents[i]["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        print("\n=====Agent BR======")
        tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()
        stat.summary(epoch)
        if args.agent_finetune:
            for i in range(len(args.load_model)):
                print(f"\n=====Agent {i}======")
                tachometer_agents[i].lap(replay_buffer_agents[i], args.epoch_len * args.batchsize, count_factor)
                stopwatch_agents[i].summary()
                stat_agents[i].summary(epoch)
        
        # eval all agents
        for i in range(len(args.load_model)+1):
            eval_seed = (9917 + epoch * 999999 + i) % 7777777
            if i==0:
                eval_agent.load_state_dict(agent_br.state_dict())
                eval_agents = [eval_agent for _ in range(args.num_player)]
                for j in range(len(args.load_model)):
                    eval_coop_agents[j].load_state_dict(coop_agents[j].state_dict())
                    eval_idxs = np.random.choice(
                        len(eval_coop_agents), args.num_player - 1, replace=False
                    )
                    eval_agents = [eval_agent]
                    for idx in eval_idxs:
                        eval_agents.append(eval_coop_agents[idx])
            elif args.agent_finetune:
                eval_coop_agents[i-1].load_state_dict(coop_agents[i-1].state_dict())
                eval_agents = [eval_coop_agents[i-1] for _ in range(args.num_player)]
            else:
                continue
            score, perfect, *_ = evaluate(
                eval_agents,
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
                device=args.train_device,
            )

            if i==0:
                print("\n=====Eval Scores=====")
                print(f"BR agent eval score: {score}, perfect: {perfect}")
            elif args.agent_finetune:
                print(f"Agent {i} eval score: {score}, perfect: {perfect}")

        # save model
        force_save_name = None
        if epoch > 0 and epoch % args.save_model_after == 0:
            for i in range(len(args.load_model)+1):
                if i==0:
                    force_save_name = f"BR_agent_seed_{args.seed}_epoch_{epoch}"
                    model_saved = saver.save(
                        None, agent_br.online_net.state_dict(), score, force_save_name=force_save_name
                    )
                    print(f"BR agent saved: {model_saved}")
                elif args.agent_finetune:
                    force_save_name = f"coop_agent_{i}_seed_{args.seed}_epoch_{epoch}"
                    model_saved = saver.save(
                        None, coop_agents[i-1].online_net.state_dict(), score, force_save_name=force_save_name
                    )
                    print(f"Agent {i} saved: {model_saved}")
                else:
                    continue
        print("==========\n")