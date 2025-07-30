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

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2_br
import utils
from process_data import save_replay_buffer
from replay_load import PrioritizedReplayBuffer
from replay_load import LoadDataset, DataLoader, process_batch
from replay_load import ChunkedDataLoader
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

    parser.add_argument("--load_model", type=str, default="")
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

    #loading trained diverse agents
    parser.add_argument("--include", type=str, nargs="+", default=None)
    parser.add_argument("--exclude", type=str, nargs="+", default=None)

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

    # models = common_utils.get_all_files(args.models_dir, ".pthw")
    # if args.include is not None:
    #     models = filter_include(models, args.include)
    # if args.exclude is not None:
    #     models = filter_exclude(models, args.exclude)

    # pprint.pprint(models)

    # diverse_agents = load_models(models)
    # for agent in diverse_agents:
    #     agent.to(args.train_device)
    #     agent.sync_target_with_online()

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

    agent_br = r2d2_br.R2D2Agent(
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
        False,  # uniform priority
        args.off_belief,
        )

    agent_br.sync_target_with_online()
    # optim_br = torch.optim.Adam(agent_br.online_net.parameters(), lr=args.lr, eps=args.eps)

    print('Best Response agent: ', agent_br)

    agents = []
    for i in range(args.num_agents):
        agents.append(
            r2d2_br.R2D2Agent(
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
                False,  # uniform priority
                args.off_belief,
            )
        )
        agents[i].sync_target_with_online()
        agents[i] = agents[i].to(args.train_device)
    
    # Create a single optimizer for all agents
    all_parameters = []
    all_parameters.extend(list(agent_br.online_net.parameters()))
    for agent in agents:
        all_parameters.extend(list(agent.online_net.parameters()))
    optim = torch.optim.Adam(all_parameters, lr=args.lr, eps=args.eps)

    eval_agent = agent_br.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
    
    # print("Loading dataset...")
    train_dataset = LoadDataset(args)
    train_dataset = train_dataset.load()
    train_loader = DataLoader(train_dataset)
    batch_loader = torch.utils.data.DataLoader(
        dataset=train_loader,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_data_thread,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=args.prefetch
    )

    print("Dataset loaded")
    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    
    # Store the original div_weight value
    original_div_weight = args.div_weight

    # Check if cluster information already exists
    cluster_save_path = os.path.join(args.clu_mod_dir, 'best_cluster.pt')
    model_save_path = os.path.join(args.clu_mod_dir, 'best_model.pt')
    if os.path.exists(cluster_save_path) and os.path.exists(model_save_path):
        encoder, cluster_centers_tensor = load_cluster(cluster_save_path, model_save_path, args)
    else:
        #train the mi model
        encoder, cluster_centers_tensor = train_mi(args, batch_loader=batch_loader, epochs=110, beta=0.0)
        plot_tsne(batch_loader, cluster_save_path, model_save_path, args)
        encoder.eval()
    
    kmeans = cluster_data(batch_loader, encoder, args)

    # Function to compute distances and normalized weights
    def compute_distance_weights(batch_lstm_o, agent_idx):
        # Reshape lstm_o if needed (depends on batch structure)
        if len(batch_lstm_o.shape) > 2:
            batch_lstm_o = batch_lstm_o.reshape(batch_lstm_o.shape[0], -1)

        # Compute Euclidean distance to each cluster center
        distances = torch.cdist(batch_lstm_o, cluster_centers_tensor)
        
        # Get distance to this agent's assigned cluster
        agent_cluster_dist = distances[:, agent_idx]
        
        # Get minimum distance to any other cluster (excluding agent's cluster)
        other_indices = [i for i in range(distances.shape[1]) if i != agent_idx]
        other_distances = distances[:, other_indices]
        min_other_dist = other_distances.min(dim=1)[0]
        
        # Calculate ratio: distance to other clusters / distance to agent's cluster
        # Higher ratio means sample is close to agent's cluster but far from others
        # Add small epsilon to avoid division by zero
        epsilon = 1e-6
        distance_ratio = min_other_dist / (agent_cluster_dist + epsilon)
        
        # Scale the ratio with an exponential function to emphasize the difference
        # similarity = torch.exp(distance_ratio - 1.0)
        # Use linear scaling instead of exponential
        similarity = distance_ratio
        # Normalize weights to range [0, 1] using min-max normalization
        # Handle the case where all values are the same
        sim_min, sim_max = similarity.min(), similarity.max()
        if sim_min == sim_max:
            weights = torch.ones_like(similarity)
        else:
            weights = (similarity - sim_min) / (sim_max - sim_min)
        
        return weights

    #train the agents
    for epoch in range(args.num_epoch+1):
        # Decay bc_weight if decay_bc is set
        if args.cp_bc_decay_factor > 0 and epoch >= args.cp_bc_decay_start:
            # Linear decay from initial bc_weight to eps
            original_cp_bc_weight = args.cp_bc_weight
            args.cp_bc_weight = args.cp_bc_weight * args.cp_bc_decay_factor + args.eps * (1 - args.cp_bc_decay_factor)
            print(f"Epoch {epoch}: Decaying cp_bc_weight from {original_cp_bc_weight:.6f} to {args.cp_bc_weight:.6f}")
        
        # Start using div_weight after start_div epochs
        if epoch == args.start_div and args.start_div > 0:
            args.div_weight = original_div_weight
            print(f"Epoch {epoch}: Starting to use div_weight: {args.div_weight:.6f}")
        elif epoch < args.start_div:
            args.div_weight = 0.0
        
        # print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()
        # if epoch % args.load_after == 0:
        #     chunked_loader.load_next_chunk()
        # epoch_bar = tqdm(chunked_loader.current_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        epoch_bar = tqdm(batch_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        for batch_idx, sample in enumerate(epoch_bar):
            # print("Buffer size: ", replay_buffer.size())
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                print("\nUpdated the target with online\n")
                agent_br.sync_target_with_online()
                for i in range(args.num_agents):
                    agents[i].sync_target_with_online()

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")
            
            start = time.time()
            batch, weight = process_batch(sample, args)
            stopwatch.time("sample data")
            # print("sample time: ", time.time() - start)

            # Get batch lstm_o from agent_bc for distance calculation
            # Forward pass through agent_bc to get lstm_o for this batch
            with torch.no_grad():
                batch_lstm_o = encode_batch(batch, encoder)
                batch_lstm_o = batch_lstm_o.detach()
            
            # Store losses for each agent to combine later
            agent_losses = []
            agent_loss_weights = []
            
            # Run forward and backward passes for all agents in parallel
            for i in range(args.num_agents):
                # # Calculate distance-based weights for this agent
                # agent_weights = compute_distance_weights(batch_lstm_o, i)

                # # threshold the weights
                # agent_weights = torch.where(agent_weights > args.wgt_thr, torch.ones_like(agent_weights), 0.00 * torch.ones_like(agent_weights))
                # agent_loss_weights.append(agent_weights)

                agent_weights = kmeans.predict(batch_lstm_o.cpu().numpy())
                agent_weights = (agent_weights == i).astype(float)
                agent_weights = torch.from_numpy(agent_weights).to(args.train_device)
                agent_loss_weights.append(agent_weights)
                loss, bc_loss, cql_loss, priority, _, _ = agents[i].loss(batch, args.aux_weight, stat, args.bc, args.cql)

                # Weight losses by distance-based weights
                weighted_loss_sp = args.sp_weight * loss * agent_weights
                weighted_loss_bc = args.bc_weight * bc_loss * agent_weights
                weighted_loss_cql = args.cql_weight * cql_loss * agent_weights
                agent_losses.append((weighted_loss_sp, weighted_loss_bc, weighted_loss_cql))
                
            # Aggregate all agent losses with their weights
            loss_sp = sum([loss_tuple[0] for loss_tuple in agent_losses])
            loss_bc = sum([loss_tuple[1] for loss_tuple in agent_losses])
            loss_cql = sum([loss_tuple[2] for loss_tuple in agent_losses])

            # Apply the agent's loss calculation
            loss_cp, online_q_values, valid_masks = train_br_agent(agent_br, agents, agent_loss_weights, batch, args)
            
            loss_cp = args.cp_weight * loss_cp

            loss_div = torch.zeros(len(weight), device=args.train_device)
            if args.div and args.div_weight > 0:
                # Calculate diversity loss separately for each agent, with that agent's Q-values having gradients
                agent_div_losses = []
                for i in range(args.num_agents):
                    div_loss = diversity_loss(online_q_values, valid_masks, args, i)
                    agent_div_losses.append(div_loss * agent_loss_weights[i])
                
                # Sum the diversity losses from all agents
                loss_div = args.div_weight * sum(agent_div_losses)
                
            loss = loss_cp + loss_sp + loss_bc + loss_cql - loss_div
            
            # Weight by importance sampling weights and take mean
            loss = (loss * weight).mean()
            loss.backward()
            
            # Stats tracking
            stat["Self-play loss"].feed(loss_sp.mean().detach().item())
            stat["Cross-play loss"].feed(loss_cp.mean().detach().item())
            stat["Behavior Cloning loss"].feed(loss_bc.mean().detach().item())
            stat["CQL loss"].feed(loss_cql.mean().detach().item())
            stat["Diversity loss"].feed(loss_div.mean().detach().item())
            stat["loss"].feed(loss.detach().item())

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent_br.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            stat["grad_norm"].feed(g_norm)

            del batch, weight
            if batch_idx % 100 == 0:
                if torch.cuda.memory_allocated(args.train_device) > 0.9 * torch.cuda.get_device_properties(args.train_device).total_memory:
                    torch.cuda.empty_cache()
                    gc.collect()
        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        # tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()
        stat.summary(epoch)

        if epoch % args.num_eval_after == 0:
            eval_seed = (9917 + epoch * 999999) % 7777777

            #Evaluate the ensemble agent
            eval_agent = eval_agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
            eval_agent.load_state_dict(agent_br.state_dict())

            score_br, perfect_br, *_ = evaluate(
                [eval_agent for _ in range(args.num_player)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
                device=args.train_device
            )
            
            print(f"BR Agent eval score: {score_br:.4f}, perfect: {perfect_br * 100:.2f}%")

            # evaluate the agents
            scores = []
            perfects = []
            for i in range(args.num_agents):
                eval_agent = eval_agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
                eval_agent.load_state_dict(agents[i].state_dict())
                score, perfect, *_ = evaluate(
                    [eval_agent for _ in range(args.num_player)],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device=args.train_device    
                )
                print(f"Agent {i} eval score: {score:.4f}, perfect: {perfect * 100:.2f}%")
                scores.append(score)
                perfects.append(perfect)
            
        force_save_name = None
        if epoch % args.save_model_after == 0:
            #save the model
            agent_force_save_name = f"model_seed_{args.seed}_agent_br_epoch_{epoch}"
            agent_model_saved = saver.save(
                None, agent_br.online_net.state_dict(), score_br, force_save_name=agent_force_save_name
            )
            print(f"epoch {epoch}, agent br eval score: {score_br:.4f}, perfect: {perfect_br * 100:.2f}%, model saved: {agent_model_saved}")
            
            # Save models for each agent individually
            for i in range(args.num_agents):
                agent_force_save_name = f"model_seed_{args.seed}_agent_{i}_epoch_{epoch}"
                agent_model_saved = saver.save(
                    None, agents[i].online_net.state_dict(), scores[i], force_save_name=agent_force_save_name
                )
                print(
                    f"epoch {epoch}, agent {i} eval score: {scores[i]:.4f}, perfect: {perfects[i] * 100:.2f}%, model saved: {agent_model_saved}"
                ) 
        print("==========")