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
from losses import cp_loss, diversity_loss
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
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
    parser.add_argument("--decay_bc", type=int, default=0, help="Number of epochs to decay bc_weight from initial value to eps")

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

    args = parser.parse_args()
    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]
    return args

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

    agent_bc = r2d2_br.R2D2Agent(
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

    agent_bc.sync_target_with_online()
    optim_bc = torch.optim.Adam(agent_bc.online_net.parameters(), lr=args.lr, eps=args.eps)

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
    for agent in agents:
        all_parameters.extend(list(agent.online_net.parameters()))
    optim = torch.optim.Adam(all_parameters, lr=args.lr, eps=args.eps)
    
    print('Best Response agent: ', agents[0])

    if args.load_model and args.load_model != "None":
        if args.off_belief and args.belief_model != "None":
            belief_config = utils.get_train_config(args.belief_model)
            args.load_model = belief_config["policy"]

        print("*****loading pretrained model*****")
        print(args.load_model)
        utils.load_weight(agents[0].online_net, args.load_model, args.train_device)
        print("*****done*****")

    # use clone bot for additional bc loss
    if args.clone_bot and args.clone_bot != "None":
        clone_bot = utils.load_supervised_agent(args.clone_bot, args.train_device)
    else:
        clone_bot = None

    eval_agent = agents[0].clone(args.train_device, {"vdn": False, "boltzmann_act": False})

    belief_model = None
    if args.off_belief and args.belief_model != "None":
        print(f"load belief model from {args.belief_model}")
        from belief_model import ARBeliefModel

        belief_devices = args.belief_device.split(",")
        belief_config = utils.get_train_config(args.belief_model)
        belief_model = []
        for device in belief_devices:
            belief_model.append(
                ARBeliefModel.load(
                    args.belief_model,
                    device,
                    5,
                    args.num_fict_sample,
                    belief_config["fc_only"],
                )
            )
    
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

    # Function to load BC agent model
    def load_bc_agent(agent, model_path, device):
        """Load the BC agent model from a saved state dict."""
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            agent.online_net.load_state_dict(state_dict)
            print(f"BC agent model loaded from {model_path}")
            return True
        else:
            print(f"BC agent model not found at {model_path}")
            return False

    # Check if cluster information already exists
    cluster_save_path = os.path.join(args.save_dir, 'lstm_o_clusters.pt')
    if os.path.exists(cluster_save_path):
        print(f"Found existing cluster information at {cluster_save_path}")
        cluster_data = torch.load(cluster_save_path)
        cluster_centers = cluster_data['cluster_centers']
        cluster_labels = cluster_data['cluster_labels']
        print("\nCluster Statistics:")
        unique_labels, counts = np.unique(cluster_labels.numpy(), return_counts=True)
        n_samples = len(cluster_labels)
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples ({count/n_samples*100:.2f}%)")
        
        # Convert to PyTorch tensors for efficient computation
        cluster_labels_tensor = cluster_labels
        cluster_centers_tensor = cluster_centers.to(args.train_device)
        
        # Load the BC agent model if available and not already loaded from command line
        bc_model_path = os.path.join(args.save_dir, f"bc_model_seed_{args.seed}_epoch_15.pthw")
        if os.path.exists(bc_model_path):
            if load_bc_agent(agent_bc, bc_model_path, args.train_device):
                print(f"Using pre-trained BC agent from {bc_model_path}")
            else:
                print("No pre-trained BC agent found, will train from scratch.")
            print("================================================")
    else:
        #train the behavior policy
        epoch_len = 15
        for epoch in range(epoch_len+1):
            tachometer.start()
            stat.reset()
            stopwatch.reset()
            epoch_bar = tqdm(batch_loader, desc=f'BC Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
            for batch_idx, sample in enumerate(epoch_bar):
                num_update = batch_idx + epoch * epoch_len
                if num_update % args.num_update_between_sync == 0:
                    print("\nUpdated the bc target with online\n")
                    agent_bc.sync_target_with_online()

                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

                batch, weight = process_batch(sample, args)
                stopwatch.time("sample data")
                
                #Behavior Cloning + Self Play
                loss_bc = torch.zeros(len(weight), device=args.train_device)
                loss_sp = torch.zeros(len(weight), device=args.train_device)
                sp_loss, bc_loss, priority, online_q, lstm_o = agent_bc.loss(batch, args.aux_weight, stat, args.bc)

                # Get the lstm_o for the behavior policy
                lstm_o = lstm_o.detach()

                loss_bc += bc_loss
                loss_sp += sp_loss

                loss_bc = args.bc_weight*loss_bc
                loss_sp = args.sp_weight*loss_sp

                loss = loss_bc + loss_sp

                loss = (loss * weight).mean()
                loss.backward()

                torch.cuda.synchronize()
                stopwatch.time("forward & backward")

                g_norm = torch.nn.utils.clip_grad_norm_(agent_bc.online_net.parameters(), args.grad_clip)
                optim_bc.step()
                optim_bc.zero_grad()

                torch.cuda.synchronize()
                stopwatch.time("update model")

                stat["BC agent Behavior Cloning loss"].feed(loss_bc.mean().detach().item())
                stat["BC agent Self Play loss"].feed(loss_sp.mean().detach().item())
                stat["BC agent loss"].feed(loss.detach().item())
                stat["BC agent grad_norm"].feed(g_norm)

                del batch, weight
                if batch_idx % 100 == 0:
                    if torch.cuda.memory_allocated(args.train_device) > 0.9 * torch.cuda.get_device_properties(args.train_device).total_memory:
                        torch.cuda.empty_cache()
                        gc.collect()

            print("EPOCH: %d" % epoch)
            stopwatch.summary()
            stat.summary(epoch)

            if epoch % args.num_eval_after == 0:
                eval_seed = (9917 + epoch * 999999) % 7777777
                
                # Create a copy of eval_agent and load the state dict of the current agent
                eval_agent = eval_agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
                eval_agent.load_state_dict(agent_bc.state_dict())

                score, perfect, *_ = evaluate(
                    [eval_agent for _ in range(args.num_player)],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device=args.train_device,
                )
            else:
                score = 0
                perfect = 0
            
            # Save the bc model
            if epoch % epoch_len == 0:
                force_save_name = f"bc_model_seed_{args.seed}_epoch_{epoch}"
            model_saved = saver.save(
                None, agent_bc.online_net.state_dict(), score, force_save_name=force_save_name
            )
            
            print(f"Epoch {epoch}, score: {score:.4f}, perfect: {perfect * 100:.2f}%, model saved: {model_saved}")
            print("================================================")

        #load the bc dataset
        bc_batch_loader = torch.utils.data.DataLoader(
            dataset=train_loader,
            batch_size=args.batchsize,
            shuffle=False,
            num_workers=args.num_data_thread,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=args.prefetch
        )

        #find the lstm_o for the behavior policy for each sequence in the dataset 
        all_lstm_o = []
        #write a tqdm for this
        epoch_bar = tqdm(bc_batch_loader, desc=f'Finding lstm_o', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        for batch_idx, sample in enumerate(epoch_bar):
            batch, weight = process_batch(sample, args)
            _, _, _, _, lstm_o = agent_bc.loss(batch, args.aux_weight, stat, args.bc)
            lstm_o = lstm_o.detach()
            all_lstm_o.append(lstm_o)
        
        # Concatenate all lstm_o tensors
        all_lstm_o = torch.cat(all_lstm_o, dim=0)
        
        # Save the concatenated tensor
        save_path = os.path.join(args.save_dir, 'lstm_o.pt')
        torch.save(all_lstm_o, save_path)
        print(f"Saved lstm_o tensor of shape {all_lstm_o.shape} to {save_path}")
        
        # Reshape lstm_o for clustering (flatten the last dimension)
        lstm_o_np = all_lstm_o.cpu().numpy()
        n_samples = lstm_o_np.shape[0]
        lstm_o_reshaped = lstm_o_np.reshape(n_samples, -1)
        
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=args.num_agents, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(lstm_o_reshaped)
        
        # Save cluster centers and labels
        cluster_centers = kmeans.cluster_centers_
        torch.save({
            'cluster_centers': torch.from_numpy(cluster_centers),
            'cluster_labels': torch.from_numpy(cluster_labels),
            'k': args.num_agents
        }, cluster_save_path)
        
        # Print cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        print("\nCluster Statistics:") 
        for label, count in zip(unique_labels, counts):
            print(f"Cluster {label}: {count} samples ({count/n_samples*100:.2f}%)")
        print(f"\nSaved cluster information to {cluster_save_path}")

        # Convert to PyTorch tensors for efficient computation
        cluster_labels_tensor = torch.from_numpy(cluster_labels)
        cluster_centers_tensor = torch.from_numpy(cluster_centers).to(args.train_device)
    
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
        if args.decay_bc > 0 and epoch <= args.decay_bc:
            # Linear decay from initial bc_weight to eps
            decay_factor = 1.0 - (epoch / args.decay_bc)
            original_bc_weight = args.bc_weight
            args.bc_weight = args.bc_weight * decay_factor + args.eps * (1 - decay_factor)
            print(f"Epoch {epoch}: Decaying bc_weight from {original_bc_weight:.6f} to {args.bc_weight:.6f}")
        
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
                _, _, _, _, batch_lstm_o = agent_bc.loss(batch, args.aux_weight, stat, False)
                batch_lstm_o = batch_lstm_o.detach()
            
            # Store losses for each agent to combine later
            agent_losses = []
            agent_loss_weights = []
            
            # Run forward and backward passes for all agents in parallel
            for i in range(args.num_agents):
                # Calculate distance-based weights for this agent
                agent_weights = compute_distance_weights(batch_lstm_o, i)

                # threshold the weights
                agent_weights = torch.where(agent_weights > args.wgt_thr, torch.ones_like(agent_weights), 0.01 * torch.ones_like(agent_weights))
                agent_loss_weights.append(agent_weights)
                
                # Apply the agent's loss calculation
                loss, bc_loss, priority, _, _ = agents[i].loss(batch, args.aux_weight, stat, args.bc)
                
                # Weight losses by distance-based weights
                weighted_loss_sp = args.sp_weight * loss * agent_weights
                weighted_loss_bc = args.bc_weight * bc_loss * agent_weights
                
                # Store for later aggregation
                agent_losses.append((weighted_loss_sp, weighted_loss_bc))
                
                # # Record per-agent statistics 
                # stat[f"Agent {i} weight mean"].feed(agent_weights.mean().item())
                # stat[f"Agent {i} weight std"].feed(agent_weights.std().item())
                
            # Cross Play computation between all agent pairs
            loss_cp = torch.zeros(len(weight), device=args.train_device)
            if args.num_agents > 1 and args.cp:
                online_q_values = [None] * args.num_agents  # Initialize with None for each agent
                valid_masks = [None] * args.num_agents  # Initialize with None for each agent
                
                # Compute cross-play losses between all agent pairs
                cp_losses = []
                for i in range(args.num_agents):
                    for j in range(args.num_agents):
                        if i==j:
                            continue
                        cp_value, ag1_online_q, ag2_online_q, valid_mask = cp_loss(agents[i], agents[j], batch, stat, args)
                        cp_losses.append(cp_value * agent_loss_weights[i])

                        # Store Q-values only if they haven't been stored yet
                        if online_q_values[i] is None:
                            online_q_values[i] = ag1_online_q
        
                        #Store the valid masks if they haven't been stored yet
                        if valid_masks[i] is None:
                            valid_masks[i] = valid_mask
                        
                if cp_losses:
                    loss_cp = args.cp_weight * sum(cp_losses)
                
            # Diversity loss
            loss_div = torch.zeros(len(weight), device=args.train_device)
            if args.div and args.div_weight > 0:
                # Calculate diversity loss separately for each agent, with that agent's Q-values having gradients
                agent_div_losses = []
                for i in range(args.num_agents):
                    div_loss = diversity_loss(online_q_values, valid_masks, args, i)
                    agent_div_losses.append(div_loss * agent_loss_weights[i])
                
                # Sum the diversity losses from all agents
                loss_div = args.div_weight * sum(agent_div_losses)

            # Aggregate all agent losses with their weights
            loss_sp = sum([loss_tuple[0] for loss_tuple in agent_losses])
            loss_bc = sum([loss_tuple[1] for loss_tuple in agent_losses])
            
            # Combine all loss components
            loss = loss_sp + loss_bc + loss_cp - loss_div
                
            # Weight by importance sampling weights and take mean
            loss = (loss * weight).mean()
            loss.backward()
            
            # Stats tracking
            stat["Self-play loss"].feed(loss_sp.mean().detach().item())
            stat["Cross-play loss"].feed(loss_cp.mean().detach().item())
            stat["Diversity loss"].feed(loss_div.mean().detach().item())
            stat["Behavior Cloning loss"].feed(loss_bc.mean().detach().item())
            stat["loss"].feed(loss.detach().item())

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                all_parameters, args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            # start = time.time()
            # replay_buffer.update_priorities(priority)   
            # stopwatch.time("updating priority")
            # print("update priority time: ", time.time() - start)

            stat["grad_norm"].feed(g_norm)
            # stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

            # Add periodically to track memory usage
            # print(f"Memory allocated on {args.train_device}: {torch.cuda.memory_allocated(args.train_device) / 1e9:.2f} GB")
            # print(f"Memory reserved on {args.train_device}: {torch.cuda.memory_reserved(args.train_device) / 1e9:.2f} GB")

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
            
            # Evaluate each agent individually
            agent_scores = []
            agent_perfects = []
            
            for i in range(args.num_agents):
                # Create a copy of eval_agent and load the state dict of the current agent
                current_eval_agent = eval_agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
                current_eval_agent.load_state_dict(agents[i].state_dict())
                
                # Evaluate the agent
                score, perfect, *_ = evaluate(
                    [current_eval_agent for _ in range(args.num_player)],
                    1000,
                    eval_seed,
                    args.eval_bomb,
                    0,  # explore eps
                    args.sad,
                    args.hide_action,
                    device=args.train_device,
                )
                
                agent_scores.append(score)
                agent_perfects.append(perfect)
                
                print(f"Agent {i} eval score: {score:.4f}, perfect: {perfect * 100:.2f}%")
            
            # Use agent 0's score for saving the model
            score = agent_scores[0]
            perfect = agent_perfects[0]
        else:
            score = 0
            perfect = 0
        
        force_save_name = None
        if epoch % args.save_model_after == 0:
            # Save models for each agent individually
            for i in range(args.num_agents):
                agent_force_save_name = f"model_seed_{args.seed}_agent_{i}_epoch_{epoch}"
                agent_model_saved = saver.save(
                    None, agents[i].online_net.state_dict(), agent_scores[i], force_save_name=agent_force_save_name
                )
                print(
                    f"epoch {epoch}, agent {i} eval score: {agent_scores[i]:.4f}, perfect: {agent_perfects[i] * 100:.2f}%, model saved: {agent_model_saved}"
                )
                
        # save_rb_epochs = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
        # if epoch in save_rb_epochs:
        #     save_replay_buffer(replay_buffer, epoch, score)

        if clone_bot is not None:
            score, perfect, *_ = evaluate(
                [clone_bot] + [eval_agent for _ in range(args.num_player - 1)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
            )
            print(f"clone bot score: {np.mean(score)}")

        # if args.off_belief:
        #     actors = common_utils.flatten(act_group.actors)
        #     success_fict = [actor.get_success_fict_rate() for actor in actors]
        #     print(
        #         "epoch %d, success rate for sampling ficticious state: %.2f%%"
        #         % (epoch, 100 * np.mean(success_fict))
        #     )
        # print("Max priority: ", replay_buffer.max_priority)

        print("==========")