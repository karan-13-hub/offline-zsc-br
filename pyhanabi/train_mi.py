import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from replay_load import LoadDataset, DataLoader, process_batch
import argparse
import numpy as np
from tqdm import tqdm
import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
import datetime
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import gc
from sklearn.metrics import silhouette_score
from g_means_clustering import gmeans_clustering

def setup_logging(log_dir):
    """Set up comprehensive logging for training"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create timestamp for log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Training log started - Log file: {log_file}")
    return logger, log_file

def capture_and_log_output(logger, func, *args, **kwargs):
    """Capture stdout/stderr from a function and log it"""
    # Create string buffers to capture output
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    
    # Redirect stdout and stderr to buffers
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    
    # Get the captured output
    stdout_output = stdout_buffer.getvalue()
    stderr_output = stderr_buffer.getvalue()
    
    # Log the captured output
    if stdout_output.strip():
        logger.info("Dataset loading output:")
        for line in stdout_output.strip().split('\n'):
            if line.strip():
                logger.info(f"  {line}")
    
    if stderr_output.strip():
        logger.warning("Dataset loading warnings/errors:")
        for line in stderr_output.strip().split('\n'):
            if line.strip():
                logger.warning(f"  {line}")
    
    return result

def check_gpu_memory():
    """Check available GPU memory and return percentage free"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - allocated_memory
        free_percentage = (free_memory / total_memory) * 100
        return free_percentage
    return 100.0

class AsyncModelSaver:
    """Handles asynchronous model saving, clustering, and plotting operations"""
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
    
    def submit_save_operations(self, save_cluster_path, model_save_path, best_encoder, best_decoder, train_dataset, args):
        """Submit save operations to run asynchronously"""
        print("Started asynchronous saving operations")
        logging.info("Started asynchronous saving operations")
        
        # Save model state dicts (CPU tensors) - this is fast and can be done synchronously
        torch.save({
            "encoder": best_encoder.state_dict(), 
            "decoder": best_decoder.state_dict()
        }, model_save_path)
        
        # Submit clustering and plotting operations to run asynchronously
        # Pass train_dataset instead of batch_loader to create new DataLoader instances
        cluster_future = self.executor.submit(self._save_cluster_async, save_cluster_path, best_encoder, train_dataset, args)
        plot_future = self.executor.submit(self._plot_tsne_async, train_dataset, save_cluster_path, model_save_path, args)
        
        self.futures.extend([cluster_future, plot_future])
        return cluster_future, plot_future
    
    def _save_cluster_async(self, save_cluster_path, encoder, train_dataset, args):
        """Async version of save_cluster"""
        try:
            # Set CUDA device for this thread
            if torch.cuda.is_available():
                torch.cuda.set_device(args.train_device)
            
            # Create a new DataLoader instance for this thread
            train_loader = DataLoader(train_dataset)
            batch_loader = torch.utils.data.DataLoader(
                dataset=train_loader,
                batch_size=min(args.batchsize, args.async_batch_size),  # Use smaller batch size for async operations
                shuffle=False,  # No need to shuffle for clustering
                num_workers=0,  # Use 0 workers to avoid multiprocessing issues
                pin_memory=False
            )
            result = save_cluster(save_cluster_path, encoder, batch_loader, args)
            print("Completed saving cluster")
            logging.info("Completed saving cluster")
            return result
        except Exception as e:
            print(f"Error in async clustering: {e}")
            logging.error(f"Error in async clustering: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_tsne_async(self, train_dataset, cluster_save_path, model_save_path, args):
        """Async version of plot_tsne"""
        try:
            # Set CUDA device for this thread
            if torch.cuda.is_available():
                torch.cuda.set_device(args.train_device)
            
            # Create a new DataLoader instance for this thread
            train_loader = DataLoader(train_dataset)
            batch_loader = torch.utils.data.DataLoader(
                dataset=train_loader,
                batch_size=min(args.batchsize, args.async_batch_size),  # Use smaller batch size for async operations
                shuffle=False,  # No need to shuffle for plotting
                num_workers=0,  # Use 0 workers to avoid multiprocessing issues
                pin_memory=False
            )
            plot_tsne(batch_loader, cluster_save_path, model_save_path, args)
            print("Completed plotting")
            logging.info("Completed plotting")
            return True
        except Exception as e:
            print(f"Error in async plotting: {e}")
            logging.error(f"Error in async plotting: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def cleanup_completed_futures(self):
        """Clean up completed futures to prevent memory accumulation"""
        completed_futures = []
        for future in self.futures:
            if future.done():
                completed_futures.append(future)
                # Check if the future completed with an error
                try:
                    future.result()  # This will raise an exception if the future failed
                except Exception as e:
                    print(f"Async operation failed: {e}")
        
        for future in completed_futures:
            self.futures.remove(future)
    
    def get_status(self):
        """Get status of async operations"""
        completed = sum(1 for f in self.futures if f.done())
        total = len(self.futures)
        return f"Async operations: {completed}/{total} completed"
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)

def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/br_medium_data_seed_9_42_1234_1e9+7/vdn_cp_bc_0.4")
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
    parser.add_argument("--clu_mod_dir", type=str, default="exps/br_medium_data_seed_9_42_1234_1e9+7/")
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
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:2")
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
    parser.add_argument("--reqd_con_eps", type=int, default=10, help="Number of epochs to wait for consecutive small gradient norms")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta (KL weight)")
    parser.add_argument("--grad_norm_threshold", type=float, default=1e-5, help="Gradient norm threshold")
    parser.add_argument("--async_batch_size", type=int, default=64, help="Batch size for async operations")

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

# Encoder Network
class TrajectoryEncoder(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, latent_dim):
        super().__init__()
        self.action_dim = action_dim
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=1, batch_first=False)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, observations, actions):
        obs_enc = self.obs_encoder(observations)
        actions_one_hot = F.one_hot(actions, self.action_dim).float()
        actions_enc = self.action_encoder(actions_one_hot)
        observations_actions = torch.cat([obs_enc, actions_enc], dim=-1)
        _, (h_n, _) = self.lstm(observations_actions)
        h_n = h_n.squeeze(0)
        mu = self.fc_mu(h_n)
        sigma = F.softplus(self.fc_sigma(h_n)).clamp(min=1e-3, max=10)
        return mu, sigma

# Decoder Network
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, obs_dim, action_dim, reward_dim):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        self.latent_encoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
        )
        self.action_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, action_dim)
        )
        self.reward_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, reward_dim),
            # nn.Sigmoid()
        )       
        # self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, z, obs):
        # import pdb; pdb.set_trace()
        z = z.unsqueeze(0).repeat(obs.size(0), 1, 1)
        obs_enc = self.obs_encoder(obs)
        latent_enc = self.latent_encoder(z)
        concat_enc = torch.cat([obs_enc, latent_enc], dim=-1)
        # Predict actions
        actions_pred = self.action_out(concat_enc)
        # Predict rewards
        rewards_pred = self.reward_out(concat_enc)
        return actions_pred, rewards_pred

# Corrected CLUB Mutual Information estimator (for I(z;s))
class CLUBEstimator(nn.Module):
    def __init__(self, obs_dim, latent_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=False)
        self.net_mu = nn.Linear(hidden_dim, latent_dim)
        self.net_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, s):
        _, (h_n, _) = self.lstm(s)
        h_n = h_n.squeeze(0)
        mu = self.net_mu(h_n)
        logvar = self.net_logvar(h_n).clamp(min=-5, max=5)
        return mu, logvar


def calc_actual_rewards(dataset_rewards, args):
    # dataset rewards is a multi step reward, we need to find actual rewards
    actual_dataset_rewards = []
    # Calculate the discount factors once
    discount_factors = np.array([1.0 / (args.gamma ** i) for i in range(1,args.multi_step)])
    for dataset_reward in tqdm(dataset_rewards):
        # Convert to numpy for vectorized operations
        reward_array = dataset_reward.numpy()
        reward_length = len(reward_array)
        
        # Create a copy to store the adjusted rewards
        adjusted_rewards = np.zeros_like(reward_array)
    
        
        # Vectorized calculation for each position
        for i in range(reward_length - args.multi_step):
            pos = reward_length - args.multi_step - i
            # Calculate the discounted sum of future rewards
            future_rewards = adjusted_rewards[pos+1:pos+args.multi_step]
            discounted_sum = np.sum(future_rewards * discount_factors[:len(future_rewards)])
            # Adjust the current reward
            adjusted_rewards[pos] = reward_array[pos] - discounted_sum
        # Convert back to tensor and add to the new dataset
        adjusted_rewards = adjusted_rewards.astype(int)
        actual_dataset_rewards.append(adjusted_rewards)
    actual_dataset_rewards = np.array(actual_dataset_rewards)
    actual_dataset_rewards = torch.from_numpy(actual_dataset_rewards).to(torch.float32, non_blocking=True)
    return actual_dataset_rewards

def encode_batch(batch, encoder):
    observations, actions_dict, rewards, seq_len = batch.obs, batch.action, batch.reward, batch.seq_len
    priv_s = observations["priv_s"]
    publ_s = observations["publ_s"]
    legal_move = observations["legal_move"]
    actions = actions_dict["a"]
    max_seq_len = priv_s.size(0)
    batch_size = priv_s.size(1)

    # Remove actions with value 20 (inaction)
    valid_action_mask = (actions != 20)
    
    # Create alternating pattern based on first row
    # Get the pattern from first row
    first_row_pattern = valid_action_mask[0]  # Shape: [batch_size, 2]
    
    # Create alternating pattern for each sequence
    # Create a tensor of alternating True/False values for each timestep
    alternating_timesteps = torch.arange(max_seq_len, device=valid_action_mask.device) % 2 == 0
    alternating_timesteps = alternating_timesteps.view(-1, 1)  # Shape: [max_seq_len, 1]
    
    # Expand first row pattern to match sequence length
    expanded_pattern = first_row_pattern.unsqueeze(0).expand(max_seq_len, -1, -1)  # Shape: [max_seq_len, batch_size, 2]
    
    # Create alternating mask using broadcasting
    alternating_mask = torch.where(alternating_timesteps.unsqueeze(-1), expanded_pattern, ~expanded_pattern)
    valid_action_mask = valid_action_mask & alternating_mask

    actions = actions[valid_action_mask]
    actions = actions.view(max_seq_len, batch_size)

    priv_s = priv_s[valid_action_mask]
    priv_s = priv_s.view(max_seq_len, batch_size, -1)
    publ_s = publ_s[valid_action_mask]
    publ_s = publ_s.view(max_seq_len, batch_size, -1)
    legal_move = legal_move[valid_action_mask]
    legal_move = legal_move.view(max_seq_len, batch_size, -1)            
    
    obs = torch.cat([priv_s, publ_s, legal_move], dim=-1)
    obs_rew = torch.cat([obs, rewards.unsqueeze(-1)], dim=-1)

    # Encode
    mu, sigma = encoder(obs_rew, actions)
    # z = mu + sigma * torch.randn_like(mu)
    z = mu #karan
    return z


def train_mi(args, batch_loader, train_dataset, epochs, beta):    
    # Training Loop
    encoder = TrajectoryEncoder(obs_dim=1213, action_dim=21, hidden_dim=512, latent_dim=512)
    decoder = Decoder(latent_dim=512, hidden_dim=512, obs_dim=1212, action_dim=21, reward_dim=1)
    # club_estimator = CLUBEstimator(obs_dim=1212, latent_dim=512, hidden_dim=512)

    encoder.to(args.train_device, non_blocking=True)
    decoder.to(args.train_device, non_blocking=True)
    # club_estimator.to(args.train_device, non_blocking=True)
    
    all_params = list(encoder.parameters()) + list(decoder.parameters()) #+ list(club_estimator.parameters())
    optimizer = optim.Adam(all_params, lr=5e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)

    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    # bce_loss = nn.BCELoss(reduction='mean')
    # mse_loss = nn.MSELoss(reduction='mean')

    summary_writer = SummaryWriter(log_dir=args.clu_mod_dir)
    
    # Initialize async model saver
    async_saver = AsyncModelSaver(max_workers=2)

    best_loss = float('inf')
    
    # Early stopping variables
    consecutive_small_grad_epochs = 0
    
    logger.info("Starting training...")

    avg_loss = 0
    avg_action_loss = 0
    avg_kl_loss = 0
    for epoch in tqdm(range(epochs+1)):
        encoder.train()
        decoder.train()
        epoch_bar = tqdm(batch_loader, desc=f'Clustering {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        for batch_idx, sample in enumerate(epoch_bar):
            batch, _ = process_batch(sample, args)
            observations, actions_dict, rewards, seq_len = batch.obs, batch.action, batch.reward, batch.seq_len
            priv_s = observations["priv_s"]
            publ_s = observations["publ_s"]
            legal_move = observations["legal_move"]
            actions = actions_dict["a"]
            max_seq_len = priv_s.size(0)
            batch_size = priv_s.size(1)

            # Remove actions with value 20 (inaction)
            valid_action_mask = (actions != 20)
            
            # Create alternating pattern based on first row
            # Get the pattern from first row
            first_row_pattern = valid_action_mask[0]  # Shape: [batch_size, 2]
            
            # Create alternating pattern for each sequence
            # Create a tensor of alternating True/False values for each timestep
            alternating_timesteps = torch.arange(max_seq_len, device=valid_action_mask.device) % 2 == 0
            alternating_timesteps = alternating_timesteps.view(-1, 1)  # Shape: [max_seq_len, 1]
            
            # Expand first row pattern to match sequence length
            expanded_pattern = first_row_pattern.unsqueeze(0).expand(max_seq_len, -1, -1)  # Shape: [max_seq_len, batch_size, 2]
            
            # Create alternating mask using broadcasting
            alternating_mask = torch.where(alternating_timesteps.unsqueeze(-1), expanded_pattern, ~expanded_pattern)

            valid_action_mask = valid_action_mask & alternating_mask
            actions = actions[valid_action_mask]
            actions = actions.view(max_seq_len, batch_size)

            priv_s = priv_s[valid_action_mask]
            priv_s = priv_s.view(max_seq_len, batch_size, -1)
            publ_s = publ_s[valid_action_mask]
            publ_s = publ_s.view(max_seq_len, batch_size, -1)
            legal_move = legal_move[valid_action_mask]
            legal_move = legal_move.view(max_seq_len, batch_size, -1)            
            
            obs = torch.cat([priv_s, publ_s, legal_move], dim=-1)
            obs_rew = torch.cat([obs, rewards.unsqueeze(-1)], dim=-1)

            # Encode
            mu, sigma = encoder(obs_rew, actions)
            z = mu + sigma * torch.randn_like(mu)

            # Decode
            actions_pred, _ = decoder(mu, obs)

            # Masking            
            mask = torch.arange(0, max_seq_len, device=seq_len.device)
            mask = (mask.unsqueeze(1) < seq_len.unsqueeze(0)).float()
            
            #apply mask to actions_pred and rewards_pred
            actions_pred = actions_pred * mask.unsqueeze(-1)
            # rewards_pred = rewards_pred * mask.unsqueeze(-1)

            # Reshape predictions and targets
            actions_pred = actions_pred.view(-1, actions_pred.size(-1))
            # rewards_pred = rewards_pred.squeeze(-1)
            actions = actions.view(-1)

            # Reconstruction losses
            action_recon_loss = ce_loss(actions_pred, actions)
            # reward_recon_loss = mse_loss(rewards_pred, player_rewards)

            # state_recon_loss = mse_loss(states_pred, observations).mean(dim=-1)
            # state_recon_loss = state_recon_loss.view(max_seq_len, -1)
            # state_recon_loss = (state_recon_loss * mask).mean()

            # KL divergence
            kl_loss = -0.5*(1 + torch.log(sigma**2) - mu**2 - sigma**2).mean()

            # # CLUB MI estimation
            # mu_club, logvar_club = club_estimator(concat_observation)
            # mi_loss = beta * 0.5 * ((z - mu_club)**2 / logvar_club.exp()).sum(dim=-1).mean()

            # Total loss
            # kl_loss = min(1.0, epoch / 50)*kl_loss
            loss = action_recon_loss + beta*kl_loss

            #update the average loss using moving average based on the batch index
            avg_loss = avg_loss + loss.item()
            avg_action_loss = avg_action_loss + action_recon_loss.item()
            avg_kl_loss = avg_kl_loss + kl_loss.item()

            optimizer.zero_grad()
            loss.backward()

            # Calculate gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
            
            # Gradient norm will be checked at the end of each epoch
            optimizer.step()
            
            # Clear intermediate tensors to free memory
            del batch, loss, action_recon_loss, kl_loss, actions_pred
            if 'z' in locals():
                del z
            if 'mu' in locals():
                del mu
            if 'sigma' in locals():
                del sigma
            if 'obs_rew' in locals():
                del obs_rew
            if 'obs' in locals():
                del obs
            if 'actions' in locals():
                del actions
            if 'mask' in locals():
                del mask
            
            # Clear GPU cache every 10 batches to prevent memory accumulation
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                # Check memory usage and log if low
                free_mem = check_gpu_memory()
                if free_mem < 10.0:  # Less than 10% free memory
                    logger.warning(f"Low GPU memory: {free_mem:.1f}% free at epoch {epoch}, batch {batch_idx}")

            epoch_bar.set_postfix(
                loss=avg_loss/(batch_idx + 1), 
                action_loss=avg_action_loss/(batch_idx + 1),
                # reward_loss=reward_recon_loss.item(),
                # state_loss=state_recon_loss.item(),
                kl_loss=avg_kl_loss/(batch_idx + 1), 
                grad_norm=grad_norm.item(),
                lr=scheduler.get_last_lr()[0]
                # mi_loss=mi_loss.item()
            )
        avg_loss = avg_loss / (batch_idx + 1)
        avg_action_loss = avg_action_loss / (batch_idx + 1)
        avg_kl_loss = avg_kl_loss / (batch_idx + 1)
        scheduler.step()
        
        # Clear GPU cache more frequently to prevent memory accumulation
        if epoch % 5 == 0:  # Clear cache every 5 epochs
            torch.cuda.empty_cache()
            gc.collect()

        #add the average loss & learning rate to the tensorboard
        summary_writer.add_scalar('loss', avg_loss, epoch)
        summary_writer.add_scalar('action_loss', avg_action_loss, epoch)
        summary_writer.add_scalar('kl_loss', avg_kl_loss, epoch)
        summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
        summary_writer.add_scalar('grad_norm', grad_norm.item(), epoch)
        summary_writer.add_scalar('consecutive_small_grad_epochs', consecutive_small_grad_epochs, epoch)
        
        # Log epoch summary
        logger.info(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f} | Action Loss: {avg_action_loss:.6f} | KL Loss: {avg_kl_loss:.6f} | Grad Norm: {grad_norm:.2e} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Check for early stopping based on consecutive small gradient norms
        if grad_norm < args.grad_norm_threshold:
            consecutive_small_grad_epochs += 1
            logger.warning(f"Epoch {epoch}: Small gradient norm ({grad_norm:.2e}) - consecutive count: {consecutive_small_grad_epochs}/{args.reqd_con_eps}")
            
            if consecutive_small_grad_epochs >= args.reqd_con_eps:
                logger.info(f"Training stopped at epoch {epoch}: gradient norm < {args.grad_norm_threshold} for {args.reqd_con_eps} consecutive epochs")
                # save the best encoder and decoder and plot the tsne
                save_cluster_path = os.path.join(args.clu_mod_dir, f'best_cluster.pt')
                model_save_path = os.path.join(args.clu_mod_dir, f'best_model.pt')
                logger.info("Saving final model and generating t-SNE plot...")
                plot_tsne(batch_loader, save_cluster_path, model_save_path, args)
                logger.info("Training completed successfully with early stopping")
                return encoder, None
        else:
            # Reset counter if gradient norm is not small
            consecutive_small_grad_epochs = 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder = encoder
            best_decoder = decoder
            logger.info(f"New best model at epoch {epoch} with loss: {best_loss:.6f}")
            
        if epoch % 100 == 0:
            try:
                save_cluster_path = os.path.join(args.clu_mod_dir, f'best_cluster.pt')
                model_save_path = os.path.join(args.clu_mod_dir, f'best_model.pt')
                
                # Submit save operations to run asynchronously
                cluster_future, plot_future = async_saver.submit_save_operations(
                    save_cluster_path, model_save_path, best_encoder, best_decoder, train_dataset, args
                )
                
                # Clean up any completed futures to prevent memory accumulation
                async_saver.cleanup_completed_futures()
                
                logger.info(f"Epoch {epoch}: Submitted async save operations - {async_saver.get_status()}")
                
            except Exception as e:
                logger.error(f"Error submitting async operations at epoch {epoch}: {e}")
            
            # Clear GPU cache after heavy operations
            torch.cuda.empty_cache()
            gc.collect()
    # Wait for any remaining async operations to complete and cleanup
    logger.info(f"Training completed. Final status: {async_saver.get_status()}")
    async_saver.shutdown()
    
    logger.info("="*50)
    logger.info("TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Final epoch: {epoch}")
    logger.info(f"Best loss achieved: {best_loss:.6f}")
    logger.info(f"Final gradient norm: {grad_norm:.2e}")
    logger.info(f"Consecutive small grad epochs: {consecutive_small_grad_epochs}")
    logger.info(f"Log file saved to: {log_file}")
    logger.info("="*50)
    
    return encoder, None  # cluster_centers_tensor will be None since we're not waiting for it


def save_cluster(save_cluster_path, encoder, batch_loader, args):
    all_lstm_o = []
    #write a tqdm for this
    epoch_bar = tqdm(batch_loader, desc=f'Finding lstm_o', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
    with torch.no_grad():
        encoder.eval()
        for _, sample in enumerate(epoch_bar):
            batch, _ = process_batch(sample, args)
            z = encode_batch(batch, encoder)
            all_lstm_o.append(z)
                
    # Concatenate all lstm_o tensors
    all_lstm_o = torch.cat(all_lstm_o, dim=0)
        
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
    }, save_cluster_path)
    
    # Print cluster statistics
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    print("\nCluster Statistics:") 
    for label, count in zip(unique_labels, counts):
        print(f"Cluster {label}: {count} samples ({count/n_samples*100:.2f}%)")
    print(f"\nSaved cluster information to {save_cluster_path}")

    # Convert to PyTorch tensors for efficient computation
    cluster_centers_tensor = torch.from_numpy(cluster_centers).to(args.train_device)
    return cluster_centers_tensor

def load_cluster(cluster_save_path, model_save_path, args):
        print(f"Found existing cluster information at {cluster_save_path}")
        cluster_data = torch.load(cluster_save_path)
        cluster_centers = cluster_data['cluster_centers']
        # cluster_labels = cluster_data['cluster_labels']
        # print("\nCluster Statistics:")
        # unique_labels, counts = np.unique(cluster_labels.numpy(), return_counts=True)
        # n_samples = len(cluster_labels)
        # for label, count in zip(unique_labels, counts):
        #     print(f"Cluster {label}: {count} samples ({count/n_samples*100:.2f}%)")
        
        # Convert to PyTorch tensors for efficient computation
        cluster_centers_tensor = cluster_centers.to(args.train_device)
        
        # Load the BC agent model if available and not already loaded from command line
        encoder = TrajectoryEncoder(obs_dim=1213, action_dim=21, hidden_dim=512, latent_dim=512)
        encoder.load_state_dict(torch.load(model_save_path)['encoder'])
        encoder = encoder.to(args.train_device)
        encoder.eval()
        print("Successfully loaded encoder")
        return encoder, cluster_centers_tensor

# def adaptive_gmeans_clustering(batch_loader, encoder, args):
#     all_lstm_o = []
#     epoch_bar = tqdm(batch_loader, desc=f'Finding lstm_o', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
#     with torch.no_grad():
#         encoder.eval()
#         for _, sample in enumerate(epoch_bar):
#             batch, _ = process_batch(sample, args)
#             z = encode_batch(batch, encoder)
#             all_lstm_o.append(z)
#     all_lstm_o = torch.cat(all_lstm_o, dim=0)
#     lstm_o_np = all_lstm_o.cpu().numpy()
#     n_samples = lstm_o_np.shape[0]
#     lstm_o_reshaped = lstm_o_np.reshape(n_samples, -1)

#     labels, k = gmeans_clustering(lstm_o_reshaped, seed=args.seed)
#     print(f"G-means adaptive clustering found k={k}")
#     return labels

# write a function to perform adaptive kmeans clustering
def adaptive_kmeans_clustering(batch_loader, encoder, args, min_k=2, max_k=10):
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

    best_k = None
    best_score = -1
    best_kmeans = None

    epoch_bar = tqdm(range(min_k, min(max_k, n_samples) + 1), desc=f'Finding best k', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
    for k in epoch_bar:  # cannot have more clusters than samples
        kmeans = KMeans(n_clusters=k, random_state=args.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(lstm_o_reshaped)
        
        # silhouette score requires at least 2 clusters
        if len(set(cluster_labels)) > 1:
            score = silhouette_score(lstm_o_reshaped, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k
                best_kmeans = kmeans
        
        postfix = {'k': k, 'score': score}
        epoch_bar.set_postfix(postfix)

    print(f"Adaptive KMeans picked k={best_k} with silhouette score={best_score:.4f}")
    return best_kmeans

def plot_tsne(batch_loader, cluster_save_path, model_save_path, args):
    #extract all the folder names except the last one
    folder_names = args.clu_mod_dir
    plt_save_path = 'tsne_plot.png'  # Define plt_save_path before using it
    plt_save_path = os.path.join(folder_names, plt_save_path)
    encoder, _ = load_cluster(cluster_save_path, model_save_path, args)
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
    # kmeans = KMeans(n_clusters=args.num_agents, random_state=args.seed, n_init=10)
    kmeans = adaptive_kmeans_clustering(batch_loader, encoder, args)
    cluster_labels = kmeans.fit_predict(lstm_o_reshaped)
    # cluster_labels = adaptive_gmeans_clustering(batch_loader, encoder, args)
    
    #plot the tsne
    tsne = TSNE(n_components=2, random_state=args.seed)
    lstm_o_tsne = tsne.fit_transform(lstm_o_reshaped)
    plt.scatter(lstm_o_tsne[:, 0], lstm_o_tsne[:, 1], c=cluster_labels, cmap='viridis')
    # plt.legend(args.num_agents)
    plt.colorbar()
    plt.savefig(plt_save_path)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.clu_mod_dir):
        os.makedirs(args.clu_mod_dir)
    # Set up logging
    logger, log_file = setup_logging(args.clu_mod_dir)
    
    # Log training configuration
    logger.info("="*50)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*50)
    logger.info(f"Device: {args.train_device}")
    logger.info(f"Batch size: {args.batchsize}")
    logger.info(f"Learning rate: 5e-4")
    logger.info(f"Epochs: {args.num_epoch}")
    logger.info(f"Beta (KL weight): {args.beta}")
    logger.info(f"Required consecutive epochs for early stopping: {args.reqd_con_eps}")
    logger.info(f"Gradient norm threshold: {args.grad_norm_threshold}")
    logger.info(f"Number of agents: {args.num_agents}")
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Async batch size: {args.async_batch_size}")
    logger.info("="*50)

    # Load your dataset
    logger.info("Loading dataset...")   
    
    # Capture and log dataset loading output
    def load_dataset_data():
        train_dataset = LoadDataset(args)
        return train_dataset.load()
    
    publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len = capture_and_log_output(logger, load_dataset_data)
    
    # Calculate actual rewards (this might also have output)
    def calc_rewards():
        return calc_actual_rewards(reward, args)
    
    reward = capture_and_log_output(logger, calc_rewards)
    train_dataset = (publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len)
    logger.info("Dataset loaded successfully")
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
    # Check if cluster information already exists
    # train_mi(args, batch_loader, train_dataset, epochs=args.num_epoch, beta=args.beta)
    cluster_save_path = os.path.join(args.clu_mod_dir, 'best_cluster.pt')
    model_save_path = os.path.join(args.clu_mod_dir, 'best_model.pt')
    # encoder, cluster_centers_tensor = load_cluster(cluster_save_path, model_save_path, args)
    plot_tsne(batch_loader, cluster_save_path, model_save_path, args)