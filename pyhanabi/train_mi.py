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


def train_mi(args, batch_loader, epochs, beta):
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

    best_loss = float('inf')

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

            # grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)

            optimizer.step()

            epoch_bar.set_postfix(
                loss=avg_loss/(batch_idx + 1), 
                action_loss=avg_action_loss/(batch_idx + 1),
                # reward_loss=reward_recon_loss.item(),
                # state_loss=state_recon_loss.item(),
                kl_loss=avg_kl_loss/(batch_idx + 1), 
                # grad_norm=grad_norm.item(),
                lr=scheduler.get_last_lr()[0]
                # mi_loss=mi_loss.item()
            )
        avg_loss = avg_loss / (batch_idx + 1)
        avg_action_loss = avg_action_loss / (batch_idx + 1)
        avg_kl_loss = avg_kl_loss / (batch_idx + 1)
        scheduler.step()

        #add the average loss & learning rate to the tensorboard
        summary_writer.add_scalar('loss', avg_loss, epoch)
        summary_writer.add_scalar('action_loss', avg_action_loss, epoch)
        summary_writer.add_scalar('kl_loss', avg_kl_loss, epoch)
        summary_writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_encoder = encoder
            best_decoder = decoder
        if epoch % 100 == 0:
            save_cluster_path = os.path.join(args.clu_mod_dir, f'best_cluster.pt')
            cluster_centers_tensor = save_cluster(save_cluster_path, best_encoder, batch_loader, args)
            #save the encoder and decoder
            torch.save({"encoder": best_encoder.state_dict(), "decoder": best_decoder.state_dict()}, os.path.join(args.clu_mod_dir, f'best_model.pt'))   
    return encoder, cluster_centers_tensor


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

def plot_tsne(batch_loader, cluster_save_path, model_save_path, args):
    #extract all the folder names except the last one
    folder_names = args.save_dir.split('/')[:-1]
    folder_names = '/'.join(folder_names)
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
    kmeans = KMeans(n_clusters=args.num_agents, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(lstm_o_reshaped)
    
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
    # Load your dataset
    train_dataset = LoadDataset(args)
    publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len = train_dataset.load()
    reward = calc_actual_rewards(reward, args)
    train_dataset = (publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len)
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
    # train_mi(args, batch_loader, 1000, 0.0)
    cluster_save_path = os.path.join(args.clu_mod_dir, 'best_cluster.pt')
    model_save_path = os.path.join(args.clu_mod_dir, 'best_model.pt')
    # encoder, cluster_centers_tensor = load_cluster(cluster_save_path, model_save_path, args)
    plot_tsne(batch_loader, cluster_save_path, model_save_path, args)