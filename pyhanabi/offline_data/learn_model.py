import torch
import torch.distributed as dist 
import pickle
from tqdm import tqdm
import numpy as np
import tensorboardX
import os
import argparse
import time
from torch.utils.data.distributed import DistributedSampler
import os
import datetime

class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, filename, local_rank, world_size):
        self.mmap = np.load(filename, mmap_mode='r')
        self.total_samples = len(self.mmap['terminal']) - 1

        # Calculate shard indices
        self.per_worker = self.total_samples // world_size
        self.start_idx = local_rank * self.per_worker
        self.end_idx = min((local_rank + 1) * self.per_worker, self.total_samples)
        
        # Precompute valid indices
        self.valid_indices = self._find_valid_indices()
    
    def _get_dim(self):
        dims = {
            'num_plyrs': self.mmap['action'][0].shape[0],
            'priv_dim': self.mmap['priv_s_cur'][0].shape[-1],
            'publ_dim': self.mmap['publ_s_cur'][0].shape[-1],
            'legal_dim': self.mmap['legal_actions_curr'][0].shape[-1]
            }
        return dims

    def _find_valid_indices(self):
        """Skip terminal states where next state is invalid"""
        valid = []
        for idx in range(self.start_idx, self.end_idx):
            if not self.mmap['terminal'][idx]:
                valid.append(idx)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, local_idx):
        global_idx = self.valid_indices[local_idx]
        return (
            torch.from_numpy(self.mmap['publ_s_cur'][global_idx], dtype=torch.float32),
            torch.from_numpy(self.mmap['publ_s_nxt'][global_idx+1], dtype=torch.float32),
            torch.from_numpy(self.mmap['priv_s_cur'][global_idx], dtype=torch.float32),
            torch.from_numpy(self.mmap['priv_s_nxt'][global_idx+1], dtype=torch.float32),
            torch.from_numpy(self.mmap['legal_actions_curr'][global_idx], dtype=torch.float32),
            torch.from_numpy(self.mmap['legal_actions_nxt'][global_idx+1], dtype=torch.float32),
            torch.from_numpy(self.mmap['action'][global_idx], dtype=torch.float32),
            torch.from_numpy(self.mmap['reward'][global_idx], dtype=torch.float32),
            torch.from_numpy(self.mmap['terminal'][global_idx], dtype=torch.float32)
        )


def create_distributed_loader(filename, batch_size, local_rank, world_size):
    dataset = MMapDataset(filename, local_rank, world_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    ), sampler, dataset._get_dim()

class model(torch.nn.Module):
    def __init__(self, hidden_size, num_players, publ_dim, priv_dim, legal_dim):
        super(model, self).__init__()
        # self.fc1 = torch.nn.Linear(input_size, hidden_size)

        self.publ_fc1 = torch.nn.Linear(publ_dim*num_players, hidden_size)
        self.priv_fc1 = torch.nn.Linear(priv_dim*num_players, hidden_size)
        self.legal_fc1 = torch.nn.Linear(legal_dim*num_players, hidden_size)
        self.act_fc1 = torch.nn.Linear(num_players, hidden_size)

        # self.publ_c1D = torch.nn.Conv1d(2, 1, kernel_size=5, stride=1)
        # self.priv_c1D = torch.nn.Conv1d(2, 1, kernel_size=5, stride=1)
        # self.legal_act_c1D = torch.nn.Conv1d(2, 1, kernel_size=5, stride=1)
        # self.act_c1D = torch.nn.Conv1d(2, 1, kernel_size=5, stride=1)
        
        # self.fc_publ = torch.nn.Linear(publ_dim, hidden_size)
        # self.fc_priv = torch.nn.Linear(priv_dim, hidden_size)
        # self.fc_legal = torch.nn.Linear(legal_dim, hidden_size)
        # self.fc_act = torch.nn.Linear(1, hidden_size)

        self.shared_fc1 = torch.nn.Linear(hidden_size*4, hidden_size*2)
        self.shared_fc2 = torch.nn.Linear(hidden_size*2, hidden_size)

        self.out_publ = torch.nn.Linear(hidden_size, publ_dim*num_players)
        self.out_priv = torch.nn.Linear(hidden_size, priv_dim*num_players)
        self.out_legal = torch.nn.Linear(hidden_size, legal_dim*num_players)
        self.out_reward = torch.nn.Linear(hidden_size, 1)
        self.out_terminal = torch.nn.Linear(hidden_size, 1)
        
        # self.publ_sig = torch.nn.Sigmoid()
        # self.priv_sig = torch.nn.Sigmoid()
        # self.legal_sig = torch.nn.Sigmoid()
        # self.terminal_sig = torch.nn.Sigmoid()

        self.log_publ_lambda = torch.nn.Parameter(torch.zeros(1))
        self.log_priv_lambda = torch.nn.Parameter(torch.zeros(1))
        self.log_legal_lambda = torch.nn.Parameter(torch.zeros(1))
        self.log_reward_lambda = torch.nn.Parameter(torch.zeros(1))
        self.log_terminal_lambda = torch.nn.Parameter(torch.zeros(1))

    
    def forward(self, publ_s_cur, priv_s_cur, legal_actions_curr, actions):
        B, N, _ = publ_s_cur.shape
        publ_s_cur = publ_s_cur.view(B, -1)
        priv_s_cur = priv_s_cur.view(B, -1)
        legal_actions_curr = legal_actions_curr.view(B, -1)
        actions = actions.view(B, -1)
        
        publ_s_cur = torch.relu(self.publ_fc1(publ_s_cur))
        priv_s_cur = torch.relu(self.priv_fc1(priv_s_cur))
        legal_actions_curr = torch.relu(self.legal_fc1(legal_actions_curr))
        actions = torch.relu(self.act_fc1(actions))

        # publ_s_cur = self.publ_c1D(publ_s_cur)
        # priv_s_cur = self.priv_c1D(priv_s_cur)
        # legal_actions_curr = self.legal_act_c1D(legal_actions_curr)
        # actions = self.act_c1D(actions)

        # publ_s_cur = publ_s_cur.squeeze(dim=1)
        # priv_s_cur = priv_s_cur.squeeze(dim=1)
        # legal_actions_curr = legal_actions_curr.squeeze(dim=1)
        # actions = actions.squeeze(dim=1)

        # publ_s_cur = torch.relu(self.fc_publ(publ_s_cur))
        # priv_s_cur = torch.relu(self.fc_priv(priv_s_cur))
        # legal_actions_curr = torch.relu(self.fc_legal(legal_actions_curr))
        # actions = torch.relu(self.fc_act(actions))

        x = torch.cat([publ_s_cur, priv_s_cur, legal_actions_curr, actions], dim=1)

        x = torch.relu(self.shared_fc1(x))
        x = torch.relu(self.shared_fc2(x))

        publ_s_nxt = self.out_publ(x).view(B, N, -1)
        priv_s_nxt = self.out_priv(x).view(B, N, -1)
        legal_actions_nxt = self.out_legal(x).view(B, N, -1)
        reward = self.out_reward(x).view(B, 1)
        terminal = self.out_terminal(x).view(B, 1)
        
        # publ_s_nxt = self.publ_sig(publ_s_nxt)
        # priv_s_nxt = self.priv_sig(priv_s_nxt)
        # legal_actions_nxt = self.legal_sig(legal_actions_nxt)
        # terminal = self.terminal_sig(terminal)
        return publ_s_nxt, priv_s_nxt, legal_actions_nxt, reward, terminal

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def train(model, train_loader, train_sampler, args):
    model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    pubs_mse_loss = torch.nn.MSELoss()
    privs_mse_loss = torch.nn.MSELoss()
    legal_bce_loss = torch.nn.BCEWithLogitsLoss()
    reward_mse_loss = torch.nn.MSELoss()
    terminal_bce_loss = torch.nn.BCEWithLogitsLoss()
    
    # Mixed precision setup
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}', disable=local_rank != 0, bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        avg_loss = 0
        start_time = time.time()
        for batch_idx, batch in enumerate(epoch_bar):
            (publ_s_cur, publ_s_nxt, priv_s_cur, priv_s_nxt, legal_actions_curr, legal_actions_nxt, actions, rewards, terminals) = (t.to(device, non_blocking=True) for t in batch)
 
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                publ_s_nxt_pred, priv_s_nxt_pred, legal_actions_nxt_pred, reward_pred, terminal_pred = model(publ_s_cur, priv_s_cur, legal_actions_curr, actions)
            
                publ_loss = (torch.exp(-model.module.log_publ_lambda))*pubs_mse_loss(publ_s_nxt_pred, publ_s_nxt)
                priv_loss = (torch.exp(-model.module.log_priv_lambda))*privs_mse_loss(priv_s_nxt_pred, priv_s_nxt)
                legal_loss = (torch.exp(-model.module.log_legal_lambda))*legal_bce_loss(legal_actions_nxt_pred, legal_actions_nxt)
                reward_loss = (torch.exp(-model.module.log_reward_lambda))*reward_mse_loss(reward_pred, rewards)
                terminal_loss = (torch.exp(-model.module.log_terminal_lambda))*terminal_bce_loss(terminal_pred, terminals)
                
                loss = publ_loss + priv_loss + legal_loss + reward_loss + terminal_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            #sync losses across all GPUs
            loss_tensor = loss.detach().clone()
            dist.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_loss = (loss_tensor.item() / dist.get_world_size() + avg_loss * batch_idx) / (batch_idx + 1)

            if local_rank == 0:
                epoch_bar.set_postfix(
                    avg_loss=f'{avg_loss:.5f}',
                    batch_rate=f'{(batch_idx+1)/(time.time()-start_time):.2f}'
                )

        
        if epoch % 10 == 0 and local_rank == 0:
            model_dir = args.model_dir + args.run_name
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = model_dir +'/model_'+str(epoch)+'.pth'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

        if writer is not None and local_rank == 0:
            writer.add_scalar('Avg_loss', avg_loss, epoch)
            writer.add_scalar('publ_loss', publ_loss.item(), epoch)
            writer.add_scalar('priv_loss', priv_loss.item(), epoch)
            writer.add_scalar('legal_loss', legal_loss.item(), epoch)
            writer.add_scalar('reward_loss', reward_loss.item(), epoch)
            writer.add_scalar('terminal_loss', terminal_loss.item(), epoch)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="offline_model", help="Run name, default: OFL-MOD")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0, 1, 2], help='List of GPU device IDs')
    parser.add_argument('--log_dir', type=str, default='/data/kmirakho/offline_data/logs/')
    parser.add_argument('--model_dir', type=str, default='/data/kmirakho/offline_data/model/')
    parser.add_argument('--load_model', type=bool, default=False)
    parser.add_argument('--dataset_path', type=str, default='/data/kmirakho/offline_data/data_*.pickle')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_config()
    # Set NCCL environment variables
    os.environ['NCCL_ALGO'] = 'Ring'
    os.environ['NCCL_NSOCKS_PERTHREAD'] = '4'
    os.environ['NCCL_SOCKET_NTHREADS'] = '2'
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])

    # filenames = glob(args.dataset_path)[1:]
    filename = '/data/kmirakho/offline_data/dataset_1040640.npz'

    # Create distributed loader
    train_loader, train_sampler, dims = create_distributed_loader(filename, args.batch_size, local_rank, world_size)

    # Initialize process group first
    dist.init_process_group(
        backend='nccl',  # Use 'gloo' for CPU
        init_method='env://',
        world_size=world_size,
        rank=local_rank,
        timeout=datetime.timedelta(seconds=30)
        )
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    log_dir = args.log_dir + args.run_name
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    writer = tensorboardX.SummaryWriter(log_dir)

    num_players = dims['num_plyrs']
    publ_dim = dims['publ_dim']
    priv_dim = dims['priv_dim']
    legal_dim = dims['legal_dim']
    
    train_model = model(args.hidden_size, num_players, publ_dim, priv_dim, legal_dim)

    train(train_model, train_loader, train_sampler, args)

    # torchrun --nproc-per-node=2 learn_model.py --run_name="model_1040640" --batch_size=256 --num_epochs=100 --hidden_size=1024 --lr=1e-4

    





    


