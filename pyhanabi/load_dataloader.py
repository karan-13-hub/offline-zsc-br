import numpy as np
import random
import torch
import pickle
from collections import deque, namedtuple
import threading
import time
from tqdm import tqdm
import argparse

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
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)

    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )

    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )

    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--num_eval_after", type=int, default=100)
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="/data/kmirakho/offline-model/dataset_rl_1040640_sml.npz")


    args = parser.parse_args()
    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]
    return args

class PrioritizedReplayBuffer:
    def __init__(self, args):
        """Initialize prioritized replay buffer with enhanced features.
        
        Args:
            priority_exponent (float): α parameter controlling prioritization strength
            priority_weight (float): β parameter for importance sampling correction
            prefetch (int): Number of batches to prefetch in background
        """
        self.device = args.train_device
        self.memory = {}
        self.priorities = []
        self.buffer_size = args.replay_buffer_size
        self.batch_size = args.batchsize
        self.num_agents = args.num_player
        self.alpha = args.priority_exponent
        self.beta = args.priority_weight
        self.prefetch = args.prefetch
        self.hid_dim = args.rnn_hid_dim
        self.num_lstm_layer = args.num_lstm_layer
        self.eps = 1e-5
        self.max_priority = 100
        self.multi_step = args.multi_step
        self.max_len = args.max_len
        self.data = None
        self.indices = None
       
        self.experience = namedtuple("Experience", field_names=["publ_s", "priv_s", "legal_move", "h0", "action", "reward", "bootstrap", "terminal", "seq_len"])

        # Prefetch queue for asynchronous loading
        self.prefetch_queue = deque(maxlen=self.prefetch)
        self.prefetch_thread = None

    def _prefetch_batches(self):
        """Background thread for prefetching batches"""
        while len(self.prefetch_queue) < self.prefetch:
            batch = self._sample_batch()
            self.prefetch_queue.append(batch)

    def sample(self):
        """Get a batch with priority-based sampling and importance weights"""
        if self.prefetch > 0:
            if self.prefetch_thread is None or not self.prefetch_thread.is_alive():
                self.prefetch_thread = threading.Thread(target=self._prefetch_batches, daemon=True)
                self.prefetch_thread.start()
                print("Prefetching batches in background...")
            
            start_time = time.time()
            print("len of q size before: ", len(self.prefetch_queue))
            while not self.prefetch_queue:
                time.sleep(0.001)

            print("Prefetching batches..., ", time.time()-start_time)
            print("len of q size after: ", len(self.prefetch_queue))
            

            return self.prefetch_queue.popleft()
            # return None
        else:
            print("No prefetching enabled.")
            return self._sample_batch()

    def _sample_batch(self):
        print("sampling")
        """Core sampling logic with priority exponents and weights"""
        priorities = np.array(self.memory["priorities"]) + self.eps
        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory["priorities"]), size=self.batch_size, p=probs)

        # Importance sampling weights
        weights = (len(self.memory["priorities"]) * probs[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights

        # Convert weights to tensor on the correct device
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        start_time = time.time()

        sample = namedtuple("batch", field_names=["obs", "h0", "action", "reward", "bootstrap", "terminal", "seq_len"])

        # batch_data = [self.memory[idx] for idx in indices]

        # Stack tensors efficiently using zip(*batch_data)
        
        
        # publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len = zip(*batch_data)

        # print("time here 1:  ", time.time()-start_time)
        
        # breakpoint()
        ## TODO: problem is here
        ## TODO: consider using dataloader for creating the objects
        # sample.obs = {
        #     "publ_s": torch.stack(publ_s).transpose(1, 0).to(self.device),
        #     "priv_s": torch.stack(priv_s).transpose(1, 0).to(self.device),
        #     "legal_move": torch.stack(legal_move).transpose(1, 0).to(self.device),
        # }
        # sample.obs = {
        #     "publ_s": torch.stack(publ_s),
        #     "priv_s": torch.stack(priv_s),
        #     "legal_move": torch.stack(legal_move),
        # }

        sample.obs = {
            "publ_s": self.memory["publ_s"][indices].transpose(1, 0).to(self.device),
            "priv_s": self.memory["priv_s"][indices].transpose(1, 0).to(self.device),
            "legal_move": self.memory["legal_move"][indices].transpose(1, 0).to(self.device),
        }
        
        

        # publ_s = torch.cat([x[0].unsqueeze(0) for x in batch_data], dim=0)
        # priv_s = torch.cat([x[1].unsqueeze(0) for x in batch_data], dim=0)

        # print("time here 2 : ", time.time()-start_time)
        
        # sample.h0 = {
        #     "h0": torch.stack([h["h0"] for h in h0]).transpose(1, 0).to(self.device),
        #     "c0": torch.stack([h["c0"] for h in h0]).transpose(1, 0).to(self.device),
        # }
        # print("time here 3: ", time.time()-start_time)
        # sample.action = {
        #     "a" : torch.stack(action).transpose(1, 0).to(self.device)
        # }
        # breakpoint()
        # sample.obs = {
        #     "publ_s": self.memory["publ_s"].transpose(1, 0).to(self.device),

        # }
        # sample.reward = torch.stack(reward).transpose(1, 0).to(self.device)
        # sample.bootstrap = torch.stack(bootstrap).transpose(1, 0).to(self.device)
        # sample.terminal = torch.stack(terminal).transpose(1, 0).to(self.device)
        # sample.seq_len = torch.stack(seq_len).to(self.device)
        self.indices = indices

        

        return sample, weights   

    def update_priorities(self, td_errors):
        # """Update priorities using TD errors and priority exponent"""
        # td_errors = td_errors.detach().cpu().numpy()
        # for idx, error in zip(self.indices, td_errors):
        #     new_priority = np.max(np.abs(error)) + self.eps
        #     self.priorities[idx] = new_priority
        #     self.max_priority = max(self.max_priority, new_priority)
        """Update priorities using TD errors."""
        td_errors_np = td_errors.detach().cpu().numpy()
        
        for idx, error in zip(self.indices, td_errors_np):
            new_priority = np.abs(error).max() + self.eps  # Use max absolute error for stability
            self.priorities[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

    def load(self, filename):
        
        # Load entire dataset to memory first
        with np.load(filename) as data:
            self.data = {k: data[k] for k in data.files}
        data.close()

        print("data loaded!!")

        
        data_len = len(self.data['publ_s'])
        shape = (data_len, self.num_lstm_layer, self.num_agents, self.hid_dim)
        self.memory["publ_s"] = torch.from_numpy(self.data['publ_s']).to(torch.float32).to(self.device)
        self.memory["priv_s"] = torch.from_numpy(self.data['priv_s']).to(torch.float32).to(self.device)
        self.memory["legal_move"] = torch.from_numpy(self.data['legal_move']).to(torch.float32).to(self.device)
        self.memory["action"] = torch.from_numpy(self.data['action']).to(torch.float32).to(self.device)
        self.memory["reward"] = torch.from_numpy(self.data['reward']).to(torch.float32).to(self.device)
        self.memory["terminal"] = torch.from_numpy(self.data['terminal']).to(torch.float32).to(self.device)
        self.memory["h0"] = torch.zeros(*shape, dtype=torch.float32).to(self.device)
        self.memory["c0"] = torch.zeros(*shape, dtype=torch.float32).to(self.device)
        # self.memory["priorities"] = torch.ones(data_len, dtype=torch.float32).to(self.device)*self.max_priority
        self.memory["priorities"] = np.ones(data_len, dtype=np.float32)*self.max_priority
        
        # TODO: fix these
        self.memory["seq_len"] = torch.from_numpy(self.data['terminal']).to(torch.float32).to(self.device)
        self.memory["bootstrap"] = torch.zeros((data_len, 1), dtype=torch.bool).to(self.device)

        
        # for i in tqdm(range(len(self.data['terminal']))):
        #     publ_s = torch.from_numpy(self.data['publ_s'][i]).to(torch.float32)
        #     priv_s = torch.from_numpy(self.data['priv_s'][i]).to(torch.float32)
        #     legal_move = torch.from_numpy(self.data['legal_move'][i]).to(torch.bool)
        #     action = torch.from_numpy(self.data['action'][i]).to(torch.int64)
        #     reward = torch.from_numpy(self.data['reward'][i]).to(torch.float32)
        #     terminal = torch.from_numpy(self.data['terminal'][i]).to(torch.bool)
        #     bootstrap = torch.zeros(len(terminal), dtype=torch.bool)
        #     idx = torch.where(terminal==1)[0][0]
        #     if idx - self.multi_step >= 0:
        #         bootstrap[: idx-self.multi_step+1] = 1
        #     seq_len = torch.sum(terminal == 0)+1
        #     e = self.experience(publ_s, priv_s, legal_move, hid, action, reward, bootstrap, terminal, seq_len)
        #     self.memory.append(e)
        #     self.priorities.append(self.max_priority)

        return
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory["priorities"])


if __name__ == "__main__":
    args = parse_args()
    print("Loading dataset...")
    replay_buffer = PrioritizedReplayBuffer(args)
    replay_buffer.load(args.dataset_path)
    print("Dataset loaded.")

    import time
    for i in tqdm(range(100)):
        print("device is: ", replay_buffer.device)
        start_time = time.time()
        # data = replay_buffer.sample()
        # time.sleep(0.1)
        replay_buffer._sample_batch()  
        print("i can see this")
        print("time taken: ", time.time()-start_time)
        breakpoint()

