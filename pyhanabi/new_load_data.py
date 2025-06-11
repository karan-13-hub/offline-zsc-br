import numpy as np
import random
import torch
import pickle
from collections import namedtuple
import queue
import threading
import time
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor
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
        self.memory = []
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
        self.prefetch_queue = queue.Queue(maxsize=self.prefetch)
        self.prefetch_thread = ThreadPoolExecutor(max_workers=args.num_data_thread)
        self.shutdown_flag = threading.Event()

        # Separate locks for priority access and memory access
        self.priority_lock = threading.RLock()
        self.memory_lock = threading.RLock()

        # Start worker threads to prefill the queue
        # self._initialize_workers()

    def _initialize_workers(self):
        """Start the worker threads to fill the batch queue"""
        for _ in range(self.prefetch_thread._max_workers):
            self.prefetch_thread.submit(self._prefetch_batches)

    def _prefetch_batches(self):
        """Background thread for prefetching batches"""
        while not self.shutdown_flag.is_set():
            try:
                # Only generate batches if there's space in the queue
                if self.prefetch_queue.qsize() < self.prefetch:
                    batch = self._sample_batch()
                    self.prefetch_queue.put(batch, timeout=0.01)
                else:
                    time.sleep(0.01)
            except queue.Full:
                # If the queue is full, wait for space to open up
                continue
            except Exception as e:
                print(f"Error in prefetching: {e}")
                time.sleep(0.1)

    def _sample_batch(self):
        with self.priority_lock:
            """Core sampling logic with priority exponents and weights"""
            priorities = np.array(self.priorities) + self.eps
            probs = priorities ** self.alpha
            probs /= probs.sum()

            indices = np.random.choice(len(self.memory), size=self.batch_size, p=probs)

            # Importance sampling weights
            weights = (len(self.memory) * probs[indices]) ** -self.beta
            weights /= weights.max()  # Normalize weights

            # Convert weights to tensor on the correct device
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            
            self.indices = indices

        with self.memory_lock:
            # Sample experiences from memory
            batch_data = [self.memory[idx] for idx in indices]

        sample = namedtuple("batch", field_names=["obs", "h0", "action", "reward", "bootstrap", "terminal", "seq_len"])

        # Stack tensors efficiently using zip(*batch_data)
        publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len = zip(*batch_data)

        sample.obs = {
            "publ_s": torch.stack(publ_s).transpose(1, 0).to(self.device, non_blocking=True),
            "priv_s": torch.stack(priv_s).transpose(1, 0).to(self.device, non_blocking=True),
            "legal_move": torch.stack(legal_move).transpose(1, 0).to(self.device, non_blocking=True),
        }
        sample.h0 = {
            "h0": torch.stack([h["h0"] for h in h0]).transpose(1, 0).to(self.device, non_blocking=True),
            "c0": torch.stack([h["c0"] for h in h0]).transpose(1, 0).to(self.device, non_blocking=True),
        }
        sample.action = {
            "a" : torch.stack(action).transpose(1, 0).to(self.device, non_blocking=True)
        }
        sample.reward = torch.stack(reward).transpose(1, 0).to(self.device, non_blocking=True)
        sample.bootstrap = torch.stack(bootstrap).transpose(1, 0).to(self.device, non_blocking=True)
        sample.terminal = torch.stack(terminal).transpose(1, 0).to(self.device, non_blocking=True)
        sample.seq_len = torch.stack(seq_len).to(self.device, non_blocking=True)
        return sample, weights   

    def sample(self):
        """Get a batch with priority-based sampling and importance weights"""
        try:
            # print("Sampling from prefetch queue")
            return self.prefetch_queue.get(timeout=0.01)
        except queue.Empty:
            # If the queue is empty, wait for a batch to be prefetched
            # print("Prefetch queue is empty, sampling directly")
            return self._sample_batch()

    def update_priorities(self, td_errors):
        """Update priorities using TD errors and priority exponent"""
        td_errors = td_errors.detach().cpu().numpy()
        for idx, error in zip(self.indices, td_errors):
            new_priority = np.max(np.abs(error)) + self.eps
            self.priorities[idx] = new_priority
            self.max_priority = max(self.max_priority, new_priority)

    def load(self, filename):
        shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
        hid = {"h0": torch.zeros(*shape, dtype=torch.float32).to(self.device), "c0": torch.zeros(*shape, dtype=torch.float32).to(self.device)}
        
        # Load entire dataset to memory first
        with np.load(filename) as data:
            self.data = {k: data[k] for k in data.files}
        data.close()
        
        for i in tqdm(range(len(self.data['terminal']))):
            publ_s = torch.from_numpy(self.data['publ_s'][i]).to(torch.float32)
            priv_s = torch.from_numpy(self.data['priv_s'][i]).to(torch.float32)
            legal_move = torch.from_numpy(self.data['legal_move'][i]).to(torch.bool)
            action = torch.from_numpy(self.data['action'][i]).to(torch.int64)
            reward = torch.from_numpy(self.data['reward'][i]).to(torch.float32)
            terminal = torch.from_numpy(self.data['terminal'][i]).to(torch.bool)
            bootstrap = torch.zeros(len(terminal), dtype=torch.bool)
            idx = torch.where(terminal==1)[0][0]
            if idx - self.multi_step >= 0:
                bootstrap[: idx-self.multi_step+1] = 1
            seq_len = torch.sum(terminal == 0)+1
            e = self.experience(publ_s, priv_s, legal_move, hid, action, reward, bootstrap, terminal, seq_len)
            self.memory.append(e)
            self.priorities.append(self.max_priority)
        
        self._initialize_workers()
        time.sleep(0.1)
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def shutdown(self):
        """Clean shutdown of thread pool"""
        self.shutdown_flag.set()
        self.prefetch_thread.shutdown(wait=True)

if __name__ == "__main__":
    args = parse_args()
    print("Loading dataset...")
    # replay_buffer = PrioritizedReplayBuffer(args)
    # replay_buffer.load(args.dataset_path)
    print("Dataset loaded.")

    # for i in tqdm(range(10000000)):
    #     sample, weights = replay_buffer.sample()  
    #     # print(sample.obs['publ_s'].shape)
    #     del sample, weights
    #     torch.cuda.empty_cache()
    #     gc.collect()

    start_time = time.time()
    data = np.load('/data/kmirakho/offline-rl-data/data_10/64.npz')
    print("Data loaded in: ", time.time() - start_time)

    data = {k: data[k] for k in data.files}
    with open('data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Data saved to data.pkl")

    # Load the data back from the pickle file
    start_time = time.time()
    with open('data.pkl', 'rb') as f:
        loaded_data = pickle.load(f)
    print("Data loaded from pickle in: ", time.time() - start_time)