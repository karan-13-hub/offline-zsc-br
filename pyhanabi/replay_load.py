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
from glob import glob
import os
import multiprocessing as mp
from functools import partial

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

    parser.add_argument("--visit_weight", type=float, default=0.5, help="visit weight")

    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)
    parser.add_argument("--num_data_thread", type=int, default=4)

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

def process_npz_file(filename, multi_step):
    try:
        # Load NPZ data
        with np.load(filename) as data_file:
            data = {k: data_file[k].copy() for k in data_file.files}        
        # Calculate bootstrap values
        bootstrap = np.zeros(len(data['terminal']), dtype=bool)
        terminal_idx = np.where(data['terminal']==1)[0]
        if len(terminal_idx) > 0:
            idx = terminal_idx[0]
            if idx - multi_step >= 0:
                bootstrap[: idx-multi_step+1] = 1
        
        seq_len = np.sum(data['terminal'] == 0) + 1
        
        # Process data (all as numpy arrays for multiprocessing compatibility)
        return {
            'publ_s': data['publ_s'],
            'priv_s': data['priv_s'],
            'legal_move': data['legal_move'],
            'action': data['action'],
            'reward': data['reward'],
            'bootstrap': bootstrap,
            'terminal': data['terminal'],
            'seq_len': seq_len
        }
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

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
        self.visits = []
        self.buffer_size = args.replay_buffer_size
        self.batch_size = args.batchsize
        self.num_agents = args.num_player
        self.alpha = args.priority_exponent
        self.beta = args.priority_weight
        self.visit_weight = args.visit_weight
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
            visits = np.array(self.visits) + 1

            priorities = priorities / (visits ** self.visit_weight)
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
            self.visits[idx] += 1
    
    def load(self, data_path):        
        # Prepare the hidden state once (reused for all experiences)
        shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
        hid = {"h0": torch.zeros(*shape, dtype=torch.float32).to(self.device), 
            "c0": torch.zeros(*shape, dtype=torch.float32).to(self.device)}
        
        # Get all npz files from the folders
        folders = os.listdir(data_path)
        filenames = []
        for folder in folders:
            filenames.extend(glob(os.path.join(data_path, folder, '*.npz')))
        
        # import pdb; pdb.set_trace()
        # Set up multiprocessing pool
        filenames = filenames[:10000]
        num_workers = max(1, mp.cpu_count() - 1)  # Use all CPUs except one

        worker_func = partial(
            process_npz_file,
            multi_step=self.multi_step
        )
        
        # Process files in parallel
        # Use multiprocessing to process chunks in parallel
        print(f"Loading dataset using {num_workers} processes...")
        try:
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, filenames),
                    total=len(filenames),
                    desc="Loading files in parallel"
                ))
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        finally:
            gc.collect()
        
        # Process results in the main thread (where CUDA is available)
        valid_results = [r for r in results if r is not None]
        for result in tqdm(valid_results, desc="Converting to tensors"):
            # Convert numpy arrays to torch tensors and move to device
            publ_s = torch.from_numpy(result['publ_s']).to(torch.float32, non_blocking=True)
            priv_s = torch.from_numpy(result['priv_s']).to(torch.float32, non_blocking=True)
            legal_move = torch.from_numpy(result['legal_move']).to(torch.bool, non_blocking=True)
            action = torch.from_numpy(result['action']).to(torch.int64, non_blocking=True)
            reward = torch.from_numpy(result['reward']).to(torch.float32, non_blocking=True)
            terminal = torch.from_numpy(result['terminal']).to(torch.bool, non_blocking=True)
            bootstrap = torch.from_numpy(result['bootstrap']).to(torch.bool, non_blocking=True)
            seq_len = torch.tensor(result['seq_len'], dtype=torch.int32)
            
            # Create experience and add to memory
            e = self.experience(publ_s, priv_s, legal_move, hid, action, reward, bootstrap, terminal, seq_len)
            self.memory.append(e)
            self.priorities.append(self.max_priority)
            self.visits.append(0)
        time.sleep(0.1)
        self._initialize_workers()
        print("Initialized workers for prefetching...")
        time.sleep(0.1)

            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def get_visits(self):
        return self.visits
    
    def shutdown(self):
        """Clean shutdown of thread pool"""
        self.shutdown_flag.set()
        self.prefetch_thread.shutdown(wait=True)
        

class LoadDataset:
    def __init__(self, args):
        self.args = args
        self.num_lstm_layer = args.num_lstm_layer
        self.num_agents = args.num_player
        self.hid_dim = args.rnn_hid_dim
        self.multi_step = args.multi_step
        self.data_path = args.dataset_path
        self.num_workers = args.num_data_thread
        self.publ_s = []
        self.priv_s = []
        self.legal_move = []
        self.h0 = []
        self.action = []
        self.reward = []
        self.bootstrap = []
        self.terminal = []
        self.seq_len = []

    def _process_reward_sequence(self, reward, multi_step=3, gamma=0.999, new_multi_step=3):
        """
        Process each reward in the sequence one by one. This can be used for multi-step reward adjustment or other per-step processing.
        """
        discount_factors = np.array([1.0 / (gamma ** i) for i in range(1, multi_step)])
        reward_array = reward.numpy()
        reward_length = len(reward_array)
        adjusted_rewards = np.zeros_like(reward_array)

        # Combine both loops
        for i in range(reward_length):
            if i <= reward_length - multi_step:
                pos = reward_length - multi_step - i
                future_rewards = adjusted_rewards[pos+1:pos+multi_step]
                factors = discount_factors[:len(future_rewards)]
                discounted_sum = np.sum(future_rewards * factors)
                adjusted_rewards[pos] = reward_array[pos] - discounted_sum
                adjusted_rewards[pos] = round(adjusted_rewards[pos], 1)
            else:
                pos = i
                future_rewards = adjusted_rewards[pos+1:reward_length]
                factors = discount_factors[:len(future_rewards)]
                discounted_sum = np.sum(future_rewards * factors)
                adjusted_rewards[pos] = reward_array[pos] - discounted_sum

        # Adjust the rewards for the new multi step
        discount_factors = np.array([gamma ** i for i in range(1, new_multi_step)])
        for i in range(reward_length):
            pos = i
            if i <= reward_length - new_multi_step:
                future_rewards = adjusted_rewards[pos+1:pos+new_multi_step]
                discounted_sum = np.sum(future_rewards * discount_factors[:len(future_rewards)])
                adjusted_rewards[pos] = adjusted_rewards[pos] + discounted_sum
            else:
                pos = i
                future_rewards = adjusted_rewards[pos+1:reward_length]
                discounted_sum = np.sum(future_rewards * discount_factors[:len(future_rewards)])
                adjusted_rewards[pos] = adjusted_rewards[pos] + discounted_sum

        adjusted_rewards = torch.from_numpy(adjusted_rewards).to(torch.float32, non_blocking=True)
        return adjusted_rewards

    def load(self):     
        # Prepare the hidden state once (reused for all experiences)
        shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
        hid = {"h0": torch.zeros(*shape, dtype=torch.float32), 
            "c0": torch.zeros(*shape, dtype=torch.float32)}
        
        # Get all npz files from the folders
        # folders = os.listdir(self.data_path)

        #Expert replay dataset
        # folders = ['/data/kmirakho/vdn-offline-data/data_640', '/data/kmirakho/vdn-offline-data/data_1280']

        #Medium replay dataset
        # folders = ['/data/kmirakho/vdn-offline-data/data_20', '/data/kmirakho/vdn-offline-data/data_40', '/data/kmirakho/vdn-offline-data/data_80']

        #Small replay dataset
        # folders = ['/data/kmirakho/vdn-offline-data-seed1234/data_vdn_80']
        # folders = ['/data/kmirakho/vdn-offline-data/data_20', '/data/kmirakho/vdn-offline-data/data_40', '/data/kmirakho/vdn-offline-data/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed9/data_80', '/data/kmirakho/vdn-offline-data-seed1234/data_80']#, '/data/kmirakho/vdn-offline-data-seed-42/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed-42/data_80', '/data/kmirakho/vdn-offline-data-seed-1e9+7/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed-777/data_20','/data/kmirakho/vdn-offline-data-seed-777/data_40','/data/kmirakho/vdn-offline-data-seed-777/data_80']
        folders = ['/data/kmirakho/vdn-offline-data-seed-777/data_80', '/data/kmirakho/vdn-offline-data-seed-31337/data_80', '/data/kmirakho/vdn-offline-data-seed-1e9+7/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed-31337/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed9/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed-42/data_80']
        # folders = ['/data/kmirakho/vdn-offline-data-seed-1e9+7/data_80'] 


        filenames = []
        for folder in folders:
            print(f"Loading dataset from {folder}")
            #randomly select 50% of the files from each folder
            files = glob(os.path.join(self.data_path, folder, '*.npz'))
            files = random.sample(files, int(len(files) * 0.5))
            filenames.extend(files)
            # filenames.extend(glob(os.path.join(self.data_path, folder, '*.npz')))
        filenames = filenames[:1000]
        # #load the filenames from the split1.txt file
        # print(f"Loading dataset from split1.txt")
        # with open('/data/kmirakho/vdn-offline-data/split_medium_replay_in_2/split1.txt', 'r') as f:
        #     #remove the \n from the filenames
        #     filenames = [line.strip() for line in f.readlines()]    
        # f.close()

        # #load the filenames from the split2.txt file
        # print(f"Loading dataset from split2.txt")
        # with open('/data/kmirakho/vdn-offline-data/split_medium_replay_in_2/split2.txt', 'r') as f:
        #     #remove the \n from the filenames
        #     filenames = [line.strip() for line in f.readlines()]    
        # f.close()

        # load the filenames from the expert_replay_10k split.txt file
        # with open('/data/kmirakho/vdn-offline-data/expert_replay_10k/split.txt', 'r') as f:
        #     filenames = [line.strip() for line in f.readlines()]
        # f.close()
        # print(f"Loaded from expert_replay_10k split.txt")

        # # load the filenames from the data_40_10k split.txt file
        # with open('/data/kmirakho/vdn-offline-data/data_40_10k/split.txt', 'r') as f:
        #     filenames = [line.strip() for line in f.readlines()]
        # f.close()
        # print(f"Loaded from data_40_10k split.txt")

        # random.shuffle(filenames)
        # filenames = filenames[:10000]
        # with open('/data/kmirakho/vdn-offline-data/expert_replay_10k/split.txt', 'w') as f:
        #     for filename in filenames:
        #         f.write(filename + '\n')
        # f.close()

        #split the filenames into 2 halves and save them as 2 different files
        # random.shuffle(filenames)
        # split1 = filenames[:len(filenames)//2]
        # split2 = filenames[len(filenames)//2:]
        # with open('/data/kmirakho/vdn-offline-data/split_medium_replay_in_2/split1.txt', 'w') as f:
        #     for filename in split1:
        #         f.write(filename + '\n')
        # f.close()

        # with open('/data/kmirakho/vdn-offline-data/split_medium_replay_in_2/split2.txt', 'w') as f:
        #     for filename in split2:
        #         f.write(filename + '\n')
        # f.close()

        # Set up multiprocessing pool
        num_workers = max(1, mp.cpu_count() - 1)  # Use all CPUs except one

        worker_func = partial(
            process_npz_file,
            multi_step=self.multi_step
        )
        
        # Process files in parallel
        # Use multiprocessing to process chunks in parallel
        print(f"Loading dataset using {num_workers} processes...")
        try:
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, filenames),
                    total=len(filenames),
                    desc="Loading files in parallel"
                ))
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        finally:
            gc.collect()
        
        # Process results in the main thread (where CUDA is available)
        valid_results = [r for r in results if r is not None]
        for result in tqdm(valid_results, desc="Converting to tensors"):
            # Convert numpy arrays to torch tensors and move to device
            publ_s = torch.from_numpy(result['publ_s']).to(torch.float32, non_blocking=True)
            priv_s = torch.from_numpy(result['priv_s']).to(torch.float32, non_blocking=True)
            legal_move = torch.from_numpy(result['legal_move']).to(torch.bool, non_blocking=True)
            action = torch.from_numpy(result['action']).to(torch.int64, non_blocking=True)
            terminal = torch.from_numpy(result['terminal']).to(torch.bool, non_blocking=True)
            bootstrap = torch.from_numpy(result['bootstrap']).to(torch.bool, non_blocking=True)
            # change the bootstrap according to the multi step
            # bootstrap is a boolean tensor of shape (seq_len, num_agents)
            # basically bootstrap is terminal (for seq len) - multi_step
            # import pdb; pdb.set_trace()
            bootstrap = ~terminal
            seq_len = torch.tensor(result['seq_len'], dtype=torch.int32)
            bootstrap[seq_len - self.multi_step:seq_len] = False
            reward = torch.from_numpy(result['reward']).to(torch.float32, non_blocking=True)

            # Process the reward sequence one by one
            reward = self._process_reward_sequence(reward, gamma=self.args.gamma, new_multi_step=self.multi_step)

            # Create experience and add to memory
            self.publ_s.append(publ_s)
            self.priv_s.append(priv_s)
            self.legal_move.append(legal_move)
            self.h0.append(hid)
            self.action.append(action)
            self.reward.append(reward)
            self.bootstrap.append(bootstrap)
            self.terminal.append(terminal)
            self.seq_len.append(seq_len)
        return (self.publ_s, self.priv_s, self.legal_move, self.h0, self.action, self.reward, self.bootstrap, self.terminal, self.seq_len)

    def load_subset(self, filenames):
        # Prepare the hidden state once (reused for all experiences)
        shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
        hid = {"h0": torch.zeros(*shape, dtype=torch.float32), 
            "c0": torch.zeros(*shape, dtype=torch.float32)}
        
        num_workers = min(self.args.num_data_thread, mp.cpu_count() - 1)  # Use all CPUs except one

        worker_func = partial(
            process_npz_file,
            multi_step=self.multi_step
        )

        # Process files in parallel
        # Use multiprocessing to process chunks in parallel
        print(f"Loading dataset using {num_workers} processes...")
        try:
            with mp.Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(worker_func, filenames),
                    total=len(filenames),
                    desc="Loading files in parallel"
                ))
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
        finally:
            gc.collect()

        # Process results in the main thread (where CUDA is available)
        valid_results = [r for r in results if r is not None]
        for result in tqdm(valid_results, desc="Converting to tensors"):
            # Convert numpy arrays to torch tensors and move to device
            publ_s = torch.from_numpy(result['publ_s']).to(torch.float32, non_blocking=True)
            priv_s = torch.from_numpy(result['priv_s']).to(torch.float32, non_blocking=True)
            legal_move = torch.from_numpy(result['legal_move']).to(torch.bool, non_blocking=True)
            action = torch.from_numpy(result['action']).to(torch.int64, non_blocking=True)
            reward = torch.from_numpy(result['reward']).to(torch.float32, non_blocking=True)
            terminal = torch.from_numpy(result['terminal']).to(torch.bool, non_blocking=True)
            bootstrap = torch.from_numpy(result['bootstrap']).to(torch.bool, non_blocking=True)
            seq_len = torch.tensor(result['seq_len'], dtype=torch.int32)
            
            # Create experience and add to memory
            self.publ_s.append(publ_s)
            self.priv_s.append(priv_s)
            self.legal_move.append(legal_move)
            self.h0.append(hid)
            self.action.append(action)
            self.reward.append(reward)
            self.bootstrap.append(bootstrap)
            self.terminal.append(terminal)
            self.seq_len.append(seq_len)
        return (self.publ_s, self.priv_s, self.legal_move, self.h0, self.action, self.reward, self.bootstrap, self.terminal, self.seq_len)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.publ_s = dataset[0]
        self.priv_s = dataset[1]
        self.legal_move = dataset[2]
        self.h0 = dataset[3]
        self.action = dataset[4]
        self.reward = dataset[5]
        self.bootstrap = dataset[6]
        self.terminal = dataset[7]
        self.seq_len = dataset[8]

    def __len__(self):
        return len(self.seq_len)
    
    def __getitem__(self, idx):
        return (self.publ_s[idx], self.priv_s[idx], self.legal_move[idx], self.h0[idx], self.action[idx], self.reward[idx], self.bootstrap[idx], self.terminal[idx], self.seq_len[idx])

def process_batch(batch_data, args):
    device = args.train_device
    sample = namedtuple("batch", field_names=["obs", "h0", "action", "reward", "bootstrap", "terminal", "seq_len"])

    # Stack tensors efficiently using zip(*batch_data)
    publ_s, priv_s, legal_move, h0, action, reward, bootstrap, terminal, seq_len = batch_data

    sample.obs = {
        "publ_s": publ_s.transpose(1, 0).to(device, non_blocking=True),
        "priv_s": priv_s.transpose(1, 0).to(device, non_blocking=True),
        "legal_move": legal_move.transpose(1, 0).to(device, non_blocking=True),
    }
    sample.h0 = {
        "h0": h0["h0"].transpose(1, 0).to(device, non_blocking=True),
        "c0": h0["c0"].transpose(1, 0).to(device, non_blocking=True),
    }
    sample.action = {
        "a" : action.transpose(1, 0).to(device, non_blocking=True)
    }
    sample.reward = reward.transpose(1, 0).to(device, non_blocking=True)
    sample.bootstrap = bootstrap.transpose(1, 0).to(device, non_blocking=True)
    sample.terminal = terminal.transpose(1, 0).to(device, non_blocking=True)
    sample.seq_len = seq_len.to(device, non_blocking=True)
    weights = torch.ones(len(seq_len), dtype=torch.float32, device=device)
    return sample, weights  

class ChunkedDataLoader:
    def __init__(self, args, chunk_size=1000):
        self.args = args
        self.data_path = args.dataset_path
        self.chunk_size = chunk_size
        self.current_chunk = 0
        self.total_chunks = 0
        self.filenames = self._get_all_filenames()
        self.chunks = self._split_into_chunks()
        self.current_loader = None
        
    def _get_all_filenames(self):
        folders = os.listdir(self.data_path)
        filenames = []
        for folder in folders:
            filenames.extend(glob(os.path.join(self.data_path, folder, '*.npz')))
        return filenames
    
    def _split_into_chunks(self):
        chunks = [self.filenames[i:i + self.chunk_size] 
                  for i in range(0, len(self.filenames), self.chunk_size)]
        self.total_chunks = len(chunks)
        print(f"Dataset split into {self.total_chunks} chunks of ~{self.chunk_size} files each")
        return chunks
    
    def load_next_chunk(self):
        """Load the next chunk of data"""
        if self.current_chunk >= self.total_chunks:
            print("All chunks have been processed. Resetting to the first chunk.")
            self.current_chunk = 0
            
        print(f"Loading chunk {self.current_chunk + 1}/{self.total_chunks}")
        
        # Clear previous data if any
        if self.current_loader is not None:
            del self.current_loader
            gc.collect()
        
        # Load new chunk
        dataset_loader = LoadDataset(self.args)
        train_dataset = dataset_loader.load_subset(self.chunks[self.current_chunk])
        
        train_loader = DataLoader(train_dataset)
        self.current_loader = torch.utils.data.DataLoader(
            dataset=train_loader,
            batch_size=self.args.batchsize,
            shuffle=True,
            num_workers=self.args.num_data_thread,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=self.args.prefetch
        )
        
        self.current_chunk += 1
        return self.current_loader
    
    # def __iter__(self):
    #     """Returns an iterator over all chunks"""
    #     self.current_chunk = 0
    #     return self
    
    # def __next__(self):
    #     """Get the next chunk's data loader"""
    #     if self.current_chunk >= self.total_chunks:
    #         raise StopIteration
    #     return self.load_next_chunk()


if __name__ == "__main__":
    args = parse_args()
    # replay_buffer = PrioritizedReplayBuffer(args)
    # replay_buffer.load(args.dataset_path)
    # print("Dataset loaded.")
    # exit(0)

    dataset_loader = LoadDataset(args)
    train_dataset = dataset_loader.load()
    import pdb; pdb.set_trace()
    dataset_rewards = train_dataset[5]
    if not os.path.exists('/data/kmirakho/vdn-offline-data-seed-777/data_rewards'):
        os.makedirs('/data/kmirakho/vdn-offline-data-seed-777/data_rewards')
    # rewards = []
    # # import pdb; pdb.set_trace()
    # for i in range(len(dataset_rewards)):
    #     rewards.append(dataset_rewards[i].sum().item())
    # rewards = np.array(rewards)

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
            # round to 2 decimal places
            adjusted_rewards[pos] = round(adjusted_rewards[pos], 1)
        # Convert back to tensor and add to the new dataset
        actual_dataset_rewards.append(adjusted_rewards.sum())
    actual_dataset_rewards = np.array(actual_dataset_rewards)
    np.save('/data/kmirakho/vdn-offline-data-seed-777/data_rewards/rewards_data_80.npy', actual_dataset_rewards)
    
    # Replace the original rewards with the adjusted ones
    # train_dataset = list(train_dataset)
    # train_dataset[5] = new_dataset_rewards
    # train_dataset = tuple(train_dataset)
    
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # train_loader = DataLoader(train_dataset)

    # batch_loader = torch.utils.data.DataLoader(
    #     dataset=train_loader,
    #     batch_size=args.batchsize,
    #     shuffle=True,
    #     num_workers=args.num_data_thread,
    #     persistent_workers=True,
    #     pin_memory=True,
    #     prefetch_factor=args.prefetch
    # )
    # print("Dataset loaded.")
    # for batch in tqdm(batch_loader):
    #     sample, weights = process_batch(batch, args)
    # for i in tqdm(range(10000000)):
    #     sample, weights = replay_buffer.sample()  
    #     # print(sample.obs['publ_s'].shape)
    #     del sample, weights
    #     torch.cuda.empty_cache()
    #     gc.collect()

    # start_time = time.time()
    # data = np.load('/data/kmirakho/offline-rl-data/data_10/64.npz')
    # print("Data loaded in: ", time.time() - start_time)

    # data = {k: data[k] for k in data.files}
    # with open('data.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # print("Data saved to data.pkl")

    # # Load the data back from the pickle file
    # start_time = time.time()
    # with open('data.pkl', 'rb') as f:
    #     loaded_data = pickle.load(f)
    # print("Data loaded from pickle in: ", time.time() - start_time)

    # chunked_loader = ChunkedDataLoader(args, chunk_size=100*args.batchsize)
    # for epoch in range(1000):
    #     chunked_loader.load_next_chunk()
    #     epoch_bar = tqdm(chunked_loader.current_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
    #     for batch in epoch_bar:
    #         sample, weights = process_batch(batch, args)