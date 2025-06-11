import numpy as np
import torch
from tqdm import tqdm
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial
import gc
import time
Experience = namedtuple("Experience", field_names=["publ_s", "priv_s", "legal_move", "h0", "action", "reward", "bootstrap", "terminal", "seq_len"])
class Example:
    def __init__(self):
        self.num_lstm_layer = 2
        self.num_agents = 4
        self.hid_dim = 128
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory = []
        self.priorities = []
        self.max_priority = 1.0
        self.multi_step = 5
        self.experience = Experience
        self.data = None

    def load(self, filename):
        # Define LSTM hidden state shape
        shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
        hid = {
            "h0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
            "c0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
        }

        start_time = time.time()
        # Load entire dataset to memory first
        with np.load(filename) as f:
            self.data = {k: f[k] for k in f.files}
        f.close()
        print(f"Time taken to load dataset: {time.time() - start_time} seconds")

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
        return self.memory

    # def load(self, filename):
    #     # Define LSTM hidden state - use the same reference for all experiences
    #     shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
    #     hid = {
    #         "h0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
    #         "c0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
    #     }

    #     # Memory-map the file and process in batches
    #     with np.load(filename, mmap_mode='r') as data:
    #         total_samples = len(data['terminal'])
    #         self.memory = []
            
    #         # Process in larger batches for efficiency
    #         batch_size = 5000  # Adjust based on available memory
            
    #         for start_idx in tqdm(range(0, total_samples, batch_size), desc="Loading experiences"):
    #             end_idx = min(start_idx + batch_size, total_samples)
    #             batch_slice = slice(start_idx, end_idx)
                
    #             # Load and convert batch data all at once
    #             publ_s_batch = torch.tensor(data['publ_s'][batch_slice], dtype=torch.float32)
    #             priv_s_batch = torch.tensor(data['priv_s'][batch_slice], dtype=torch.float32)
    #             legal_move_batch = torch.tensor(data['legal_move'][batch_slice], dtype=torch.bool)
    #             action_batch = torch.tensor(data['action'][batch_slice], dtype=torch.int64)
    #             reward_batch = torch.tensor(data['reward'][batch_slice], dtype=torch.float32)
    #             terminal_batch = torch.tensor(data['terminal'][batch_slice], dtype=torch.bool)
                
    #             # Process each example in the batch
    #             for i in range(end_idx - start_idx):
    #                 terminal = terminal_batch[i]
                    
    #                 # Optimized bootstrap and sequence length calculation
    #                 bootstrap = torch.zeros_like(terminal, dtype=torch.bool)
                    
    #                 # Find terminal states more efficiently
    #                 term_indices = torch.where(terminal)[0]
    #                 if len(term_indices) > 0:
    #                     term_idx = term_indices[0].item()
    #                     if term_idx - self.multi_step >= 0:
    #                         bootstrap[:term_idx - self.multi_step + 1] = True
    #                     seq_len = torch.sum(~terminal[:term_idx+1]).item() + 1
    #                 else:
    #                     # No terminal state found
    #                     seq_len = len(terminal)
                    
    #                 # Create experience object
    #                 e = self.experience(
    #                     publ_s_batch[i],
    #                     priv_s_batch[i],
    #                     legal_move_batch[i],
    #                     hid,  # Reuse the same hidden state for all experiences
    #                     action_batch[i],
    #                     reward_batch[i],
    #                     bootstrap,
    #                     terminal,
    #                     seq_len
    #                 )
    #                 self.memory.append(e)
        
    #     return self.memory

    # # Multi threading loading
    # def process_batch(self, batch_data):
    #     # Unpack the batch data
    #     start_idx, end_idx, data = batch_data

    #     # Define LSTM hidden state - reused for all experiences in this batch
    #     shape = (self.num_lstm_layer, self.num_agents, self.hid_dim)
    #     hid = {
    #         "h0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
    #         "c0": torch.zeros(*shape, dtype=torch.float32).to(self.device),
    #     }

    #     # Load and convert batch data all at once
    #     publ_s_batch = torch.tensor(data['publ_s'][start_idx:end_idx], dtype=torch.float32)
    #     priv_s_batch = torch.tensor(data['priv_s'][start_idx:end_idx], dtype=torch.float32)
    #     legal_move_batch = torch.tensor(data['legal_move'][start_idx:end_idx], dtype=torch.bool)
    #     action_batch = torch.tensor(data['action'][start_idx:end_idx], dtype=torch.int64)
    #     reward_batch = torch.tensor(data['reward'][start_idx:end_idx], dtype=torch.float32)
    #     terminal_batch = torch.tensor(data['terminal'][start_idx:end_idx], dtype=torch.bool)

    #     batch_experiences = []
    #     # Process each example in the batch
    #     for i in range(end_idx - start_idx):
    #         terminal = terminal_batch[i]
            
    #         # Optimized bootstrap and sequence length calculation
    #         bootstrap = torch.zeros_like(terminal, dtype=torch.bool)
            
    #         # Find terminal states
    #         term_indices = torch.where(terminal)[0]
    #         if len(term_indices) > 0:
    #             term_idx = term_indices[0].item()
    #             if term_idx - self.multi_step >= 0:
    #                 bootstrap[:term_idx - self.multi_step + 1] = True
    #             seq_len = torch.sum(~terminal[:term_idx+1]).item() + 1
    #         else:
    #             seq_len = len(terminal)
            
    #         # Create experience object
    #         e = self.experience(
    #             publ_s_batch[i],
    #             priv_s_batch[i],
    #             legal_move_batch[i],
    #             hid,
    #             action_batch[i],
    #             reward_batch[i],
    #             bootstrap,
    #             terminal,
    #             seq_len
    #         )
    #         batch_experiences.append(e)

    #     return batch_experiences

    # def load(self, filename):
    #     # Memory-map the file
    #     with np.load(filename, mmap_mode='r') as data:
    #         total_samples = len(data['terminal'])
    #         print(f"Total samples: {total_samples}")
    #         # Process in larger batches for efficiency
    #         batch_size = 8192  # Adjust based on available memory
    #         batches = []
            
    #         # Prepare batch parameters
    #         for start_idx in range(0, total_samples, batch_size):
    #             end_idx = min(start_idx + batch_size, total_samples)
    #             batches.append((start_idx, end_idx, data))
            
    #         # Process batches in parallel using ThreadPoolExecutor
    #         with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
    #             # Use tqdm to show progress
    #             results = list(tqdm(executor.map(self.process_batch, batches), 
    #                                total=len(batches), 
    #                                desc="Loading experiences"))
            
    #         # Flatten results
    #         for batch_result in results:
    #             self.memory.extend(batch_result)
                
    #     return self.memory

    # Multi processing loading
    @staticmethod
    def process_chunk(filename, chunk_range, num_lstm_layer, num_agents, hid_dim, multi_step):
        """Process a chunk of data in a separate process"""
        start_idx, end_idx = chunk_range
        print(f"Processing chunk {start_idx} to {end_idx}")
        # Each process opens its own memory-mapped file
        try:
            with np.load(filename, mmap_mode='r') as data:
                # Pre-allocate lists for batch results
                batch_experiences = []
                
                # Define LSTM hidden state once for all experiences in this chunk
                shape = (num_lstm_layer, num_agents, hid_dim)
                hid = {
                    "h0": torch.zeros(*shape, dtype=torch.float32),
                    "c0": torch.zeros(*shape, dtype=torch.float32),
                }
                
                # Process chunk efficiently - load larger arrays at once
                publ_s_chunk = torch.tensor(data['publ_s'][start_idx:end_idx], dtype=torch.float32)
                print(f"publ_s_chunk: {publ_s_chunk.shape} for {start_idx} to {end_idx}", flush=True)
                priv_s_chunk = torch.tensor(data['priv_s'][start_idx:end_idx], dtype=torch.float32)
                print(f"priv_s_chunk: {priv_s_chunk.shape} for {start_idx} to {end_idx}", flush=True)
                legal_move_chunk = torch.tensor(data['legal_move'][start_idx:end_idx], dtype=torch.bool)
                print(f"legal_move_chunk: {legal_move_chunk.shape} for {start_idx} to {end_idx}")
                action_chunk = torch.tensor(data['action'][start_idx:end_idx], dtype=torch.int64)
                print(f"action_chunk: {action_chunk.shape} for {start_idx} to {end_idx}", flush=True)
                reward_chunk = torch.tensor(data['reward'][start_idx:end_idx], dtype=torch.float32)
                print(f"reward_chunk: {reward_chunk.shape} for {start_idx} to {end_idx}", flush=True)
                terminal_chunk = torch.tensor(data['terminal'][start_idx:end_idx], dtype=torch.bool)
                print(f"terminal_chunk: {terminal_chunk.shape} for {start_idx} to {end_idx}", flush=True)
                # Process each sample in the chunk
                
                return (publ_s_chunk, priv_s_chunk, legal_move_chunk, action_chunk, reward_chunk, terminal_chunk)
                # for i in range(120):#(end_idx - start_idx):
                #     terminal = terminal_chunk[i]
                    
                #     # Compute bootstrap and sequence length
                #     bootstrap = torch.zeros_like(terminal, dtype=torch.bool)
                #     term_indices = torch.where(terminal)[0]
                    
                #     if len(term_indices) > 0:
                #         term_idx = term_indices[0].item()
                #         if term_idx - multi_step >= 0:
                #             bootstrap[:term_idx - multi_step + 1] = True
                #         seq_len = torch.sum(~terminal[:term_idx+1]).item() + 1
                #     else:
                #         # No terminal state found
                #         seq_len = len(terminal)
                    
                #     # Create experience object
                #     e = Experience(
                #         publ_s_chunk[i],
                #         priv_s_chunk[i],
                #         legal_move_chunk[i],
                #         hid,  # Same hidden state reference for all
                #         action_chunk[i],
                #         reward_chunk[i],
                #         bootstrap,
                #         terminal,
                #         seq_len
                #     )
                #     batch_experiences.append(e)
                # print(f"ye hogaya {start_idx} to {end_idx}", flush=True)
                # return batch_experiences
        except Exception as e:
            print(f"Error processing chunk {start_idx} to {end_idx}: {e}")
            return []
        finally:
            gc.collect()

    def load(self, filename):
        # Get file size for efficient chunking
        with np.load(filename, mmap_mode='r') as data:
            total_samples = len(data['terminal'])
        data.close()
        
        # Calculate optimal chunk size based on CPU count
        num_processes = mp.cpu_count()  # Limit to 4 processes or fewer
        chunk_size = max(4096, total_samples // (num_processes * 2))  # Ensure each process has meaningful work
        
        # Prepare chunk ranges
        chunk_ranges = []
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            chunk_ranges.append((start_idx, end_idx))
        
        # Prepare the worker function with partial to pass static parameters
        worker_func = partial(
            self.process_chunk,
            filename,
            num_lstm_layer=self.num_lstm_layer,
            num_agents=self.num_agents,
            hid_dim=self.hid_dim,
            multi_step=self.multi_step
        )
        
        # Use multiprocessing to process chunks in parallel
        print(f"Loading data using {num_processes} processes...")
        try:
            with mp.Pool(processes=num_processes) as pool:
                # Process all chunks in parallel and show progress
                results = list(tqdm(
                    pool.imap(worker_func, chunk_ranges),
                    total=len(chunk_ranges),
                    desc="Processing data chunks"
                ))
        except Exception as e:
            print(f"Error loading data: {e}")
            return []


        # Combine results from all processes
        self.memory = []
        for chunk_result in results:
            self.memory.extend(chunk_result)
        
        # # Move to the desired device (after multiprocessing)
        # if self.device != 'cpu':
        #     print(f"Moving data to {self.device}...")
        #     for i in tqdm(range(len(self.memory)), desc="Moving to device"):
        #         e = self.memory[i]
        #         # Create a new experience with tensors on the correct device
        #         self.memory[i] = self.experience(
        #             e.publ_s.to(self.device),
        #             e.priv_s.to(self.device),
        #             e.legal_move.to(self.device),
        #             {"h0": e.h0["h0"].to(self.device), "c0": e.h0["c0"].to(self.device)},
        #             e.action.to(self.device),
        #             e.reward.to(self.device),
        #             e.bootstrap.to(self.device),
        #             e.terminal.to(self.device),
        #             e.seq_len
        #         )
        
        print(f"Loaded {len(self.memory)} experiences.")
        return self.memory

if __name__ == "__main__":
    # Example usage
    example = Example()
    filename = "/data/kmirakho/offline-model/dataset_rl_1040640_lrg.npz"
    memory = example.load(filename)

