import torch
import pickle
from tqdm import tqdm
import concurrent.futures
from glob import glob
import h5py
import numpy as np

def load_data(filenames):
    for i in tqdm(range(len(filenames)), desc="loading data files..."): 
        filename = filenames[i]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        if i == 0:
            actions = data['action']
            observations = data['obs']
            rewards = data['reward']
            terminals = data['terminal']
        else:
            observations = observations + data['obs']
            actions = actions + data['action']
            rewards = rewards + data['reward']
            terminals = terminals + data['terminal']
    return observations, actions, rewards, terminals


# def preprocess_data(observations, actions, rewards, terminals):
#     dataset = []
    
#     # Process all trajectories in a single pass
#     for obs, act, rew, term in tqdm(zip(observations, actions, rewards, terminals), 
#                                   total=len(observations), 
#                                   desc="Processing trajectories"):
#         # Calculate valid indices for transitions (current -> next)
#         n_steps = len(obs['publ_s'])
#         for i in range(n_steps - 1):
#             dataset.append({
#                 # Current state
#                 'publ_s_curr': obs['publ_s'][i],
#                 'priv_s_curr': obs['priv_s'][i],
#                 'legal_actions_curr': obs['legal_move'][i],
                
#                 # Next state
#                 'publ_s_next': obs['publ_s'][i+1],
#                 'priv_s_next': obs['priv_s'][i+1],
#                 'legal_actions_next': obs['legal_move'][i+1],
                
#                 # Transition data
#                 'action': act['a'][i],
#                 'reward': rew[i],
#                 'terminal': term[i]
#             })
    
#     return dataset

def preprocess_data(observations, actions, rewards, terminals):
    # Preallocate numpy arrays with optimized dtypes
    n_trans = sum(len(obs['publ_s'])-1 for obs in observations)
    dataset = {
        'publ_s_curr': np.empty((n_trans, *observations[0]['publ_s'][0].shape), dtype=np.float32),
        'priv_s_curr': np.empty((n_trans, *observations[0]['priv_s'][0].shape), dtype=np.float32),
        'legal_actions_curr': np.empty((n_trans, *observations[0]['legal_move'][0].shape), dtype=np.bool_),
        'publ_s_next': np.empty((n_trans, *observations[0]['publ_s'][0].shape), dtype=np.float32),
        'priv_s_next': np.empty((n_trans, *observations[0]['priv_s'][0].shape), dtype=np.float32),
        'legal_actions_next': np.empty((n_trans, *observations[0]['legal_move'][0].shape), dtype=np.bool_),
        'action': np.empty((n_trans, *actions[0]['a'][0].shape), dtype=np.int8),
        'reward': np.empty(n_trans, dtype=np.float16),
        'terminal': np.empty(n_trans, dtype=np.bool_)
    }
    
    idx = 0
    # Process all trajectories in a single pass
    for obs, act, rew, term in tqdm(zip(observations, actions, rewards, terminals), 
                                  total=len(observations), 
                                  desc="Processing trajectories"):
        n_steps = len(obs['publ_s'])
        trans = n_steps - 1
        slc = slice(idx, idx+trans)
        
        dataset['publ_s_curr'][slc] = obs['publ_s'][:-1]
        dataset['priv_s_curr'][slc] = obs['priv_s'][:-1]
        dataset['legal_actions_curr'][slc] = obs['legal_move'][:-1]
        dataset['publ_s_next'][slc] = obs['publ_s'][1:]
        dataset['priv_s_next'][slc] = obs['priv_s'][1:]
        dataset['legal_actions_next'][slc] = obs['legal_move'][1:]
        dataset['action'][slc] = act['a'][:-1]
        dataset['reward'][slc] = rew[:-1]
        dataset['terminal'][slc] = term[:-1]
        
        idx += trans
    
    return dataset


if __name__ == "__main__":
    filenames = [f'/data/kmirakho/offline_data/data/data_{i}.pickle' 
                for i in [20, 40, 80, 160, 320, 640, 1280]]
    observations, actions, rewards, terminals = load_data(filenames)
    dataset = preprocess_data(observations, actions, rewards, terminals)
    np.savez_compressed('/data/kmirakho/offline_data/data/dataset_model.npz', **dataset)
    print("Data saved to /data/kmirakho/offline_data/data/dataset_model.npz")