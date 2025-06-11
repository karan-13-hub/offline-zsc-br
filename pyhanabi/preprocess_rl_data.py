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

def preprocess_data(observations, actions, rewards, terminals):
    n_eps = len(observations)
    dataset = {
        'publ_s': np.empty((n_eps, *observations[0]['publ_s'].shape), dtype=np.float32),
        'priv_s': np.empty((n_eps, *observations[0]['priv_s'].shape), dtype=np.float32),
        'legal_move': np.empty((n_eps, *observations[0]['legal_move'].shape), dtype=np.bool_),
        'action': np.empty((n_eps, *actions[0]['a'].shape), dtype=np.int8),
        'reward': np.empty((n_eps, *rewards[0].shape), dtype=np.float16),
        'terminal': np.empty((n_eps, *terminals[0].shape), dtype=np.bool_)
    }
    
    idx = 0
    # Process all trajectories in a single pass
    for obs, act, rew, term in tqdm(zip(observations, actions, rewards, terminals), 
                                  total=len(observations), 
                                  desc="Processing trajectories"):
        
        dataset['publ_s'][idx] = obs['publ_s']
        dataset['priv_s'][idx] = obs['priv_s']
        dataset['legal_move'][idx] = obs['legal_move']
        dataset['action'][idx] = act['a']
        dataset['reward'][idx] = rew
        dataset['terminal'][idx] = term
        idx += 1
    return dataset


if __name__ == "__main__":
    filenames = [f'/data/kmirakho/offline_data/data/data_{i}.pickle' 
                for i in [10, 20, 40, 80, 160, 320, 640, 1280]]
    observations, actions, rewards, terminals = load_data(filenames)
    dataset = preprocess_data(observations, actions, rewards, terminals)
    np.savez_compressed('/data/kmirakho/offline_data/dataset_rl.npz', **dataset)
    print("Data saved to /data/kmirakho/offline_data/dataset_rl.npz")