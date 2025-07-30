import torch
import pickle
import os
from tqdm import tqdm
import concurrent.futures
from glob import glob
import h5py
import numpy as np
import argparse 

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argument parser example.")
    parser.add_argument("--file_num", type=int, default=1)
    return parser.parse_args()
    

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['obs'], data['action'], data['reward'], data['terminal']

def process_file(observations, actions, rewards, terminals, args):

    dir_name = f'/data/kmirakho/vdn-offline-data-seed-777/data_{args.file_num}'
    print("Saving to: ", dir_name)
    # if os.path.exists(dir_name): print("Warning: Folder exist!!")
    os.makedirs(dir_name, exist_ok=True)
    
    idx = 0
    for obs, act, rew, term in tqdm(zip(observations, actions, rewards, terminals), total=len(observations), desc="Processing trajectories"):
        data = {}
        import pdb; pdb.set_trace()
        data['publ_s'] = np.array(obs['publ_s'],dtype=np.float32)
        data['priv_s'] = np.array(obs['priv_s'],dtype=np.float32)
        data['legal_move'] = np.array(obs['legal_move'],dtype=np.bool_)
        data['action'] = np.array(act['a'],dtype=np.int8)
        data['reward'] = np.array(rew, dtype=np.float16)
        data['terminal'] = np.array(term, dtype=np.bool_)
        filename = os.path.join(dir_name, f"{idx}.npz")
        np.savez_compressed(filename, **data)
        idx += 1


if __name__ == "__main__":
    args = parse_args()
    # for filename in filenames:
    filename = f'/data/kmirakho/vdn_offline_data_seed_777/data_obl_{args.file_num}.pickle'
    # filename = filenames[args.file_num]
    print(f"Processing file: {filename}")
    observations, actions, rewards, terminals = load_data(filename)
    print(f"Loaded data from {filename}")
    process_file(observations, actions, rewards, terminals, args)
    # observations, actions, rewards, terminals = load_data(filenames)
    # dataset = preprocess_data(observations, actions, rewards, terminals)
    # dataset = preprocess_data(observations, actions, rewards, terminals)
    # np.savez_compressed('/data/kmirakho/offline_data/dataset_rl.npz', **dataset)
    # print("Data saved to /data/kmirakho/offline_data/dataset_rl.npz")