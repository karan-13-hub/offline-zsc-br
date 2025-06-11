import torch
import pickle
from tqdm import tqdm
import concurrent.futures
from glob import glob
import h5py

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
    # Initialize lists to collect tensor batches for each field.
    publ_s_cur_list = []
    publ_s_nxt_list = []
    priv_s_cur_list = []
    priv_s_nxt_list = []
    legal_actions_curr_list = []
    legal_actions_nxt_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []

    # Iterate over each observation and collect the data.
    for i in tqdm(range(len(observations)), desc="creating dataset..."): 
        publ_s_cur_list.extend(observations[i]['publ_s'][:-1])
        publ_s_nxt_list.extend(observations[i]['publ_s'][1:])
        priv_s_cur_list.extend(observations[i]['priv_s'][:-1])
        priv_s_nxt_list.extend(observations[i]['priv_s'][1:])
        legal_actions_curr_list.extend(observations[i]['legal_move'][:-1])
        legal_actions_nxt_list.extend(observations[i]['legal_move'][1:])
        actions_list.extend(actions[i]['a'][:-1])
        rewards_list.extend(rewards[i][:-1])
        terminals_list.extend(terminals[i][:-1])

    # Convert lists to tensors.
    # publ_s_cur = torch.stack(publ_s_cur_list).to(torch.float32)
    # publ_s_nxt = torch.stack(publ_s_nxt_list).to(torch.float32)
    # priv_s_cur = torch.stack(priv_s_cur_list).to(torch.float32)
    # priv_s_nxt = torch.stack(priv_s_nxt_list).to(torch.float32)
    # legal_actions_curr = torch.stack(legal_actions_curr_list).to(torch.float32)
    # legal_actions_nxt = torch.stack(legal_actions_nxt_list).to(torch.float32)
    # actions = torch.stack(actions_list).unsqueeze(-1).to(torch.float32)
    # rewards = torch.stack(rewards_list).unsqueeze(-1).to(torch.float32)
    # terminals = torch.stack(terminals_list).unsqueeze(-1).to(torch.float32)
    return publ_s_cur_list, publ_s_nxt_list, priv_s_cur_list, priv_s_nxt_list, legal_actions_curr_list, legal_actions_nxt_list, actions_list, rewards_list, terminals_list

if __name__ == "__main__":
    filenames = glob('/data/kmirakho/offline_data/data_*.pickle')[:1]
    observations, actions, rewards, terminals = load_data(filenames)
    publ_s_cur, publ_s_nxt, priv_s_cur, priv_s_nxt, legal_actions_curr, legal_actions_nxt, actions, rewards, terminals = preprocess_data(observations, actions, rewards, terminals)

    data = {'publ_s_cur': publ_s_cur, 'publ_s_nxt': publ_s_nxt, 'priv_s_cur': priv_s_cur, 'priv_s_nxt': priv_s_nxt, 'legal_actions_curr': legal_actions_curr, 'legal_actions_nxt': legal_actions_nxt, 'actions': actions, 'rewards': rewards, 'terminals': terminals}

    with open('/data/kmirakho/offline_data/preprocess.pickle', 'wb') as f:
        pickle.dump(data, f)

