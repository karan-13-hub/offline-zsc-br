import pickle
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/kmirakho/vdn-offline-data-seed1234/data_rewards")
    parser.add_argument("--save_path", type=str, default="/data/kmirakho/vdn-offline-data-seed1234/data_rewards/reward_distribution")
    parser.add_argument("--exps_path", type=str, default="./exps/train_br_seed9")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    return parser.parse_args()

#class for analyzing the reward distribution of the dataset
class PlotRewardDistribution:
    def __init__(self, args, filename=None, multiple_filenames=None):
        self.rewards = None
        if filename is not None:
            self.filename = filename
            self.rewards = np.load(self.filename)
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)

            self.data_num = self.filename.split("/")[-1].split(".")[0].split("_")[-1]
            self.folder_name = os.path.join(args.save_path, self.data_num)
            if not os.path.exists(self.folder_name):
                os.makedirs(self.folder_name)

            self.plot_name = os.path.join(self.folder_name, self.data_num)
            self.plot_reward_distribution()
            self.plot_reward_distribution_with_mean()
            self.plot_reward_distribution_with_mean_and_std()
        
        if multiple_filenames is not None:
            self.plot_multiple_reward_distribution(multiple_filenames)
            
    def plot_multiple_reward_distribution(self, multiple_filenames):
        #plot distribution of multiple rewards
        self.rewards = []
        for filename in multiple_filenames:
            self.rewards.extend(np.load(filename))
        self.rewards = np.array(self.rewards)
        self.multiple_rewards_name = []
        for filename in multiple_filenames:
            self.multiple_rewards_name.append(filename.split("/")[-1].split(".")[0].split("_")[-1])
        self.multiple_rewards_name = "_".join(self.multiple_rewards_name)

        self.folder_name = os.path.join(args.save_path, self.multiple_rewards_name)
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

        self.plot_name = os.path.join(self.folder_name, self.multiple_rewards_name)

        self.plot_reward_distribution()
        self.plot_reward_distribution_with_mean()
        self.plot_reward_distribution_with_mean_and_std()

    def plot_reward_distribution(self):
        plt.hist(self.rewards, bins=25)
        plt.xlim(0, 25)
        plt.ylim(0, 35000)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.savefig(self.plot_name + "_distribution.png")
        plt.close()
        plt.clf()

    def plot_reward_distribution_with_mean(self):
        plt.hist(self.rewards, bins=25)
        plt.xlim(0, 25)
        plt.ylim(0, 35000)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution with Mean")
        plt.axvline(self.rewards.mean(), color='r', linestyle='dashed', linewidth=2)
        plt.savefig(self.plot_name + "_distribution_with_mean.png")
        plt.close()
        plt.clf()

    def plot_reward_distribution_with_mean_and_std(self):
        plt.hist(self.rewards, bins=25)
        plt.xlim(0, 25)
        plt.ylim(0, 35000)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution with Mean and Std")
        plt.axvline(self.rewards.mean(), color='r', linestyle='dashed', linewidth=2)
        plt.axvline(self.rewards.mean() + self.rewards.std(), color='g', linestyle='dashed', linewidth=2)
        plt.axvline(self.rewards.mean() - self.rewards.std(), color='g', linestyle='dashed', linewidth=2)
        plt.savefig(self.plot_name + "_distribution_with_mean_and_std.png")
        plt.close()
        plt.clf()

#class for analyzing the training logs
class PlotTrainingLogs:
    def __init__(self, args):
        self.exps_path = args.exps_path
        self.include = args.include
        self.exclude = args.exclude
        self.filenames = []
        self.training_logs = []
        self.get_filenames()
        self.load_training_logs()
        self.plot_training_logs()
        self.plot_name = None

    def get_filenames(self):
        output_folders = glob(self.exps_path + "/*")
        for folder in output_folders:
            #check if folder is a directory
            if not os.path.isdir(folder):
                continue
            if self.include != "":
                if self.exclude != "":
                    if self.include in folder and self.exclude not in folder:
                        self.filenames.append(folder + "/train.log")
                else:
                    if self.include in folder:
                        self.filenames.append(folder + "/train.log")
            elif self.exclude != "":
                if self.exclude not in folder:
                    self.filenames.append(folder + "/train.log")
            else:
                self.filenames.append(folder + "/train.log")
    
    def load_training_logs(self):
        for filename in self.filenames:
            with open(filename, "r") as f:
                lines = f.readlines()
            self.training_logs.append(lines)

    def plot_training_logs(self):
        for log in self.training_logs:
            epochs = []
            scores = []
            for line in log:
                if "epoch" in line and "agent 0 eval score" in line:
                    line = line.split(",")
                    ep_strt = line[0].find("epoch ")+len("epoch ")
                    sc_strt = line[1].find("score:")+len("score: ")
                    epochs.append(int(line[0][ep_strt:]))
                    scores.append(float(line[1][sc_strt:]))
            plt.plot(epochs, scores)

        legends = []
        for filename in self.filenames:
            legends.append(filename.split("/")[-2].split("_")[-1])

        #plot name using the exps path and legends
        self.plot_name = os.path.join(self.exps_path, "_".join(legends))
        plt.legend(legends)
        plt.xlabel("Epochs")
        plt.ylabel("Scores")
        plt.ylim(0, 26)
        plt.axhline(25, color='m', linestyle='dashed', linewidth=2, label="Ideal")
        plt.title("Scores vs Epochs")
        plt.savefig(self.plot_name + "_scores_vs_epochs.png")
        plt.close()
        plt.clf()

if __name__ == "__main__":
    args = parse_args()
    # filenames = glob(args.dataset_path + "/*.npy")
    # print(filenames)
    # for filename in filenames:
    #     PlotRewardDistribution(args, filename, None)
    # PlotRewardDistribution(args, "/data/kmirakho/vdn-offline-data/data_rewards/rewards_expert_replay_20k.npy", None)
 
    # multiple_filenames = ["/data/kmirakho/vdn-offline-data/data_rewards/rewards_data_20.npy", "/data/kmirakho/vdn-offline-data/data_rewards/rewards_data_40.npy", "/data/kmirakho/vdn-offline-data/data_rewards/rewards_data_80.npy"]
    # # multiple_filenames = ["/data/kmirakho/vdn-offline-data/data_rewards/rewards_data_640.npy", "/data/kmirakho/vdn-offline-data/data_rewards/rewards_data_1280.npy"]
    # PlotRewardDistribution(args, None, multiple_filenames)

    PlotTrainingLogs(args)