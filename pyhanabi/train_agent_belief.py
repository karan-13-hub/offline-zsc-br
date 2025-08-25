# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time
import os
import sys
import argparse
import pprint
import pickle

import numpy as np
import torch
from tqdm import tqdm

from act_group import ActGroup
from create import create_envs, create_threads
import r2d2
import common_utils
import rela
import utils
import belief_model
from eval import evaluate
import random
from replay_load import PrioritizedReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser(description="train agent with belief model")
    parser.add_argument("--save_dir", type=str, default="exps/dev_belief")
    parser.add_argument("--method", type=str, default="vdn")
    
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--hid_dim", type=int, default=512)
    parser.add_argument("--fc_only", type=int, default=0)
    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--act_device", type=str, default="cuda:1")

    # load policy config
    parser.add_argument("--policy", type=str, default="")
    parser.add_argument("--explore", type=int, default=1)
    parser.add_argument("--rand", type=int, default=0)
    parser.add_argument("--clone_bot", type=int, default=0)
    parser.add_argument("--num_data_thread", type=int, default=4)

    # aux loss settings
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--aux_weight", type=float, default=0.0)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument("--num_lstm_layer", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=50, help="max grad norm")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--save_model_after", type=int, default=100)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=80000)
    parser.add_argument("--replay_buffer_size", type=int, default=2 ** 20)
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )

    # thread setting
    parser.add_argument("--num_thread", type=int, default=40, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=20)
    parser.add_argument("--num_eval_after", type=int, default=100)

    # load from dataset setting
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--num_player", type=int, default=2)
    parser.add_argument("--inf_data_loop", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--shuffle_color", type=int, default=0)
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--sad", type=int, default=0)
    parser.add_argument("--hide_action", type=int, default=0)
    parser.add_argument("--multi_step", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--actor_sync_freq", type=int, default=100)

    args = parser.parse_args()
    return args


def create_rl_context(args):
    agent_overwrite = {
        "vdn": False,
        "device": args.train_device,  # batch runner will create copy on act device
        "uniform_priority": True,
    }

    if args.clone_bot:
        agent = utils.load_supervised_agent(args.policy, args.train_device)
        cfgs = {
            "act_base_eps": 0.1,
            "act_eps_alpha": 7,
            "num_game_per_thread": 80,
            "num_player": 2,
            "train_bomb": 0,
            "max_len": 80,
            "sad": 0,
            "shuffle_color": 0,
            "hide_action": 0,
            "multi_step": 1,
            "gamma": 0.999,
        }
    else:
        agent, cfgs = utils.load_agent(args.policy, agent_overwrite)

    assert cfgs["shuffle_color"] == False
    assert args.explore

    agent.sync_target_with_online()
    agent = agent.to(args.train_device)
    print(agent)
    
    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        args.priority_exponent,
        args.priority_weight,
        args.prefetch,
    )

    if args.rand:
        explore_eps = [1]
    elif args.explore:
        # use the same exploration config as policy learning
        explore_eps = utils.generate_explore_eps(
            cfgs["act_base_eps"], cfgs["act_eps_alpha"], cfgs["num_game_per_thread"]
        )
    else:
        explore_eps = [0]

    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))
    if args.clone_bot or not agent.boltzmann:
        print("no boltzmann act")
        boltzmann_t = []
    else:
        boltzmann_beta = utils.generate_log_uniform(
            1 / cfgs["max_t"], 1 / cfgs["min_t"], cfgs["num_t"]
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )

    act_group = ActGroup(
        args.act_device,
        agent,
        args.seed,
        args.num_thread,
        args.num_game_per_thread,
        args.num_player,
        explore_eps,
        boltzmann_t,
        args.method,
        args.sad,
        args.shuffle_color if not args.rand else False,
        args.hide_action,
        False,  # not trinary, need full hand for prediction
        replay_buffer,
        args.multi_step,  # not used
        args.max_len,
        args.gamma,  # not used
        False,  # turn off off-belief rewardless of how it is trained
        None,  # belief_model
    )

    context, threads = create_threads(
        args.num_thread,
        args.num_game_per_thread,
        act_group.actors,
        games,
    )
    return agent, cfgs, replay_buffer, games, act_group, context, threads


def create_sl_context(args):
    games = pickle.load(open(args.dataset, "rb"))
    print(f"total num game: {len(games)}")
    if args.shuffle_color:
        # to make data generation speed roughly the same as consumption
        args.num_thread = 10
        args.inf_data_loop = 1

    if args.replay_buffer_size < 0:
        args.replay_buffer_size = len(games) * args.num_player
    if args.burn_in_frames < 0:
        args.burn_in_frames = len(games) * args.num_player

    # priority not used
    priority_exponent = 1.0
    priority_weight = 0.0
    replay_buffer = rela.RNNPrioritizedReplay(
        args.replay_buffer_size,
        args.seed,
        priority_exponent,
        priority_weight,
        args.prefetch,
    )
    data_gen = hanalearn.CloneDataGenerator(
        replay_buffer,
        args.num_player,
        args.max_len,
        args.shuffle_color,
        False,
        args.num_thread,
    )
    game_params = {
        "players": str(args.num_player),
        "random_start_player": "0",
        "bomb": "0",
    }
    data_gen.set_game_params(game_params)
    for i, g in enumerate(games):
        data_gen.add_game(g["deck"], g["moves"])
        if (i + 1) % 10000 == 0:
            print(f"{i+1} games added")

    return data_gen, replay_buffer

def evaluate_agent(agent, args):
    eval_seed = (9917 + random.randint(0, 999999)) % 7777777
    score, perfect, *_ = evaluate(
        [agent for _ in range(args.num_player)],
        1000,
        eval_seed,
        args.eval_bomb,
        0,  # explore eps
        args.sad,
        args.hide_action,
        device=args.train_device,
    )
    return score, perfect

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 2)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.dataset is None or len(args.dataset) == 0:
        (
            agent,
            cfgs,
            replay_buffer,
            games,
            act_group,
            context,
            threads,
        ) = create_rl_context(args)
        
    else:
        data_gen, replay_buffer = create_sl_context(args)
        data_gen.start_data_generation(args.inf_data_loop, args.seed)
        # only for getting feature size
        games = create_envs(1, 1, args.num_player, 0, args.max_len)
        cfgs = {"sad": False}

    if args.dataset is None or len(args.dataset) == 0:
        optim_agent = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=args.eps)
        eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
        eval_agent.load_state_dict(agent.state_dict())
        score, perfect = evaluate_agent(eval_agent, args)
        print("\n\n================")
        print(f"Initial eval score for pretrained model: {score}, perfect: {perfect}")
        print("=======================\n\n")
        act_group.start()
        context.start()

    while replay_buffer.size() < args.burn_in_frames:
        print("warming up replay buffer:", replay_buffer.size())
        time.sleep(1)

    print("Success, Done")
    print("=======================")

    if args.load_model:
        belief_config = utils.get_train_config(cfgs["belief_model"])
        print("load belief model from:", cfgs["belief_model"])
        model = belief_model.ARBeliefModel.load(
            cfgs["belief_model"],
            args.train_device,
            5,
            0,
            belief_config["fc_only"],
        )
    else:
        model = belief_model.ARBeliefModel(
            args.train_device,
            games[0].feature_size(cfgs["sad"])[1],
            args.hid_dim,
            5,  # hand_size
            25,  # bits per card
            0,  # num_sample
            fc_only=args.fc_only,
        ).to(args.train_device)

    optim_belief = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

    stat_belief = common_utils.MultiCounter(args.save_dir)
    tachometer_belief = utils.Tachometer()
    stopwatch_belief = common_utils.Stopwatch()

    stat_agent = common_utils.MultiCounter(args.save_dir)
    tachometer_agent = utils.Tachometer()
    stopwatch_agent = common_utils.Stopwatch()
    
    for epoch in range(args.num_epoch):
        print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer_belief.start()
        stat_belief.reset()
        stopwatch_belief.reset()

        tachometer_agent.start()
        stat_agent.reset()
        stopwatch_agent.reset()

        for batch_idx in tqdm(range(args.epoch_len), desc=f'Training', bar_format='{l_bar}{bar:20}{r_bar}', leave=True):
            num_update = batch_idx + epoch * args.epoch_len

            if num_update % args.num_update_between_sync == 0:
                print(f"\nSynced target with online for agent")
                agent.sync_target_with_online()

            if num_update % args.actor_sync_freq == 0:
                act_group.update_model(agent)
            
            torch.cuda.synchronize()
            stopwatch_belief.time("sync and updating belief")
            stopwatch_agent.time("sync and updating agent")

            batch, weight = replay_buffer.sample(args.batchsize, args.train_device)
            stopwatch_agent.time("sample data agent")
            stopwatch_belief.time("sample data belief")

            assert weight.max() == 1
            loss_belief, xent, xent_v0, _ = model.loss(batch)
            loss_belief = loss_belief.mean()
            loss_belief.backward()

            stopwatch_belief.time("forward & backward belief")

            loss_agent, priority, online_q = agent.loss(batch, args.aux_weight, stat_agent)
            loss_agent = (loss_agent * weight).mean()
            loss_agent.backward()

            stopwatch_agent.time("forward & backward agent")

            g_norm_belief = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim_belief.step()
            optim_belief.zero_grad()

            g_norm_agent = torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
            optim_agent.step()
            optim_agent.zero_grad()

            torch.cuda.synchronize()
            stopwatch_belief.time("update belief")
            stopwatch_agent.time("update agent")

            replay_buffer.update_priority(priority)
            stopwatch_agent.time("update priority")

            stat_belief["loss"].feed(loss_belief.detach().item())
            stat_belief["grad_norm"].feed(g_norm_belief)
            stat_belief["xent_pred"].feed(xent.detach().mean().item())
            stat_belief["xent_v0"].feed(xent_v0.detach().mean().item())

            stat_agent["loss"].feed(loss_agent.detach().item())
            stat_agent["grad_norm"].feed(g_norm_agent)
            stat_agent["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

        print("EPOCH: %d" % epoch)

        if args.dataset is None or len(args.dataset) == 0:
            scores = [g.last_episode_score() for g in games]
            print("mean score: %.2f" % np.mean(scores))

        count_factor = args.num_player if args.method == "vdn" else 1
        print("=====Belief======")
        tachometer_belief.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch_belief.summary()
        stat_belief.summary(epoch)
        
        print("\n=====Agent======")
        tachometer_agent.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch_agent.summary()
        stat_agent.summary(epoch)

        if epoch % args.num_eval_after == 0:
            eval_seed = (9917 + epoch * 999999) % 7777777
            eval_agent.load_state_dict(agent.state_dict())
            score, perfect, *_ = evaluate(
                [eval_agent for _ in range(args.num_player)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
                device=args.train_device,
            )
            print("\n================")
            print(f"Agent eval score: {score}, perfect: {perfect}")
        else:
            score = 0
            perfect = 0

        force_save_name = None
        if epoch > 0 and epoch % args.save_model_after == 0:
            force_save_name = f"model_seed_{args.seed}"
        belief_saved = saver.save(
            None,
            model.state_dict(),
            -stat_belief["loss"].mean(),
            True,
            force_save_name=force_save_name,
        )
        print(f"\nBelief model saved: {belief_saved}")

        force_save_name = None
        if epoch > 0 and epoch % args.save_model_after == 0:
            force_save_name = args.load_model.split("/")[-1].split(".")[0]
        agent_saved = saver.save(
            None, agent.online_net.state_dict(), score, force_save_name=force_save_name,
        )
        print(f"Agent saved: {agent_saved}")
        print("===================")
