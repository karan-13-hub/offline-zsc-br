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

import numpy as np
import torch
from torch import nn

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils
from process_data import save_replay_buffer
from replay_load import PrioritizedReplayBuffer
from replay_load import LoadDataset, DataLoader, process_batch
from replay_load import ChunkedDataLoader
from tqdm import tqdm
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
    parser.add_argument("--load_after", type=int, default=20)
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)

    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )

    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )

    parser.add_argument(
        "--visit_weight", type=float, default=0.5, help="visit weight"
    )

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
    parser.add_argument("--save_model_after", type=int, default=100)
    parser.add_argument("--actor_sync_freq", type=int, default=10)
    parser.add_argument("--dataset_path", type=str, default="/data/kmirakho/offline-model/dataset_rl_1040640_sml.npz")

    #behaviour policy
    parser.add_argument("--bc", type=bool, default=False)
    parser.add_argument("--bc_weight", type=float, default=0.0)


    args = parser.parse_args()
    if args.off_belief == 1:
        args.method = "iql"
        args.multi_step = 1
        assert args.net in ["publ-lstm"], "should only use publ-lstm style network"
        assert not args.shuffle_color
    assert args.method in ["vdn", "iql"]
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    logger_path = os.path.join(args.save_dir, "train.log")
    sys.stdout = common_utils.Logger(logger_path)
    saver = common_utils.TopkSaver(args.save_dir, 5)

    common_utils.set_all_seeds(args.seed)
    pprint.pprint(vars(args))

    if args.method == "vdn":
        args.batchsize = int(np.round(args.batchsize / args.num_player))
        args.replay_buffer_size //= args.num_player
        args.burn_in_frames //= args.num_player

    explore_eps = utils.generate_explore_eps(
        args.act_base_eps, args.act_eps_alpha, args.num_t
    )
    
    expected_eps = np.mean(explore_eps)
    print("explore eps:", explore_eps)
    print("avg explore eps:", np.mean(explore_eps))

    if args.boltzmann_act:
        boltzmann_beta = utils.generate_log_uniform(
            1 / args.max_t, 1 / args.min_t, args.num_t
        )
        boltzmann_t = [1 / b for b in boltzmann_beta]
        print("boltzmann beta:", ", ".join(["%.2f" % b for b in boltzmann_beta]))
        print("avg boltzmann beta:", np.mean(boltzmann_beta))
    else:
        boltzmann_t = []
        print("no boltzmann")

    games = create_envs(
        args.num_thread * args.num_game_per_thread,
        args.seed,
        args.num_player,
        args.train_bomb,
        args.max_len,
    )
    # print(games[0].feature_size(args.sad))
    # print(games[0].num_action())

    agent = r2d2.R2D2Agent(
        (args.method == "vdn"),
        args.multi_step,
        args.gamma,
        args.eta,
        args.train_device,
        games[0].feature_size(args.sad),
        args.rnn_hid_dim,
        games[0].num_action(),
        args.net,
        args.num_lstm_layer,
        args.boltzmann_act,
        False,  # uniform priority
        args.off_belief,
    )
    agent.sync_target_with_online()

    if args.load_model and args.load_model != "None":
        if args.off_belief and args.belief_model != "None":
            belief_config = utils.get_train_config(args.belief_model)
            args.load_model = belief_config["policy"]

        print("*****loading pretrained model*****")
        print(args.load_model)
        utils.load_weight(agent.online_net, args.load_model, args.train_device)
        print("*****done*****")

    # use clone bot for additional bc loss
    if args.clone_bot and args.clone_bot != "None":
        clone_bot = utils.load_supervised_agent(args.clone_bot, args.train_device)
    else:
        clone_bot = None

    agent = agent.to(args.train_device)
    optim = torch.optim.Adam(agent.online_net.parameters(), lr=args.lr, eps=args.eps)
    print(agent)
    eval_agent = agent.clone(args.train_device, {"vdn": False, "boltzmann_act": False})
    
    # replay_buffer = rela.RNNPrioritizedReplay(
    #     args.replay_buffer_size,
    #     args.seed,
    #     args.priority_exponent,
    #     args.priority_weight,
    #     args.prefetch,
    # )

    belief_model = None
    if args.off_belief and args.belief_model != "None":
        print(f"load belief model from {args.belief_model}")
        from belief_model import ARBeliefModel

        belief_devices = args.belief_device.split(",")
        belief_config = utils.get_train_config(args.belief_model)
        belief_model = []
        for device in belief_devices:
            belief_model.append(
                ARBeliefModel.load(
                    args.belief_model,
                    device,
                    5,
                    args.num_fict_sample,
                    belief_config["fc_only"],
                )
            )

    # act_group = ActGroup(
    #     args.act_device,
    #     agent,
    #     args.seed,
    #     args.num_thread,
    #     args.num_game_per_thread,
    #     args.num_player,
    #     explore_eps,
    #     boltzmann_t,
    #     args.method,
    #     args.sad,
    #     args.shuffle_color,
    #     args.hide_action,
    #     True,  # trinary, 3 bits for aux task
    #     replay_buffer,
    #     args.multi_step,
    #     args.max_len,
    #     args.gamma,
    #     args.off_belief,
    #     belief_model,
    # )

    # context, threads = create_threads(
    #     args.num_thread,
    #     args.num_game_per_thread,
    #     act_group.actors,
    #     games,
    # )

    # act_group.start()
    # context.start()

    # while replay_buffer.size() < args.burn_in_frames:
    #     print("warming up replay buffer:", replay_buffer.size())
    #     time.sleep(1)
    # context.pause() 
    # act_group.stop()
    
    # print("Loading dataset...")
    train_dataset = LoadDataset(args)
    train_dataset = train_dataset.load()
    train_loader = DataLoader(train_dataset)
    batch_loader = torch.utils.data.DataLoader(
        dataset=train_loader,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_data_thread,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=args.prefetch
    )
    print("Dataset loaded")
    # chunked_loader = ChunkedDataLoader(args, chunk_size=1000*args.batchsize)
    # print("Buffer size: ", train_loader.__len__())
    print("Success, Done")
    print("=======================")

    frame_stat = dict()
    frame_stat["num_acts"] = 0
    frame_stat["num_buffer"] = 0

    stat = common_utils.MultiCounter(args.save_dir)
    tachometer = utils.Tachometer()
    stopwatch = common_utils.Stopwatch()
    
    for epoch in range(args.num_epoch+1):
        # print("beginning of epoch: ", epoch)
        print(common_utils.get_mem_usage())
        tachometer.start()
        stat.reset()
        stopwatch.reset()
        # if epoch % args.load_after == 0:
        #     chunked_loader.load_next_chunk()
        # epoch_bar = tqdm(chunked_loader.current_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        epoch_bar = tqdm(batch_loader, desc=f'Epoch {epoch}', bar_format='{l_bar}{bar:20}{r_bar}', leave=True)
        for batch_idx, sample in enumerate(epoch_bar):
            # print("Buffer size: ", replay_buffer.size())
            num_update = batch_idx + epoch * args.epoch_len
            if num_update % args.num_update_between_sync == 0:
                print("\nUpdated the target with online\n")
                agent.sync_target_with_online()
            # if num_update % args.actor_sync_freq == 0:
                # act_group.update_model(agent)

            torch.cuda.synchronize()
            stopwatch.time("sync and updating")
            
            start = time.time()
            batch, weight = process_batch(sample, args)
            stopwatch.time("sample data")
            # print("sample time: ", time.time() - start)

            start = time.time()
            loss, priority, online_q = agent.loss(batch, args.aux_weight, stat, args.bc, args.bc_weight)
            # print("loss time: ", time.time() - start)

            if clone_bot is not None and args.clone_weight > 0:
                bc_loss = agent.behavior_clone_loss(
                    online_q, batch, args.clone_t, clone_bot, stat
                )
                loss = loss + bc_loss * args.clone_weight
            loss = (loss * weight).mean()
            loss.backward()

            torch.cuda.synchronize()
            stopwatch.time("forward & backward")

            g_norm = torch.nn.utils.clip_grad_norm_(
                agent.online_net.parameters(), args.grad_clip
            )
            optim.step()
            optim.zero_grad()

            torch.cuda.synchronize()
            stopwatch.time("update model")

            # start = time.time()
            # replay_buffer.update_priorities(priority)   
            # stopwatch.time("updating priority")
            # print("update priority time: ", time.time() - start)

            stat["loss"].feed(loss.detach().item())
            stat["grad_norm"].feed(g_norm)
            # stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

            # Add periodically to track memory usage
            # print(f"Memory allocated on {args.train_device}: {torch.cuda.memory_allocated(args.train_device) / 1e9:.2f} GB")
            # print(f"Memory reserved on {args.train_device}: {torch.cuda.memory_reserved(args.train_device) / 1e9:.2f} GB")

            del batch, weight
            if batch_idx % 100 == 0:
                if torch.cuda.memory_allocated(args.train_device) > 0.9 * torch.cuda.get_device_properties(args.train_device).total_memory:
                    torch.cuda.empty_cache()
                    gc.collect()
        count_factor = args.num_player if args.method == "vdn" else 1
        print("EPOCH: %d" % epoch)
        # tachometer.lap(replay_buffer, args.epoch_len * args.batchsize, count_factor)
        stopwatch.summary()
        stat.summary(epoch)

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
        else :
            score = 0
            perfect = 0
        
        force_save_name = None
        if epoch % args.save_model_after == 0:
            force_save_name = f"model_seed_{args.seed}_epoch_{epoch}"
        model_saved = saver.save(
            None, agent.online_net.state_dict(), score, force_save_name=force_save_name
        )
        print(
            "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
            % (epoch, score, perfect * 100, model_saved)
        )
        # save_rb_epochs = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120]
        # if epoch in save_rb_epochs:
        #     save_replay_buffer(replay_buffer, epoch, score)

        if clone_bot is not None:
            score, perfect, *_ = evaluate(
                [clone_bot] + [eval_agent for _ in range(args.num_player - 1)],
                1000,
                eval_seed,
                args.eval_bomb,
                0,  # explore eps
                args.sad,
                args.hide_action,
            )
            print(f"clone bot score: {np.mean(score)}")

        # if args.off_belief:
        #     actors = common_utils.flatten(act_group.actors)
        #     success_fict = [actor.get_success_fict_rate() for actor in actors]
        #     print(
        #         "epoch %d, success rate for sampling ficticious state: %.2f%%"
        #         % (epoch, 100 * np.mean(success_fict))
        #     )
        # print("Max priority: ", replay_buffer.max_priority)

        print("==========")
