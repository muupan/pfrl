"""A training script of ACER on OpenAI Gym Mujoco environments."""
import argparse
import logging
import os
import sys

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import gym
import gym.wrappers
import numpy as np
import torch
from torch import nn

import pfrl
from pfrl import experiments, replay_buffers, utils


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="Hopper-v2",
        help="OpenAI Gym MuJoCo env to perform algorithm on.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=4, help="Number of envs to run in parallel."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 32)")
    parser.add_argument(
        "--load", type=str, default="", help="Directory to load agent from."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10 ** 6,
        help="Total number of timesteps to train the agent.",
    )
    parser.add_argument(
        "--eval-n-runs",
        type=int,
        default=100,
        help="Number of episodes run for each evaluation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50000,
        help="Interval in timesteps between evaluations.",
    )
    parser.add_argument(
        "--replay-start-size",
        type=int,
        default=1000,
        help="Minimum replay buffer size before " + "performing gradient updates.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render env states in a GUI window."
    )
    parser.add_argument(
        "--demo", action="store_true", help="Just run evaluation, not training."
    )
    parser.add_argument(
        "--monitor", action="store_true", help="Wrap env with gym.wrappers.Monitor."
    )
    parser.add_argument(
        "--log-level", type=int, default=logging.INFO, help="Level of the root logger."
    )
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)

    args.outdir = experiments.prepare_output_dir(args, args.outdir, argv=sys.argv)
    print("Output files are saved in {}".format(args.outdir))

    # Set a random seed used in PFRL
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and num_envs=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and num_envs=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.num_envs) + args.seed * args.num_envs
    assert process_seeds.max() < 2 ** 32

    def make_env(process_idx, test):
        env = gym.make(args.env)
        # Unwrap TimiLimit wrapper
        assert isinstance(env, gym.wrappers.TimeLimit)
        env = env.env
        # Use different random seeds for train and test envs
        process_seed = int(process_seeds[process_idx])
        env_seed = 2 ** 32 - 1 - process_seed if test else process_seed
        env.seed(env_seed)
        # Cast observations to float32 because our model uses float32
        env = pfrl.wrappers.CastObservationToFloat32(env)
        # Normalize action space to [-1, 1]^n
        env = pfrl.wrappers.NormalizeActionSpace(env)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    sample_env = make_env(process_idx=0, test=False)
    timestep_limit = sample_env.spec.max_episode_steps
    obs_space = sample_env.observation_space
    action_space = sample_env.action_space
    print("Observation space:", obs_space)
    print("Action space:", action_space)

    obs_size = obs_space.low.size
    action_size = action_space.low.size

    hidden_size = 200

    model = pfrl.agents.acer.ACERContinuousActionHead(
        pi=nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            pfrl.policies.GaussianHeadWithFixedCovariance(scale=0.3),
        ),
        v=nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ),
        adv=nn.Sequential(
            pfrl.nn.ConcatObsAndAction(),
            nn.Linear(obs_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        ),
    )
    with torch.no_grad():
        # Initialize the policy's mean close to zero
        model.pi[4].weight[:] *= 1e-2

    optimizer = pfrl.optimizers.SharedRMSpropEpsInsideSqrt(model.parameters(), lr=1e-4)
    assert optimizer.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    rbuf = replay_buffers.EpisodicReplayBuffer(10 ** 6 // args.num_envs)

    agent = pfrl.agents.ACER(
        model=model,
        optimizer=optimizer,
        t_max=50,
        gamma=0.99,
        replay_buffer=rbuf,
        replay_start_size=args.replay_start_size,
        disable_online_update=True,
        trust_region_alpha=0.995,
        trust_region_delta=0.2,
        truncation_threshold=5,
        n_times_replay=4,
        use_Q_opc=True,
    )

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_steps=None,
            n_episodes=args.eval_n_runs,
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_steps,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.num_envs,
            make_env=make_env,
            steps=args.steps,
            eval_n_steps=None,
            eval_n_episodes=args.eval_n_runs,
            eval_interval=args.eval_interval,
            save_best_so_far_agent=True,
            max_episode_len=timestep_limit,
        )


if __name__ == "__main__":
    main()
