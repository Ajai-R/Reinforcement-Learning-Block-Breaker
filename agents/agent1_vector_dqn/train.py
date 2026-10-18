"""
Agent 1: Training script for 134-Dimensional Vector Dueling MLP Q-Network.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, os.path.join(project_root, "training"))
sys.path.insert(0, script_dir)

from environment_client import BreakoutClientEnv
from replay_buffer import ReplayBuffer
from model import Agent1DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train Agent 1: Vector Dueling DQN")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-steps", type=int, default=60000)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--target-update-freq", type=int, default=1000)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train():
    args = parse_args()
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "training_log.txt")

    env = BreakoutClientEnv(frame_skip=args.frame_skip)
    agent = Agent1DQNAgent(state_dim=134, num_actions=3, learning_rate=args.lr, device=args.device)
    buffer = ReplayBuffer(capacity=args.buffer_size, state_dim=134)

    total_steps = 0
    best_avg_reward = -float("inf")
    recent_rewards = []

    log_file = open(log_file_path, "w", encoding="utf-8")
    log_file.write(f"Agent 1 (Vector-DQN) Training Log | Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.flush()

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset(max_steps=args.max_steps)
        ep_reward = 0.0

        for step in range(args.max_steps):
            total_steps += 1
            eps_progress = min(1.0, total_steps / max(1, args.epsilon_decay_steps))
            eps = args.epsilon_start + eps_progress * (args.epsilon_end - args.epsilon_start)

            action = agent.select_action(obs, eps)
            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            if total_steps >= args.warmup_steps and buffer.size >= args.batch_size:
                b_s, b_a, b_r, b_s_next, b_d = buffer.sample(args.batch_size, device=args.device)
                q_eval = agent.policy_net(b_s).gather(1, b_a)

                with torch.no_grad():
                    best_next_a = agent.policy_net(b_s_next).argmax(dim=1, keepdim=True)
                    q_next = agent.target_net(b_s_next).gather(1, best_next_a)
                    q_target = b_r + (args.gamma * q_next * (1.0 - b_d))

                loss = F.smooth_l1_loss(q_eval, q_target)
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=5.0)
                agent.optimizer.step()

            if total_steps % args.target_update_freq == 0:
                agent.update_target_network()

            if done:
                break

        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 30:
            recent_rewards.pop(0)
        avg30 = float(np.mean(recent_rewards))
        broken = info.get("broken_count", 0)

        print(f"Ep {episode:4d}/{args.episodes:4d} | Rwd: {ep_reward:7.2f} (Avg30: {avg30:6.2f}) | Bricks: {broken:2d}/88 | Steps: {step+1:4d}")
        log_file.write(f"Ep {episode:4d} | Rwd: {ep_reward:.2f} | Avg30: {avg30:.2f} | Bricks: {broken}/88 | Steps: {step+1}\n")
        log_file.flush()

        if avg30 > best_avg_reward and total_steps >= args.warmup_steps:
            best_avg_reward = avg30
            agent.save(os.path.join(checkpoints_dir, "best_model.pth"), extra_meta={"episode": episode, "avg_reward": avg30})

    agent.save(os.path.join(checkpoints_dir, "final_model.pth"), extra_meta={"episode": args.episodes, "avg_reward": avg30})
    log_file.close()
    print("Training finished.")


if __name__ == "__main__":
    train()
