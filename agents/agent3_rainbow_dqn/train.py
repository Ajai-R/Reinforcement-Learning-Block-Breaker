"""
Agent 3: Fine-Tuning Script for Full Rainbow DQN (All 6 Extensions).
1. Double Q-learning (Decoupled action selection via online net, evaluation via target net)
2. Prioritized Experience Replay (SumTree with importance sampling weights)
3. Dueling Architecture (Separated value and advantage categorical distributions)
4. Multi-step Learning (n-step TD target accumulation, n=3)
5. Distributional C51 RL (51 discrete support atoms, cross-entropy projection loss)
6. Noisy Networks + Epsilon Fine-Tuning Schedule

Optimized for:
- Strict 1-Life Zero-Tolerance Survival (Never drops the ball)
- Speedrun Step Penalty (-0.02/step for rapid board clearance)
- Master Tunneling Architecture (Outer-edge slicing + Wall excavation + Ceiling breach & sustain)
"""

import os
import sys
import time
import argparse
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
sys.path.insert(0, os.path.join(project_root, "training"))
sys.path.insert(0, script_dir)

from environment_client import BreakoutClientEnv
from prioritized_replay import PrioritizedReplayBuffer
from model import RainbowDuelingC51Network
from export import export


def parse_args():
    parser = argparse.ArgumentParser(description="Train Agent 3: Rainbow DQN Tunneling & 1-Life Speedrun")
    parser.add_argument("--episodes", type=int, default=350, help="Total fine-tuning episodes")
    parser.add_argument("--max-steps", type=int, default=4000, help="Maximum steps per episode")
    parser.add_argument("--lives", type=int, default=1, help="Lives per episode (1 for strict zero-tolerance survival)")
    parser.add_argument("--eps-start", type=float, default=0.15, help="Initial epsilon for fine-tuning exploration")
    parser.add_argument("--eps-end", type=float, default=0.01, help="Final epsilon for fine-tuning exploration")
    parser.add_argument("--eps-decay-episodes", type=int, default=200, help="Episodes to decay epsilon")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size for SGD")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (Adam)")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--n-step", type=int, default=3, help="Multi-step return horizon")
    parser.add_argument("--num-atoms", type=int, default=51, help="Number of discrete support atoms (C51)")
    parser.add_argument("--v-min", type=float, default=-30.0, help="Minimum support value for C51")
    parser.add_argument("--v-max", type=float, default=150.0, help="Maximum support value for C51")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Prioritized replay buffer capacity")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER prioritization exponent alpha")
    parser.add_argument("--beta-start", type=float, default=0.5, help="PER importance sampling beta start")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Exploration steps before training")
    parser.add_argument("--target-update-freq", type=int, default=1000, help="Steps between target net syncs")
    parser.add_argument("--frame-skip", type=int, default=4, help="Action repeat frame skip")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=os.path.join(script_dir, "checkpoints", "best_model.pth"), help="Resume checkpoint path")
    parser.add_argument("--start-episode", type=int, default=1, help="Starting episode index for logging")
    return parser.parse_args()


def train():
    args = parse_args()
    device = torch.device(args.device)

    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    logs_dir = os.path.join(script_dir, "logs")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_file_path = os.path.join(logs_dir, "training_log.txt")
    best_path = os.path.join(checkpoints_dir, "best_model.pth")

    gamma_n = args.gamma ** args.n_step
    delta_z = (args.v_max - args.v_min) / (args.num_atoms - 1)
    support = torch.linspace(args.v_min, args.v_max, args.num_atoms, device=device)

    print("=" * 85)
    print("🌈 LAUNCHING AGENT 3: RAINBOW DQN MASTER TUNNELING & SPEEDRUN FINE-TUNING")
    print(f"• Total Episodes: {args.episodes} | Strict Lives per Episode: {args.lives}")
    print(f"• Rewards: Tunneling Excavation (+15) + Ceiling Breach (+50) + Edge Slicing (+10) + Speedrun (-0.02/step)")
    print(f"• Exploration: Epsilon {args.eps_start} → {args.eps_end} + NoisyNet")
    print(f"• C51 Support: {args.num_atoms} Atoms [{args.v_min:.1f}, {args.v_max:.1f}] (Δz = {delta_z:.2f})")
    print(f"• Device: {device} | Learning Rate: {args.lr}")
    if args.resume and os.path.exists(args.resume):
        print(f"• Resuming Weights From: {args.resume}")
    print(f"• Logs: {log_file_path}")
    print("=" * 85)

    env = BreakoutClientEnv(frame_skip=args.frame_skip, render_frame=False)
    policy_net = RainbowDuelingC51Network(
        in_channels=3,
        num_actions=3,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
    ).to(device)

    target_net = RainbowDuelingC51Network(
        in_channels=3,
        num_actions=3,
        num_atoms=args.num_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
    ).to(device)

    # Resume from checkpoint if available
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy_net.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"✅ Successfully loaded pre-trained model weights from {args.resume}")

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr, eps=1e-5)
    buffer = PrioritizedReplayBuffer(
        capacity=args.buffer_size,
        state_dim=134,
        alpha=args.alpha,
        beta_start=args.beta_start,
        beta_frames=80000,
    )
    n_step_buffer = deque(maxlen=args.n_step)

    best_avg_reward = -float("inf")
    best_broken = 0
    total_steps = 0
    recent_rewards = []

    log_file = open(log_file_path, "a", encoding="utf-8")
    log_file.write(f"\n--- Rainbow DQN Tunneling & Speedrun Session | Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_file.write(f"{'Ep':>5} | {'Steps':>6} | {'TotSteps':>9} | {'Reward':>9} | {'AvgRwd30':>9} | {'Bricks':>7} | {'Outcome':<14}\n")
    log_file.flush()

    for ep_idx in range(1, args.episodes + 1):
        episode = ep_idx + args.start_episode - 1

        # 1. Cosine learning rate decay
        prog = ep_idx / args.episodes
        current_lr = args.lr * max(0.2, 0.5 * (1.0 + np.cos(np.pi * prog)))
        for g in optimizer.param_groups:
            g["lr"] = current_lr

        # 2. Epsilon fine-tuning schedule
        if ep_idx <= args.eps_decay_episodes:
            eps = args.eps_start - (args.eps_start - args.eps_end) * ((ep_idx - 1) / max(1, args.eps_decay_episodes - 1))
        else:
            eps = args.eps_end

        # 3. Reset environment
        obs, _ = env.reset(max_steps=args.max_steps, full_reset=True)
        ep_reward = 0.0
        lives = args.lives
        ceiling_breached = False
        n_step_buffer.clear()

        for step in range(args.max_steps):
            total_steps += 1

            # Select action via hybrid NoisyNet + Epsilon
            policy_net.reset_noise()
            action = policy_net.get_action(obs, epsilon=eps)

            next_obs, raw_reward, terminated, truncated, info = env.step(action)

            broken = info.get("broken_count", 0)
            breakable = info.get("breakable_count", 88)
            cleared = breakable > 0 and broken >= breakable

            # -------------------------------------------------------------
            # MASTER REWARD SHAPING: TUNNELING, 1-LIFE SURVIVAL & SPEEDRUN
            # -------------------------------------------------------------
            shaped_reward = raw_reward

            # A. Speedrun step penalty: incentivizes finishing rapidly
            shaped_reward -= 0.02

            # B. Continuous Intercept Guidance (Ball Tracking in lower third)
            # When ball is moving down (vy < 0) near the paddle level (y < -0.3)
            if next_obs[4] < 0 and next_obs[2] < -0.3:
                track_dist = abs(next_obs[0] - next_obs[1])  # |paddleX - ballX|
                shaped_reward -= 0.08 * track_dist

            # C. Outer-Edge Paddle Slicing Incentive (+10.0)
            # Ball deflected off paddle (vy flipped negative -> positive near paddle height)
            if obs[4] < 0 and next_obs[4] > 0 and obs[2] < -0.65:
                edge_offset = abs(obs[1] - obs[0])  # |ballX - paddleX|
                if edge_offset > 0.045:  # Outer 30% of paddle -> sharp 30°-60° angle cut toward side walls
                    shaped_reward += 10.0
                else:
                    shaped_reward += 3.0  # Basic catch

            # D. Side Wall Channel Excavation Bonus (+15.0 per outer-column brick)
            # Col 0 (left edge) and Col 15 (right edge) carve the vertical tunnel to ceiling
            for r in range(8):
                idx_left = 6 + r * 16 + 0
                if obs[idx_left] > 0.0 and next_obs[idx_left] == 0.0:
                    shaped_reward += 15.0
                idx_right = 6 + r * 16 + 15
                if obs[idx_right] > 0.0 and next_obs[idx_right] == 0.0:
                    shaped_reward += 15.0

            # E. Tunneling Entry / Ceiling Breach Jackpot (+50.0)
            # normBallY >= 0.88 corresponds to penetrating above the top brick layer
            if obs[2] < 0.88 and next_obs[2] >= 0.88:
                if not ceiling_breached:
                    ceiling_breached = True
                    shaped_reward += 50.0

            # F. Ceiling Frenzy / Tunnel Sustain (+3.0 per step)
            # Rewards maintaining the ball trapped in the upper corridor destroying bricks from behind
            if next_obs[2] >= 0.88:
                shaped_reward += 3.0

            # G. Strict Ball Drop Penalty vs Full Stage Clear Jackpot
            life_lost = terminated and not cleared
            if life_lost:
                lives -= 1
                shaped_reward -= 25.0  # Strict penalty: drops are completely unacceptable!

            if cleared:
                shaped_reward += 100.0  # Full clear jackpot

            ep_reward += shaped_reward

            # Termination condition
            done = (lives <= 0) or cleared or truncated

            # Multi-Step Return Accumulation
            n_step_buffer.append((obs, action, shaped_reward, next_obs, done))

            if len(n_step_buffer) == args.n_step:
                r_n = sum((args.gamma ** idx) * trans[2] for idx, trans in enumerate(n_step_buffer))
                s_0 = n_step_buffer[0][0]
                a_0 = n_step_buffer[0][1]
                s_n = n_step_buffer[-1][3]
                d_n = n_step_buffer[-1][4]
                buffer.push(s_0, a_0, r_n, s_n, d_n)

            # Categorical Bellman Projection Update (C51 + Double Q + Dueling)
            if total_steps >= args.warmup_steps and len(buffer) >= args.batch_size:
                b_s, b_a, b_r, b_s_next, b_d, b_w, indices = buffer.sample(args.batch_size, device=device)

                with torch.no_grad():
                    target_net.reset_noise()
                    policy_net.reset_noise()
                    _, next_q = policy_net(b_s_next)
                    best_next_a = next_q.argmax(dim=1)

                    target_probs, _ = target_net(b_s_next)
                    p_next = target_probs[torch.arange(args.batch_size, device=device), best_next_a]

                    Tz = b_r + (gamma_n * (1.0 - b_d)) * support.unsqueeze(0)
                    Tz = Tz.clamp(args.v_min, args.v_max)

                    b = (Tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.num_atoms - 1).long()
                    u = b.ceil().clamp(0, args.num_atoms - 1).long()

                    eq_mask = (u == l)
                    l_weight = (u.float() - b)
                    u_weight = (b - l.float())
                    l_weight[eq_mask] = 1.0
                    u_weight[eq_mask] = 0.0

                    m = torch.zeros((args.batch_size, args.num_atoms), dtype=torch.float32, device=device)
                    offset = (
                        torch.linspace(0, (args.batch_size - 1) * args.num_atoms, args.batch_size, device=device)
                        .unsqueeze(1)
                        .expand(args.batch_size, args.num_atoms)
                        .long()
                    )

                    m.view(-1).index_add_(0, (l + offset).view(-1), (p_next * l_weight).view(-1))
                    m.view(-1).index_add_(0, (u + offset).view(-1), (p_next * u_weight).view(-1))

                policy_net.reset_noise()
                probs, _ = policy_net(b_s)
                p_a = probs[torch.arange(args.batch_size, device=device), b_a.squeeze(1)]

                log_p = torch.log(p_a.clamp(min=1e-6))
                elementwise_loss = -torch.sum(m * log_p, dim=1)
                loss = torch.mean(b_w.squeeze(1) * elementwise_loss)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                optimizer.step()

                buffer.update_priorities(indices, elementwise_loss.detach().cpu().numpy())

            if total_steps % args.target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

            # If multi-life mode enabled and ball dropped:
            if life_lost and lives > 0:
                while len(n_step_buffer) > 1:
                    n_step_buffer.popleft()
                    r_n = sum((args.gamma ** idx) * trans[2] for idx, trans in enumerate(n_step_buffer))
                    s_0 = n_step_buffer[0][0]
                    a_0 = n_step_buffer[0][1]
                    s_n = n_step_buffer[-1][3]
                    d_n = n_step_buffer[-1][4]
                    buffer.push(s_0, a_0, r_n, s_n, d_n)
                n_step_buffer.clear()
                obs, _ = env.reset(max_steps=args.max_steps, full_reset=False)
            else:
                obs = next_obs

        # Drain remaining n-step transitions
        while len(n_step_buffer) > 1:
            n_step_buffer.popleft()
            r_n = sum((args.gamma ** idx) * trans[2] for idx, trans in enumerate(n_step_buffer))
            s_0 = n_step_buffer[0][0]
            a_0 = n_step_buffer[0][1]
            s_n = n_step_buffer[-1][3]
            d_n = n_step_buffer[-1][4]
            buffer.push(s_0, a_0, r_n, s_n, d_n)
        n_step_buffer.clear()

        recent_rewards.append(ep_reward)
        if len(recent_rewards) > 30:
            recent_rewards.pop(0)
        avg30 = float(np.mean(recent_rewards))

        outcome = "CLEARED" if broken >= 88 else ("MAX_STEPS" if step + 1 >= args.max_steps else "BALL_MISSED")
        print(f"Ep {episode:4d} | Steps: {step+1:4d} | Rwd: {ep_reward:7.2f} | Avg30: {avg30:7.2f} | Bricks: {broken:2d}/88 | {outcome} (eps={eps:.2f})")
        log_file.write(f"{episode:5d} | {step+1:6d} | {total_steps:9d} | {ep_reward:9.2f} | {avg30:9.2f} | {broken:2d}/88   | {outcome:<14}\n")
        log_file.flush()

        # Save and auto-export checkpoint if new best rolling average
        if avg30 > best_avg_reward and total_steps >= args.warmup_steps:
            best_avg_reward = avg30
            best_broken = max(best_broken, broken)
            meta = {
                "episode": episode,
                "avg_reward": avg30,
                "broken": broken,
                "architecture": "rainbow_c51_dqn",
                "num_atoms": args.num_atoms,
                "v_min": args.v_min,
                "v_max": args.v_max,
            }
            torch.save(
                {
                    "model_state_dict": policy_net.state_dict(),
                    "meta": meta,
                },
                best_path,
            )
            try:
                export(best_path)
            except Exception as e:
                print(f"[Warning] Auto-export failed: {e}")

    final_path = os.path.join(checkpoints_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": policy_net.state_dict(),
            "meta": {
                "episode": episode,
                "avg_reward": avg30,
                "broken": broken,
                "architecture": "rainbow_c51_dqn",
                "num_atoms": args.num_atoms,
                "v_min": args.v_min,
                "v_max": args.v_max,
            },
        },
        final_path,
    )
    log_file.close()
    print("=" * 85)
    print("🎉 RAINBOW DQN TUNNELING & SPEEDRUN FINE-TUNING COMPLETE!")
    print(f"• Best 30-Episode Avg: {best_avg_reward:.2f}")
    print(f"• Final Checkpoint: {final_path}")
    print("=" * 85)
    export(best_path)


if __name__ == "__main__":
    train()
