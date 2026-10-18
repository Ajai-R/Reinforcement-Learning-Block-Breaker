"""
Evaluation Script for Trained Vector-DQN Breakout Agent.

Loads a saved PyTorch checkpoint (e.g. checkpoints/best_model.pth),
executes N greedy evaluation episodes on the EXACT JavaScript Breakout environment,
and reports performance statistics:
- Mean Reward & Standard Deviation
- Mean Bricks Cleared (out of 88)
- Win Rate / Stage Clear Rate %
- Average Episode Duration & Score
"""

import os
import sys
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add paths so agent modules import cleanly
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment_client import BreakoutClientEnv
import agents.agent1_vector_dqn.model as m_agent1
import agents.agent2_cnn_dqn.model as m_agent2
import agents.agent3_dqn.model as m_agent3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Trained Agent on Breakout")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to checkpoint .pth file")
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes to run")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Exploration rate during evaluation (default 0.01)")
    parser.add_argument("--max-steps", type=int, default=3000, help="Max steps per episode")
    parser.add_argument("--frame-skip", type=int, default=4, help="Action repeat frame skip (default: 4)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    device = torch.device(args.device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(args.checkpoint):
        checkpoint_path = args.checkpoint
    elif os.path.exists(args.checkpoint):
        checkpoint_path = os.path.abspath(args.checkpoint)
    else:
        checkpoint_path = os.path.abspath(os.path.join(script_dir, args.checkpoint))

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location=device)
    policy_sd = ckpt.get("policy_state_dict", ckpt.get("model_state_dict", ckpt))

    # Determine model type
    is_pixel_dqn = False
    if "conv1.weight" in policy_sd:
        arch_name = "Agent 3: DeepMind Nature DQN (84x84x4 Pixels)"
        net = m_agent3.DeepMindDQN(in_channels=4, num_actions=3).to(device)
        net.load_state_dict(policy_sd)
        is_pixel_dqn = True
    elif "conv.0.weight" in policy_sd:
        arch_name = "Agent 2: Spatial Grid CNN (3x16x16 Dueling)"
        net = m_agent2.SpatialDuelingDQN(in_channels=3, num_actions=3).to(device)
        net.load_state_dict(policy_sd)
    else:
        arch_name = "Agent 1: Vector-DQN (134D Dueling MLP)"
        net = m_agent1.DuelingVectorDQN(state_dim=134, num_actions=3).to(device)
        net.load_state_dict(policy_sd)

    net.eval()

    print("=" * 75)
    print("🧪 EVALUATION: BREAKOUT REINFORCEMENT LEARNING AGENT")
    print(f"• Checkpoint: {checkpoint_path}")
    print(f"• Architecture: {arch_name}")
    print(f"• Episodes: {args.episodes} | Frame Skip: {args.frame_skip} | Device: {device}")
    print("=" * 75)

    env = BreakoutClientEnv(frame_skip=args.frame_skip, render_frame=is_pixel_dqn)
    rewards = []
    lengths = []
    scores = []
    bricks_broken_list = []
    stage_clears = 0

    try:
        for ep in range(1, args.episodes + 1):
            if is_pixel_dqn:
                init_frame, info = env.reset(max_steps=args.max_steps)
                state = np.repeat(init_frame[np.newaxis, :, :], 4, axis=0)
            else:
                state, info = env.reset(max_steps=args.max_steps)

            ep_reward = 0.0

            for step in range(args.max_steps):
                with torch.no_grad():
                    if is_pixel_dqn:
                        st = torch.as_tensor(state, device=device).unsqueeze(0)
                    else:
                        st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

                    if np.random.rand() < args.epsilon:
                        action = np.random.randint(0, 3)
                    else:
                        action = int(net(st).argmax(dim=1).item())

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += reward

                if is_pixel_dqn:
                    next_state = np.empty((4, 84, 84), dtype=np.uint8)
                    next_state[:3] = state[1:]
                    next_state[3] = next_obs
                    state = next_state
                else:
                    state = next_obs

                if terminated or truncated:
                    break

            is_clear = step_info.get("broken_count", 0) >= step_info.get("breakable_count", 88)
            if is_clear:
                stage_clears += 1

            rewards.append(ep_reward)
            lengths.append(step + 1)
            scores.append(step_info.get("score", 0))
            bricks_broken_list.append(step_info.get("broken_count", 0))

            status_str = "🏆 STAGE CLEARED" if is_clear else "💀 BALL DROPPED"
            print(
                f"Episode {ep:2d}/{args.episodes:2d} | "
                f"Reward: {ep_reward:7.2f} | "
                f"Score: {step_info.get('score', 0):4d} | "
                f"Bricks: {step_info.get('broken_count', 0):2d}/88 | "
                f"Steps: {step+1:4d} | {status_str}"
            )

        # 3. Statistical Report Summary
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        mean_score = float(np.mean(scores))
        mean_bricks = float(np.mean(bricks_broken_list))
        win_rate = (stage_clears / args.episodes) * 100.0

        print("=" * 75)
        print("📊 EVALUATION SUMMARY REPORT")
        print(f"• Mean Episode Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"• Mean Game Score:     {mean_score:.1f} points")
        print(f"• Mean Bricks Cleared: {mean_bricks:.1f} / 88 ({mean_bricks / 88 * 100:.1f}%)")
        print(f"• Stage Win Rate:      {win_rate:.1f}% ({stage_clears}/{args.episodes} wins)")
        print(f"• Max Reward Achieved: {np.max(rewards):.2f}")
        print("=" * 75)

        # Save to TXT File
        eval_report_path = os.path.join(os.path.dirname(checkpoint_path), "evaluation_report.txt")
        with open(eval_report_path, "w", encoding="utf-8") as f:
            f.write("=" * 75 + "\n")
            f.write("BREAKOUT EVALUATION REPORT\n")
            f.write(f"• Model Evaluated: {checkpoint_path}\n")
            f.write(f"• Evaluation Episodes: {args.episodes} (Greedy Policy: epsilon={args.epsilon})\n")
            f.write("-" * 75 + "\n")
            f.write(f"• Mean Episode Reward: {mean_reward:.2f} ± {std_reward:.2f}\n")
            f.write(f"• Mean Game Score:     {mean_score:.1f} points\n")
            f.write(f"• Mean Bricks Cleared: {mean_bricks:.1f} / 88 ({mean_bricks / 88 * 100:.1f}%)\n")
            f.write(f"• Stage Win Rate:      {win_rate:.1f}% ({stage_clears}/{args.episodes} wins)\n")
            f.write(f"• Max Reward Achieved: {np.max(rewards):.2f}\n")
            f.write("=" * 75 + "\n")
        print(f"📄 Evaluation report saved to: {eval_report_path}")

    finally:
        env.close()


if __name__ == "__main__":
    evaluate()
