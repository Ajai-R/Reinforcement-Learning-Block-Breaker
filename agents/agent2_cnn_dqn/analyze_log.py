import os
import sys

log_file = os.path.join(os.path.dirname(__file__), 'logs', 'training_log.txt')
if not os.path.exists(log_file):
    print(f"Log file not found: {log_file}")
    sys.exit(1)

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

episodes = []
for line in lines:
    parts = [p.strip() for p in line.split('|')]
    if len(parts) >= 11 and parts[0].isdigit():
        ep = int(parts[0])
        steps = int(parts[1])
        tot_steps = int(parts[2])
        reward = float(parts[3])
        avg30 = float(parts[4])
        score = int(parts[5])
        bricks = int(parts[6].split('/')[0])
        outcome = parts[10]
        episodes.append({
            'ep': ep, 'steps': steps, 'tot_steps': tot_steps,
            'reward': reward, 'avg30': avg30, 'score': score,
            'bricks': bricks, 'outcome': outcome
        })

print(f"Total episodes parsed: {len(episodes)}")
max_b = max(episodes, key=lambda x: x['bricks'])
print(f"Max Bricks: Ep {max_b['ep']}, Bricks: {max_b['bricks']}/88, Reward: {max_b['reward']}, Outcome: {max_b['outcome']}")

peak_avg = max(episodes, key=lambda x: x['avg30'])
print(f"Peak Avg30: Ep {peak_avg['ep']}, Avg30: {peak_avg['avg30']}, Reward: {peak_avg['reward']}, Bricks: {peak_avg['bricks']}")

print("\nEpisodes with >= 45 bricks:")
high_clears = [x for x in episodes if x['bricks'] >= 45]
print(f"Count: {len(high_clears)}")
for h in high_clears:
    print(f"  Ep {h['ep']}: Bricks {h['bricks']}/88, Reward {h['reward']:.1f}, Steps {h['steps']}, Avg30: {h['avg30']:.1f}")

print("\n100-Episode Window Breakdown:")
for i in range(0, len(episodes), 100):
    chunk = episodes[i:i+100]
    avg_b = sum(x['bricks'] for x in chunk)/len(chunk)
    max_b_chunk = max(x['bricks'] for x in chunk)
    avg_r = sum(x['reward'] for x in chunk)/len(chunk)
    print(f"Eps {chunk[0]['ep']:4d}-{chunk[-1]['ep']:4d}: Avg Bricks: {avg_b:5.1f} | Max Bricks: {max_b_chunk:2d} | Avg Reward: {avg_r:7.1f}")
