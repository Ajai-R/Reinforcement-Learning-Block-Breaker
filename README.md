# Block Breaker RL Studio & Vector-DQN (134-Dim) Training Suite

A high-performance Breakout simulator and Deep Reinforcement Learning (Vector-DQN) suite built with React, Vite, HTML5 Canvas, and PyTorch. Features the **Space Invader Army ($8 \times 16$)** layout, continuous sub-stepped collision detection, real-time telemetry, and direct JavaScript simulation execution for 100% training-demo parity.

---

## Features

- **Space Invader Army Layout ($8 \times 16$)**: 88 breakable bricks (Cyan, Emerald, Amber, Reinforced Pink) with steel unbreakable boundary accents.
- **Continuous Collision Detection (CCD)**: 4 micro-substeps per frame eliminating tunneling and corner grazing artifacts.
- **134-Dimensional Vector Observation ($s_t \in \mathbb{R}^{134}$)**:
  `[paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_remaining_ratio, ...brick_layout_8x16]`
- **Exact JavaScript Simulation Parity**:
  - The Python RL trainer connects directly to the real JavaScript environment engine (`frontend/src/engine/rlEnvironment.js`) via a high-speed Node.js IPC bridge.
  - Zero divergence in physics, floating-point geometry, boundary reflections, or rewards.
- **Vector Deep Q-Network (Dueling MLP)**:
  - Input dimension: 134 continuous features
  - Hidden layers: $256 \to 256 \to 128$
  - Output: 3 Q-values (`0: LEFT`, `1: STAY`, `2: RIGHT`)
  - Double DQN Loss: Decoupled target Q-value evaluation with Smooth L1 (Huber) loss.
- **Controls (Human Play)**:
  - `A` / `D` or `ArrowLeft` / `ArrowRight`: Move paddle
  - `Space` or Click: Launch ball / Pause / Resume
  - `R`: Reset episode

---

## Project Structure

```
rl-catch-game/
├── frontend/                        # Interactive React 19 + Canvas Studio
│   ├── src/
│   │   ├── components/
│   │   │   ├── GameBoard.jsx        # Canvas renderer with neon VFX & screen shake
│   │   │   ├── Header.jsx           # HUD stats & active player mode badge
│   │   │   ├── RLTelemetryPanel.jsx # Live observation tensors & Q-values / PPO probabilities
│   │   │   ├── RewardChart.jsx      # Cumulative reward SVG curve
│   │   │   └── RulesModal.jsx       # Physics rules & MDP formulation
│   │   ├── engine/
│   │   │   ├── rlEnvironment.js     # Exact physics engine, substepping, 134D state
│   │   │   ├── dqnInference.js      # Pure JS in-browser Conv2D + MLP neural inference
│   │   │   ├── levels.js            # Space Invader Army (8x16) layout & brick definitions
│   │   │   ├── particles.js         # Particle emitter, screen shake, floating scores
│   │   │   └── audio.js             # Web Audio API procedural sound synthesizer
│   │   ├── assets/
│   │   │   ├── model_weights_vector.json # Exported Agent 1 weights
│   │   │   ├── model_weights_cnn.json    # Exported Agent 2 weights (88/88 Cleared)
│   │   │   └── model_weights_ppo.json    # Exported Agent 3 weights
│   │   ├── App.jsx                  # Main application orchestrator & 4-way mode switcher
│   │   └── index.css                # Custom glassmorphism & neon styles
│   └── package.json
│
├── agents/                          # Self-Contained Agent Workspaces
│   ├── agent1_vector_dqn/           # Agent 1: 134D Vector Dueling MLP
│   │   ├── model.py, train.py, export.py
│   │   ├── checkpoints/best_model.pth
│   │   └── logs/
│   ├── agent2_cnn_dqn/              # Agent 2: Compact Spatial Grid CNN (88/88 Winner)
│   │   ├── model.py, train.py, export.py
│   │   ├── checkpoints/best_model.pth (Ep 535)
│   │   └── logs/training_log.txt
│   └── agent3_ppo/                  # Agent 3: PPO Actor-Critic (GAE & Clipped Loss)
│       ├── model.py, train.py, export.py
│       ├── checkpoints/
│       └── logs/
│
├── training/                        # Shared Environment Bridge & Evaluation Utilities
│   ├── environment_client.py        # Python Gym-like client executing real JS environment
│   ├── node_env_runner.js           # High-speed Node.js IPC simulation bridge
│   ├── replay_buffer.py             # Pre-allocated circular experience replay buffer
│   ├── evaluate.py                  # Universal evaluation benchmark for all 3 agents
│   └── export_model.py              # Universal weight export utility
│
├── PROJECT_WORKFLOW.md              # In-depth architectural workflow guide
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Deep Reinforcement Learning (Vector-DQN)

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Training the Model
When you are ready to train the model, run:
```bash
# Train on the exact JavaScript environment with default parameters (1000 episodes)
python training/train.py

# Custom training options example:
python training/train.py --episodes 1500 --lr 2.5e-4 --batch-size 64 --device cuda
```

**Training Arguments:**
- `--episodes`: Total training episodes (default: `1000`)
- `--frame-skip`: Action repeat / frame skip (default: `4` — agent acts every 66ms, eliminates jitter)
- `--max-steps`: Maximum agent steps per episode (default: `2000` = 8000 frames)
- `--batch-size`: Batch size for SGD update (default: `64`)
- `--lr`: Learning rate (default: `2.5e-4`)
- `--gamma`: Discount factor (default: `0.99`)
- `--buffer-size`: Replay buffer capacity (default: `50000`)
- `--warmup-steps`: Initial random exploration steps (default: `1000`)
- `--epsilon-decay-steps`: Steps over which epsilon decays $1.0 \to 0.02$ (default: `60000`)
- `--target-update-freq`: Target network hard update interval (default: `1000` steps)
- `--save-freq`: Periodic model checkpoint frequency (default: every `50` episodes)
- `--device`: Target device (`cuda` or `cpu`)

### 3. Evaluating a Trained Model
```bash
# Evaluate the best saved checkpoint over 20 greedy episodes on the real JS environment
python training/evaluate.py --checkpoint training/checkpoints/best_model.pth --episodes 20
```

---

## Interactive React Studio

### 1. Install Node Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.
