# Block Breaker RL Studio & Multi-Agent Benchmark Suite

A high-performance Breakout simulator and comparative Deep Reinforcement Learning suite built with React 19, Vite, HTML5 Canvas, and PyTorch. Features the **Space Invader Army ($8 \times 16$)** layout, continuous sub-stepped collision detection, real-time telemetry, and direct JavaScript simulation execution for 100% training-demo parity across **three distinct RL architectures**.

---

## The Three Reinforcement Learning Agents

This project implements, trains, benchmarks, and deploys three distinct architectures designed to solve the Block Breaker environment:

```
+--------------------------------------------------------------------------------------------------+
|                                    MULTI-AGENT ARCHITECTURE                                      |
+--------------------------+-----------------------------------+-----------------------------------+
| Agent 1: Vector DQN      | Agent 2: Compact Spatial CNN      | Agent 3: Distributional Rainbow   |
+--------------------------+-----------------------------------+-----------------------------------+
| • 134-Dim Two-Stream MLP | • 3-Channel 16x16 Spatial Grid    | • Multi-Step C51 Distributional   |
| • Kinematic (6) +        | • Convolutions (32->64->64) +     | • Prioritized Replay (PER)        |
|   Bricks (128)           |   Kinematic Fusion (6->64)        | • Dueling + NoisyNet Layers       |
| • Lightweight Baseline   | • Full 88/88 Stage Cleared        | • Superhuman Aiming & Tunneling   |
+--------------------------+-----------------------------------+-----------------------------------+
```

### 1. Agent 1: 134-Dimensional Vector Dueling DQN
- **Directory**: `agents/agent1_vector_dqn/`
- **State Space ($s_t \in \mathbb{R}^{134}$)**: Continuous kinematic state (`paddle_x`, `ball_x`, `ball_y`, `ball_vx`, `ball_vy`, `remaining_bricks_ratio`) concatenated with a flattened 128-element grid of all brick health states.
- **Architecture**: Two-stream Dueling MLP. Kinematic features pass through a dense stream ($6 \to 128 \to 128$), while brick states pass through a brick stream ($128 \to 64 \to 32$), fused before Dueling Value and Advantage heads.
- **Role & Characteristics**: Fast, lightweight baseline agent. Excelled at paddle tracking and ball survival, demonstrating that flat MLPs can sustain long rallies even without spatial convolutional filters.

### 2. Agent 2: Compact Spatial Grid CNN DQN
- **Directory**: `agents/agent2_cnn_dqn/`
- **State Space**: 3-channel $16 \times 16$ spatial tensor (`Channel 0: Brick layout`, `Channel 1: Paddle position`, `Channel 2: Ball position`) plus 6 auxiliary continuous kinematic values.
- **Architecture**: 3-stage 2D Convolutional network ($3 \times 3$ kernels, MaxPool) fused with a 2-layer kinematic dense stream into Dueling Q-heads.
- **Role & Characteristics**: Highly efficient spatial learner. Successfully mastered directional paddle rebounds and achieved **100% stage clearance (all 88 bricks cleared)** in episode 535.

### 3. Agent 3: Distributional Rainbow DQN (C51 + PER + Multi-Step)
- **Directory**: `agents/agent3_rainbow_dqn/`
- **Algorithm Enhancements**:
  - **Distributional RL (C51)**: Instead of estimating scalar $Q(s, a)$, models the probability distribution of returns over 51 discrete atoms across $[-150.0, 1000.0]$.
  - **Prioritized Experience Replay (PER)**: Proportional prioritization with SumTree for efficient $O(\log N)$ updates of high TD-error transitions.
  - **Multi-Step Returns ($n$-step = 3)**: Accelerates temporal credit assignment from delayed brick hits back to paddle deflections.
  - **NoisyNet Exploration**: Factorized Gaussian parameter noise replacing heuristic $\epsilon$-greedy schedules.
- **Role & Characteristics**: Master-level agent. Consistently carves side tunnels into the top wall, traps the ball behind the brick formation for autonomous multi-brick clears, and plays with maximum time efficiency.

---

## Agent Comparison Summary

| Metric / Feature | Agent 1 (Vector DQN) | Agent 2 (CNN DQN) | Agent 3 (Rainbow DQN) |
| :--- | :--- | :--- | :--- |
| **Observation Representation** | 134-Dim Flat Vector | $3 \times 16 \times 16$ Grid + Kinematics | $3 \times 16 \times 16$ Grid + Kinematics |
| **Network Type** | Two-Stream Dueling MLP | 3-Layer Conv2D + Dueling MLP | C51 Distributional Conv2D + NoisyNet |
| **Action Space** | 3 (`LEFT`, `STAY`, `RIGHT`) | 3 (`LEFT`, `STAY`, `RIGHT`) | 3 (`LEFT`, `STAY`, `RIGHT`) |
| **Exploration Strategy** | Linear $\epsilon$-Greedy | Linear $\epsilon$-Greedy | Parametric NoisyNet ($\sigma=0.5$) |
| **Replay Mechanism** | Uniform Circular Buffer | Uniform Circular Buffer | Proportional Prioritized (PER, $\alpha=0.6, \beta=0.4$) |
| **Returns Formulation** | 1-Step TD Target | 1-Step TD Target | 3-Step Bellman Distributional Projection |
| **Peak Performance** | Sustained Ball Survival | 88/88 Bricks Cleared (Ep 535) | Superhuman Tunneling & Rapid Clearance |
| **In-Browser Inference** | `dqnInference.js` (Pure JS) | `dqnInference.js` (Conv2D in JS) | `dqnInference.js` (C51 Softmax in JS) |

---

## Features

- **Space Invader Army Layout ($8 \times 16$)**: 88 breakable bricks (Cyan, Emerald, Amber, Reinforced Pink) with steel unbreakable boundary accents.
- **Continuous Collision Detection (CCD)**: 4 micro-substeps per frame eliminating tunneling and corner grazing artifacts.
- **Exact JavaScript Simulation Parity**:
  - Python RL training scripts connect directly to the real JavaScript environment engine (`frontend/src/engine/rlEnvironment.js`) via a high-speed Node.js IPC bridge (`training/node_env_runner.js`).
  - Zero divergence in physics, floating-point geometry, boundary reflections, or rewards.
- **Live In-Browser Neural Inference**:
  - Pure JavaScript matrix, 2D convolution, and C51 distributional inference engine (`frontend/src/engine/dqnInference.js`) executes trained weights in real time without external Python or WebAssembly runtimes.
- **Interactive Telemetry & Real-Time Visualization**:
  - Live observation tensors, Q-value action distributions, cumulative reward trajectory charts, and substepped particle effects.
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
│   │   │   ├── dqnInference.js      # Pure JS in-browser Conv2D, MaxPool, and C51 inference
│   │   │   ├── levels.js            # Space Invader Army (8x16) layout & brick definitions
│   │   │   ├── particles.js         # Particle emitter, screen shake, floating scores
│   │   │   └── audio.js             # Web Audio API procedural sound synthesizer
│   │   ├── assets/
│   │   │   ├── model_weights_vector.json  # Exported Agent 1 weights
│   │   │   ├── model_weights_cnn.json     # Exported Agent 2 weights (88/88 Cleared)
│   │   │   └── model_weights_rainbow.json # Exported Agent 3 Rainbow weights
│   │   ├── App.jsx                  # Main application orchestrator & 4-way mode switcher
│   │   └── index.css                # Custom glassmorphism & neon styles
│   └── package.json
│
├── agents/                          # Self-Contained Agent Workspaces
│   ├── agent1_vector_dqn/           # Agent 1: 134D Vector Dueling MLP
│   │   ├── model.py, train.py, export.py
│   │   ├── checkpoints/ (evaluation_report.txt, model_weights_vector.json)
│   │   └── logs/
│   ├── agent2_cnn_dqn/              # Agent 2: Compact Spatial Grid CNN (88/88 Winner)
│   │   ├── model.py, train.py, export.py, analyze_log.py, compare_models.py
│   │   ├── checkpoints/ (evaluation_report.txt, model_weights_cnn.json)
│   │   └── logs/ (training_log.txt - 1500 episodes)
│   └── agent3_rainbow_dqn/          # Agent 3: Distributional Rainbow DQN (C51 + PER)
│       ├── model.py, train.py, export.py, prioritized_replay.py
│       ├── checkpoints/ (model_weights_rainbow.json)
│       └── logs/ (training_log.txt - 850 episodes)
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

## Quickstart Guide

### 1. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/Ajai-R/Reinforcement-Learning-Block-Breaker.git
cd Reinforcement-Learning-Block-Breaker

# Install training dependencies
pip install -r requirements.txt
```

### 2. Training Any Agent
Run training directly in any agent directory:

```bash
# Train Agent 1 (Vector Dueling DQN)
python agents/agent1_vector_dqn/train.py --episodes 1000

# Train Agent 2 (Spatial Grid CNN DQN)
python agents/agent2_cnn_dqn/train.py --episodes 1500

# Train Agent 3 (Distributional Rainbow DQN)
python agents/agent3_rainbow_dqn/train.py --episodes 1000
```

### 3. Exporting Weights to Web App
After training, extract model parameters into web-ready JSON format:
```bash
# Agent 1
python agents/agent1_vector_dqn/export.py

# Agent 2
python agents/agent2_cnn_dqn/export.py

# Agent 3
python agents/agent3_rainbow_dqn/export.py
```

### 4. Running the Web Studio & Interactive UI
```bash
cd frontend
npm install
npm run dev
```
Open **[http://localhost:5173](http://localhost:5173)** in your browser. You can select between **Human Play**, **Agent 1 (Vector DQN)**, **Agent 2 (CNN DQN)**, and **Agent 3 (Rainbow DQN)** in real time from the top navigation bar.
