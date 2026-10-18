import React, { useState, useEffect, useRef, useCallback } from 'react';
import { BreakoutEnv, ACTIONS } from './engine/rlEnvironment';
import { LEVEL_PRESETS } from './engine/levels';
import { sound } from './engine/audio';
import { DQNInference } from './engine/dqnInference';
import { Header } from './components/Header';
import { GameBoard } from './components/GameBoard';
import { RLTelemetryPanel } from './components/RLTelemetryPanel';
import { RewardChart } from './components/RewardChart';
import { RulesModal } from './components/RulesModal';
import defaultModelWeights from './assets/model_weights.json';
import vectorModelWeights from './assets/model_weights_vector.json';
import cnnModelWeights from './assets/model_weights_cnn.json';
import rainbowModelWeights from './assets/model_weights_rainbow.json';
import {
  Play,
  Pause,
  RotateCcw,
  User,
  Bot,
  Upload,
  Sparkles,
  CheckCircle,
  Brain,
  Zap,
  Target,
  Layers,
  FlaskConical,
} from 'lucide-react';

const FIXED_DT = 1.0 / 60.0; // Standardized fixed 60Hz physics timestep matching training

export const AI_MODELS = {
  human: {
    id: 'human',
    name: 'Human Player',
    tag: 'Manual A / D',
    color: 'badge-cyan',
    desc: 'Keyboard controls with real-time RL state telemetry',
  },
  agent1_vector: {
    id: 'agent1_vector',
    name: 'Agent 1: Vector-DQN',
    tag: '134D MLP',
    color: 'badge-cyan',
    weights: vectorModelWeights,
    desc: 'Two-Stream Dueling MLP trained on 134 continuous features',
  },
  agent2_cnn: {
    id: 'agent2_cnn',
    name: 'Agent 2: CNN-DQN ⭐',
    tag: 'Grid Conv (88/88 Cleared)',
    color: 'badge-pink',
    weights: cnnModelWeights,
    desc: 'Compact Spatial Grid CNN (Ep 535, Peak Avg 1698.07, 88 Clears Winner)',
  },
  agent3_dqn: {
    id: 'agent3_dqn',
    name: 'Agent 3: Rainbow DQN 🌈',
    tag: 'Full Rainbow (C51+PER+Noisy)',
    color: 'badge-purple',
    weights: rainbowModelWeights,
    desc: 'Full Rainbow DQN: Categorical C51 (51 Atoms) + PER (SumTree) + NoisyNet + Dueling + Double + n-Step',
  },
};

export function App() {
  // 1. Environment & Inference Instances
  const envRef = useRef(null);
  if (!envRef.current) {
    envRef.current = new BreakoutEnv({ fixedDt: FIXED_DT, maxSteps: Infinity });
  }
  const env = envRef.current;

  const dqnInferenceRef = useRef(null);
  if (!dqnInferenceRef.current) {
    dqnInferenceRef.current = new DQNInference();
  }
  const dqnInference = dqnInferenceRef.current;

  // 2. State & Control Hooks
  const [playerMode, setPlayerMode] = useState('agent2_cnn'); // Default to winning CNN model
  const [isModelTrained, setIsModelTrained] = useState(false);
  const [modelMeta, setModelMeta] = useState(null);
  const [isPaused, setIsPaused] = useState(false);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [selectedLevel, setSelectedLevel] = useState('invader');
  const [isMuted, setIsMuted] = useState(false);
  const [fps, setFps] = useState(60);

  // Modals
  const [isRulesOpen, setIsRulesOpen] = useState(false);

  // HUD & Telemetry State
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(0);
  const [combo, setCombo] = useState(0);
  const [steps, setSteps] = useState(0);
  const [episodeReward, setEpisodeReward] = useState(0);
  const [lastAction, setLastAction] = useState(ACTIONS.STAY);
  const [aiQValues, setAiQValues] = useState([0, 0, 0]);
  const [obs, setObs] = useState(null);
  const [rewardHistory, setRewardHistory] = useState([]);
  const [rewardEvents, setRewardEvents] = useState([]);
  const [isTerminated, setIsTerminated] = useState(false);

  // Active keyboard keys for human control
  const keysRef = useRef({ left: false, right: false });
  const fileInputRef = useRef(null);

  // Fixed Timestep Accumulator & FPS calculation refs
  const accumulatorRef = useRef(0);
  const frameCountRef = useRef(0);
  const lastFpsTimeRef = useRef(performance.now());
  const lastFrameTimeRef = useRef(performance.now());

  // Function to load model weights based on selected mode
  const applyModelWeights = useCallback(
    (modeId) => {
      const modelDef = AI_MODELS[modeId];
      if (modelDef && modelDef.weights && modelDef.weights.weights) {
        const success = dqnInference.loadWeights(modelDef.weights);
        if (success) {
          setIsModelTrained(true);
          setModelMeta(modelDef.weights.metadata || {});
          console.log(`[App] Loaded weights for ${modelDef.name}`);
        }
      } else if (modeId === 'human') {
        setIsModelTrained(false);
      }
    },
    [dqnInference]
  );

  // Load initial model weights on mount
  useEffect(() => {
    applyModelWeights(playerMode);
  }, [applyModelWeights, playerMode]);

  // Mode Selection Handler (Human, Agent 1, Agent 2, Agent 3)
  const handleSelectMode = useCallback(
    (newMode) => {
      setPlayerMode(newMode);
      applyModelWeights(newMode);
      if (newMode !== 'human' && !env.ballLaunched) {
        env.launchBall();
      }
    },
    [applyModelWeights, env]
  );

  // Initialize observation
  useEffect(() => {
    setObs(env.getObservation());
  }, [env]);

  // Restart / Reset Episode
  const handleRestart = useCallback(() => {
    const autoLaunch = playerMode !== 'human';
    const initialObs = env.reset(true, { autoLaunch, maxSteps: Infinity });
    setObs(initialObs);
    setScore(0);
    setCombo(0);
    setSteps(0);
    setEpisodeReward(0);
    setLastAction(ACTIONS.STAY);
    setRewardHistory([{ step: 0, reward: 0 }]);
    setRewardEvents([]);
    setIsTerminated(false);
  }, [env, playerMode]);

  // Layout / Environment Selection Handler (Trained vs Generalization Test)
  const handleSelectLevel = useCallback(
    (levelKey) => {
      setSelectedLevel(levelKey);
      env.setLevel(levelKey);
      const autoLaunch = playerMode !== 'human';
      const initialObs = env.reset(true, { autoLaunch, maxSteps: Infinity });
      setObs(initialObs);
      setScore(0);
      setCombo(0);
      setSteps(0);
      setEpisodeReward(0);
      setLastAction(ACTIONS.STAY);
      setRewardHistory([{ step: 0, reward: 0 }]);
      setRewardEvents([]);
      setIsTerminated(false);
    },
    [env, playerMode]
  );

  // Toggle Human <-> AI (switches to Agent 2 CNN if human)
  const handleTogglePlayerMode = useCallback(() => {
    setPlayerMode((prev) => {
      const next = prev === 'human' ? 'agent2_cnn' : 'human';
      handleSelectMode(next);
      return next;
    });
  }, [handleSelectMode]);

  // Handle Custom Model Weights JSON Upload
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const json = JSON.parse(event.target.result);
        const success = dqnInference.loadWeights(json);
        if (success) {
          setIsModelTrained(true);
          setModelMeta(json.metadata || {});
          if (playerMode === 'human') setPlayerMode('agent2_cnn');
          if (!env.ballLaunched) env.launchBall();
          alert(`Successfully loaded custom weights: ${json.format || 'Custom Model'}`);
        } else {
          alert('Invalid model weights JSON format.');
        }
      } catch (err) {
        alert('Failed to parse JSON file: ' + err.message);
      }
    };
    reader.readAsText(file);
  };

  // Sound Toggle
  const handleToggleMute = () => {
    const isNowEnabled = sound.toggleSound();
    setIsMuted(!isNowEnabled);
  };

  // Keyboard Event Listeners for Human Control
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.repeat) return;
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        keysRef.current.left = true;
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        keysRef.current.right = true;
      } else if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        if (isTerminated) {
          handleRestart();
        } else if (!env.ballLaunched) {
          env.launchBall();
        } else {
          setIsPaused((prev) => !prev);
        }
      } else if (e.key === 'r' || e.key === 'R') {
        handleRestart();
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        keysRef.current.left = false;
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        keysRef.current.right = false;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
  }, [env, isTerminated, handleRestart]);

  // Deterministic Fixed Timestep Simulation Step
  const executeSimulationStep = useCallback(
    (dt) => {
      if (isTerminated) return;

      let action = ACTIONS.STAY;

      if (playerMode !== 'human') {
        env.trainingMode = false;
        // Auto launch ball if waiting
        if (!env.ballLaunched) {
          env.launchBall();
        }
        // AI Inference Forward Pass (Vector-DQN, Spatial CNN, or DeepMind Nature DQN)
        const currentObs = env.getObservation();
        const pred = dqnInference.predict(currentObs.vector, env);
        action = pred.action;
        if (pred.qValues) {
          setAiQValues(pred.qValues);
        }
      } else {
        env.trainingMode = false;
        // Human Player Controls
        if (keysRef.current.left) action = ACTIONS.LEFT;
        else if (keysRef.current.right) action = ACTIONS.RIGHT;
      }

      // Step physics by exact fixedDt (1/60s)
      const { obs: nextObs, reward, terminated, truncated } = env.step(action, dt);

      setLastAction(action);
      setObs(nextObs);
      setScore(env.score);
      if (env.score > highScore) setHighScore(env.score);
      setCombo(env.combo);
      setSteps(env.steps);
      setEpisodeReward(env.episodeReward);
      setRewardEvents([...env.rewardEvents]);

      // In the UI, only terminate when the ball misses below the paddle (or all bricks are cleared)
      if (terminated) {
        setIsTerminated(true);
      }

      if (env.steps % 4 === 0) {
        setRewardHistory((prev) => {
          const next = [...prev, { step: env.steps, reward: env.episodeReward }];
          return next.length > 80 ? next.slice(-80) : next;
        });
      }
    },
    [env, isTerminated, highScore, playerMode, dqnInference]
  );

  // High-Refresh-Rate Animation Loop with Fixed Timestep Accumulator
  useEffect(() => {
    let animId;

    const loop = (currentTime) => {
      const frameDelta = (currentTime - lastFrameTimeRef.current) / 1000.0;
      lastFrameTimeRef.current = currentTime;

      // Clamp delta to prevent spiral of death on tab switch
      const clampedDelta = Math.min(frameDelta, 0.1);
      accumulatorRef.current += clampedDelta;

      // Real FPS calculation
      frameCountRef.current++;
      if (currentTime - lastFpsTimeRef.current >= 500) {
        const currentFps = Math.round((frameCountRef.current * 1000) / (currentTime - lastFpsTimeRef.current));
        setFps(currentFps);
        frameCountRef.current = 0;
        lastFpsTimeRef.current = currentTime;
      }

      // Fixed 60Hz Physics Accumulator Loop (Exact 1:1 match with Python training)
      while (accumulatorRef.current >= FIXED_DT) {
        if (!isPaused) {
          executeSimulationStep(FIXED_DT);
        }
        accumulatorRef.current -= FIXED_DT;
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isPaused, executeSimulationStep]);

  const activeModel = AI_MODELS[playerMode] || AI_MODELS.human;

  return (
    <div className="min-h-screen bg-slate-950 p-3 md:p-5 flex flex-col items-center w-full">
      <div className="w-full max-w-[1850px] flex flex-col gap-3">
        {/* Header HUD with Player Mode Switch */}
        <Header
          score={score}
          highScore={highScore}
          combo={combo}
          fps={fps}
          isMuted={isMuted}
          playerMode={playerMode}
          onTogglePlayerMode={handleTogglePlayerMode}
          onSelectMode={handleSelectMode}
          isModelTrained={isModelTrained}
          onToggleMute={handleToggleMute}
          onOpenRules={() => setIsRulesOpen(true)}
        />

        {/* Master Full-Width Split Layout */}
        <div className="flex flex-col lg:flex-row gap-5 items-start w-full">
          {/* Complete Left Hand Side: Expanded Game Screen & Controls */}
          <div className="w-full lg:w-[62%] xl:w-[65%] flex flex-col gap-3">
            {/* Play Controls Top Deck with Multi-Agent Model Switcher */}
            <div className="glass-panel p-3 flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                {/* 4-Way Segmented Model Selector */}
                <div className="flex items-center gap-1 p-1 bg-slate-950/80 rounded-xl border border-slate-800">
                  {/* Human */}
                  <button
                    onClick={() => handleSelectMode('human')}
                    className={`text-xs py-1.5 px-2.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer font-mono ${
                      playerMode === 'human'
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400 font-bold shadow-md shadow-cyan-500/20'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                    }`}
                  >
                    <User className="w-3.5 h-3.5" /> Human
                  </button>

                  {/* Agent 1: Vector-DQN */}
                  <button
                    onClick={() => handleSelectMode('agent1_vector')}
                    className={`text-xs py-1.5 px-2.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer font-mono ${
                      playerMode === 'agent1_vector'
                        ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-400 font-bold shadow-md shadow-cyan-500/20'
                        : 'text-slate-400 hover:text-cyan-300 hover:bg-slate-900/60'
                    }`}
                  >
                    <Brain className="w-3.5 h-3.5 text-cyan-400" /> Agent 1 (Vector 134D)
                  </button>

                  {/* Agent 2: Spatial CNN */}
                  <button
                    onClick={() => handleSelectMode('agent2_cnn')}
                    className={`text-xs py-1.5 px-2.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer font-mono ${
                      playerMode === 'agent2_cnn'
                        ? 'bg-pink-500/25 text-pink-300 border border-pink-400 font-bold shadow-md shadow-pink-500/20 ring-1 ring-pink-400/50'
                        : 'text-slate-400 hover:text-pink-300 hover:bg-slate-900/60'
                    }`}
                  >
                    <Zap className="w-3.5 h-3.5 text-pink-400" /> Agent 2 (CNN 88/88 ⭐)
                  </button>

                  {/* Agent 3: Rainbow DQN */}
                  <button
                    onClick={() => handleSelectMode('agent3_dqn')}
                    className={`text-xs py-1.5 px-2.5 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer font-mono ${
                      playerMode === 'agent3_dqn'
                        ? 'bg-purple-500/25 text-purple-300 border border-purple-400 font-bold shadow-md shadow-purple-500/20 ring-1 ring-purple-400/50'
                        : 'text-slate-400 hover:text-purple-300 hover:bg-slate-900/60'
                    }`}
                  >
                    <Sparkles className="w-3.5 h-3.5 text-purple-400" /> Agent 3 (Rainbow DQN 🌈)
                  </button>
                </div>

                {/* Active Model Description Badge */}
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-slate-300 bg-slate-900/90 px-2.5 py-1 rounded-lg border border-slate-800">
                    {activeModel.desc}
                  </span>
                </div>
              </div>

              {/* Upload Weights & Simulation Action Buttons */}
              <div className="flex items-center gap-2">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept=".json"
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  title="Upload exported model_weights.json from any trained model"
                  className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5 border-slate-700 hover:border-cyan-400"
                >
                  <Upload className="w-3.5 h-3.5 text-cyan-400" />
                  <span>Upload Weights</span>
                </button>

                <button
                  onClick={() => setIsPaused((prev) => !prev)}
                  className={`btn-secondary text-xs py-1.5 px-3.5 ${isPaused ? 'active' : ''}`}
                >
                  {isPaused ? <Play className="w-3.5 h-3.5 fill-current" /> : <Pause className="w-3.5 h-3.5" />}
                  <span>{isPaused ? 'Resume' : 'Pause'}</span>
                </button>

                <button
                  onClick={handleRestart}
                  title="Reset Episode (R)"
                  className="btn-primary text-xs py-1.5 px-3.5"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  <span>Restart</span>
                </button>
              </div>
            </div>

            {/* Canvas Interactive Board */}
            <GameBoard
              env={env}
              isPaused={isPaused}
              playerMode={playerMode}
              showTrajectory={showTrajectory}
              isTerminated={isTerminated}
              onRestart={handleRestart}
              onTogglePause={() => setIsPaused((prev) => !prev)}
            />

            {/* Environment Layout & Generalization Testing Deck */}
            <div className="glass-panel p-3 px-4 flex flex-wrap items-center justify-between text-xs gap-3">
              <div className="flex flex-wrap items-center gap-2.5">
                <span className="text-slate-400 font-mono flex items-center gap-1.5 font-bold">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Brick Layout:
                </span>
                <div className="flex items-center gap-1.5 p-1 bg-slate-950/80 rounded-xl border border-slate-800">
                  {Object.entries(LEVEL_PRESETS).map(([key, preset]) => {
                    const isSelected = selectedLevel === key;
                    return (
                      <button
                        key={key}
                        onClick={() => handleSelectLevel(key)}
                        className={`text-xs py-1 px-3 rounded-lg flex items-center gap-2 transition-all cursor-pointer font-mono font-bold ${
                          isSelected
                            ? preset.isTrained
                              ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-400 shadow-md shadow-cyan-500/20'
                              : 'bg-amber-500/25 text-amber-300 border border-amber-400 shadow-md shadow-amber-500/20'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
                        }`}
                        title={preset.description}
                      >
                        {preset.isTrained ? (
                          <CheckCircle className="w-3.5 h-3.5 text-cyan-400" />
                        ) : (
                          <FlaskConical className="w-3.5 h-3.5 text-amber-400" />
                        )}
                        <span>{preset.shortName}</span>
                        <span
                          className={`text-[9px] px-1.5 py-0.5 rounded font-sans uppercase font-bold border ${preset.badgeColor}`}
                        >
                          {preset.badge}
                        </span>
                      </button>
                    );
                  })}
                </div>

                {/* Explanation tag for the active layout */}
                <div className="flex items-center gap-2">
                  <span
                    className={`text-[11px] font-mono px-2.5 py-1 rounded-lg border ${
                      LEVEL_PRESETS[selectedLevel]?.isTrained
                        ? 'text-cyan-300 bg-cyan-950/40 border-cyan-800/60'
                        : 'text-amber-300 bg-amber-950/40 border-amber-800/60'
                    }`}
                  >
                    {LEVEL_PRESETS[selectedLevel]?.isTrained
                      ? '🎯 Trained Domain: Standard 88-brick configuration trained across 4,000 episodes.'
                      : '🧪 Generalization Test: 100% unseen layout. Testing zero-shot adaptability!'}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
                <span>
                  Bricks: <strong className="text-emerald-400">{env.breakableCount} breakable</strong>
                  {env.totalBricks - env.breakableCount > 0 ? ` + ${env.totalBricks - env.breakableCount} steel` : ''}
                </span>
                <span>Dim: 8×16 (128D)</span>
                <span>Physics: Fixed 60Hz</span>
              </div>
            </div>
          </div>

          {/* Complete Right Hand Side: Multi-tab Telemetry & Reward Graphs */}
          <div className="w-full lg:w-[38%] xl:w-[35%] flex flex-col gap-4">
            <RLTelemetryPanel
              obs={obs}
              lastAction={lastAction}
              rewardEvents={rewardEvents}
              episodeReward={episodeReward}
              steps={steps}
              brokenCount={env.brokenCount}
              breakableCount={env.breakableCount}
              aiQValues={aiQValues}
              playerMode={playerMode}
            />

            <RewardChart
              rewardHistory={rewardHistory}
              currentReward={episodeReward}
              steps={steps}
              score={score}
            />
          </div>
        </div>

        {/* Math & MDP Rules Modal */}
        <RulesModal
          isOpen={isRulesOpen}
          onClose={() => setIsRulesOpen(false)}
        />
      </div>
    </div>
  );
}

export default App;
