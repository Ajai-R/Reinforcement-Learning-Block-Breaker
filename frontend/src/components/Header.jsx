import React from 'react';
import { Volume2, VolumeX, BookOpen, Trophy, Flame, Sparkles, Gauge, Bot, User } from 'lucide-react';

export function Header({
  score,
  highScore,
  combo,
  fps,
  isMuted,
  playerMode,
  onTogglePlayerMode,
  onSelectMode,
  isModelTrained,
  onToggleMute,
  onOpenRules,
}) {
  const isAI = playerMode !== 'human';

  const getModeBadge = () => {
    switch (playerMode) {
      case 'agent1_vector':
        return { label: 'AGENT 1 (VECTOR)', color: 'badge-cyan', desc: '134-Dim Vector Dueling MLP Policy' };
      case 'agent2_cnn':
        return { label: 'AGENT 2 (CNN 88/88)', color: 'badge-pink', desc: 'Compact Spatial Grid CNN (Stage Clear Winner)' };
      case 'agent3_dqn':
      case 'agent3_ppo':
        return { label: 'AGENT 3 (RAINBOW DQN 🌈)', color: 'badge-purple', desc: 'Full Rainbow DQN (C51 Distributional + PER + NoisyNet + Dueling Double)' };
      default:
        return { label: 'HUMAN PLAY', color: 'badge-cyan', desc: 'Human keyboard control with real-time RL telemetry' };
    }
  };

  const badge = getModeBadge();

  return (
    <header className="w-full glass-panel px-6 py-4 mb-5 flex flex-wrap items-center justify-between gap-4">
      {/* Brand Title & Player Mode Switch */}
      <div className="flex items-center gap-4">
        <div className="w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/25 border border-cyan-300/40">
          <Sparkles className="w-6 h-6 text-slate-950 font-bold" />
        </div>
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-black tracking-tight text-white font-mono">
              BLOCK BREAKER <span className="text-cyan-400 font-normal">RL STUDIO</span>
            </h1>
            <span
              className={`badge-neon text-[11px] py-1 px-3 flex items-center gap-1.5 ${badge.color} ring-2 ring-current/30 font-bold`}
            >
              {isAI ? <Bot className="w-3.5 h-3.5 animate-pulse" /> : <User className="w-3.5 h-3.5" />}
              {badge.label}
            </span>
          </div>
          <p className="text-xs text-slate-400">
            {badge.desc}
          </p>
        </div>
      </div>

      {/* Main HUD Stats */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Real-time FPS Display */}
        <div className="flex items-center gap-1.5 bg-slate-950/60 border border-slate-800 rounded-xl px-3 py-2 font-mono text-xs">
          <Gauge className="w-4 h-4 text-emerald-400" />
          <span className="text-emerald-400 font-bold">{fps} FPS</span>
        </div>

        {/* Score & High Score */}
        <div className="flex items-center gap-3 bg-slate-950/60 border border-slate-800 rounded-xl px-4 py-2">
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Score</span>
            <span className="text-xl font-black font-mono text-cyan-300">{score}</span>
          </div>
          <div className="h-7 w-[1px] bg-slate-800" />
          <div className="flex flex-col">
            <span className="text-[10px] uppercase font-bold tracking-wider text-slate-400 flex items-center gap-1">
              <Trophy className="w-3 h-3 text-amber-400" /> Best
            </span>
            <span className="text-xl font-black font-mono text-amber-400">{highScore}</span>
          </div>
        </div>

        {/* Combo Multiplier */}
        {combo > 1 && (
          <div className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-pink-500/15 border border-pink-500/30 text-pink-400 animate-pulse font-mono font-bold text-sm">
            <Flame className="w-4 h-4 fill-pink-500" />
            <span>x{combo} COMBO</span>
          </div>
        )}

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={onToggleMute}
            title={isMuted ? 'Unmute Audio' : 'Mute Audio'}
            className="p-2.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 border border-slate-700/60 text-slate-300 hover:text-cyan-300 transition-colors"
          >
            {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
          </button>
          <button
            onClick={onOpenRules}
            title="Rules & RL Framework"
            className="btn-secondary text-xs py-2 px-3"
          >
            <BookOpen className="w-4 h-4" />
            <span>Rules & Math</span>
          </button>
        </div>
      </div>
    </header>
  );
}
