import React from 'react';
import {
  Activity,
  ArrowLeft,
  Pause,
  ArrowRight,
  Layers,
  Terminal,
  Shield,
  Zap,
  Sparkles,
  AlertTriangle,
  Trophy,
  Flame,
} from 'lucide-react';
import { ACTIONS } from '../engine/rlEnvironment';

function getEventMeta(evt) {
  const types = evt.items.map((it) => it.type);
  const typeStr = types.join(' ');

  if (typeStr.includes('STAGE_CLEAR')) {
    return {
      label: 'Stage Cleared!',
      icon: Trophy,
      badgeColor: 'text-amber-300 bg-amber-500/10 border-amber-500/30 shadow-[0_0_10px_rgba(245,158,11,0.2)]',
      iconColor: 'text-amber-400 bg-amber-950/60 border-amber-500/40',
    };
  }
  if (typeStr.includes('MILESTONE')) {
    const m = evt.items.find((it) => it.type.includes('MILESTONE'))?.type || 'Milestone';
    return {
      label: m,
      icon: Sparkles,
      badgeColor: 'text-yellow-300 bg-yellow-500/10 border-yellow-500/30',
      iconColor: 'text-yellow-400 bg-yellow-950/60 border-yellow-500/40',
    };
  }
  if (typeStr.includes('BALL_MISSED')) {
    return {
      label: 'Ball Missed',
      icon: AlertTriangle,
      badgeColor: 'text-rose-400 bg-rose-500/10 border-rose-500/30',
      iconColor: 'text-rose-400 bg-rose-950/60 border-rose-500/40',
    };
  }
  if (typeStr.includes('PADDLE_CATCH')) {
    return {
      label: 'Paddle Catch',
      icon: Shield,
      badgeColor: 'text-emerald-300 bg-emerald-500/10 border-emerald-500/30',
      iconColor: 'text-emerald-400 bg-emerald-950/60 border-emerald-500/40',
    };
  }
  if (typeStr.includes('BRICK_BREAK')) {
    const breakItem = evt.items.find((it) => it.type.includes('BRICK_BREAK'));
    const isCombo = breakItem?.type.includes('(x');
    return {
      label: isCombo ? breakItem.type : 'Brick Break',
      icon: isCombo ? Flame : Zap,
      badgeColor: isCombo
        ? 'text-pink-300 bg-pink-500/10 border-pink-500/30'
        : 'text-cyan-300 bg-cyan-500/10 border-cyan-500/30',
      iconColor: isCombo
        ? 'text-pink-400 bg-pink-950/60 border-pink-500/40'
        : 'text-cyan-400 bg-cyan-950/60 border-cyan-500/40',
    };
  }
  return {
    label: evt.items[0]?.type || 'Reward Event',
    icon: Sparkles,
    badgeColor: 'text-slate-300 bg-slate-800/60 border-slate-700',
    iconColor: 'text-slate-400 bg-slate-900 border-slate-700',
  };
}

export function RLTelemetryPanel({
  obs,
  lastAction,
  rewardEvents,
  episodeReward,
  steps,
  brokenCount,
  breakableCount,
  aiQValues = [0, 0, 0],
  playerMode = 'human',
}) {
  const isAI = playerMode !== 'human';
  const isPPO = playerMode === 'agent3_ppo';
  const vector = obs?.vector || [0, 0, 0, 0, 0, 0];
  const [
    paddleX = 0,
    ballX = 0,
    ballY = 0,
    ballVx = 0,
    ballVy = 0,
    bricksLeft = 1,
  ] = vector;
  const brickLayout = obs?.brickLayout || vector.slice(6);

  return (
    <div className="flex flex-col gap-4 w-full">
      {/* 1. Observation Vector Monitor */}
      <div className="glass-panel p-4">
        <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            <h3 className="text-xs uppercase font-bold tracking-wider text-slate-200">
              RL State Observation (s_t)
            </h3>
          </div>
          <span className="badge-neon badge-cyan text-[10px]">Continuous({vector.length})</span>
        </div>

        <div className="space-y-2.5 font-mono text-xs">
          {/* Paddle X */}
          <div>
            <div className="flex justify-between text-slate-400 text-[11px] mb-1">
              <span>paddle_x (normalized)</span>
              <span className="text-cyan-300 font-bold">{paddleX.toFixed(3)}</span>
            </div>
            <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden relative">
              <div
                className="h-full bg-cyan-400 rounded-full transition-all duration-75"
                style={{ width: `${((paddleX + 1) / 2) * 100}%` }}
              />
            </div>
          </div>

          {/* Ball X & Y */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                <span>ball_x</span>
                <span className="text-cyan-300">{ballX.toFixed(3)}</span>
              </div>
              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 rounded-full"
                  style={{ width: `${((ballX + 1) / 2) * 100}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                <span>ball_y</span>
                <span className="text-cyan-300">{ballY.toFixed(3)}</span>
              </div>
              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 rounded-full"
                  style={{ width: `${((ballY + 1) / 2) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Ball Velocities */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                <span>ball_vx</span>
                <span className={ballVx >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                  {ballVx.toFixed(3)}
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${((ballVx + 1) / 2) * 100}%` }}
                />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-slate-400 text-[11px] mb-1">
                <span>ball_vy</span>
                <span className={ballVy >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
                  {ballVy.toFixed(3)}
                </span>
              </div>
              <div className="w-full h-1.5 bg-slate-900 rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full"
                  style={{ width: `${((ballVy + 1) / 2) * 100}%` }}
                />
              </div>
            </div>
          </div>

          {/* Bricks remaining ratio */}
          <div>
            <div className="flex items-center justify-between p-2 rounded-lg bg-slate-950/50 border border-slate-800/80">
              <span className="text-[11px] text-slate-400">Bricks Remaining</span>
              <span className="text-amber-400 font-bold">{(bricksLeft * 100).toFixed(0)}%</span>
            </div>
          </div>

          {/* Spatial 8x16 Brick Layout Tensor Visualizer */}
          {brickLayout && brickLayout.length > 0 && (
            <div className="pt-2 border-t border-slate-800/80">
              <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1.5">
                <span className="text-slate-300">Spatial Brick Tensor (8×16)</span>
                <span className="text-cyan-400 font-mono text-[10px]">+{brickLayout.length} dims</span>
              </div>
              <div
                className="gap-1 p-2 bg-slate-950/80 rounded-lg border border-slate-800"
                style={{ display: 'grid', gridTemplateColumns: 'repeat(16, minmax(0, 1fr))' }}
              >
                {brickLayout.map((val, idx) => {
                  let bg = 'bg-slate-900/80 border border-slate-800/40';
                  if (val === -1) {
                    bg = 'bg-slate-500 shadow-[0_0_2px_rgba(100,116,139,0.8)]';
                  } else if (val === 1) {
                    bg = 'bg-cyan-400 shadow-[0_0_3px_rgba(0,240,255,0.7)]';
                  } else if (val > 0) {
                    bg = 'bg-pink-500 shadow-[0_0_3px_rgba(236,72,153,0.7)]';
                  }
                  return (
                    <div
                      key={idx}
                      className={`h-2 rounded-[1.5px] transition-colors duration-150 ${bg}`}
                      title={`Brick [${Math.floor(idx / 16)}, ${idx % 16}]: ${val}`}
                    />
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 2. Action Space & Decision Engine Indicator */}
      <div className="glass-panel p-4">
        <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-emerald-400" />
            <h3 className="text-xs uppercase font-bold tracking-wider text-slate-200">
              {isAI ? (isPPO ? 'PPO Policy π(a|s)' : 'Q-Network Values Q(s, a)') : 'Human Action a_t'}
            </h3>
          </div>
          <span className="badge-neon badge-emerald text-[10px]">
            {isAI ? (isPPO ? 'Actor Softmax' : 'Dueling Heads') : 'Keys: A / D / Space'}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {/* Action 0: LEFT */}
          <div
            className={`flex flex-col items-center justify-center p-2.5 rounded-xl border transition-all duration-150 ${
              lastAction === ACTIONS.LEFT
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/25 scale-[1.03]'
                : 'bg-slate-950/40 border-slate-800 text-slate-500'
            }`}
          >
            <ArrowLeft className="w-5 h-5 mb-1" />
            <span className="font-mono text-xs font-bold">0: LEFT</span>
            {isAI && (
              <span className="font-mono text-[10px] text-cyan-300 font-semibold mt-0.5">
                {isPPO ? `π: ${(aiQValues[0] * 100).toFixed(1)}%` : `Q: ${aiQValues[0]?.toFixed(2)}`}
              </span>
            )}
          </div>

          {/* Action 1: STAY */}
          <div
            className={`flex flex-col items-center justify-center p-2.5 rounded-xl border transition-all duration-150 ${
              lastAction === ACTIONS.STAY
                ? 'bg-amber-500/20 border-amber-400 text-amber-300 shadow-lg shadow-amber-500/25 scale-[1.03]'
                : 'bg-slate-950/40 border-slate-800 text-slate-500'
            }`}
          >
            <Pause className="w-5 h-5 mb-1" />
            <span className="font-mono text-xs font-bold">1: STAY</span>
            {isAI && (
              <span className="font-mono text-[10px] text-amber-300 font-semibold mt-0.5">
                {isPPO ? `π: ${(aiQValues[1] * 100).toFixed(1)}%` : `Q: ${aiQValues[1]?.toFixed(2)}`}
              </span>
            )}
          </div>

          {/* Action 2: RIGHT */}
          <div
            className={`flex flex-col items-center justify-center p-2.5 rounded-xl border transition-all duration-150 ${
              lastAction === ACTIONS.RIGHT
                ? 'bg-cyan-500/20 border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/25 scale-[1.03]'
                : 'bg-slate-950/40 border-slate-800 text-slate-500'
            }`}
          >
            <ArrowRight className="w-5 h-5 mb-1" />
            <span className="font-mono text-xs font-bold">2: RIGHT</span>
            {isAI && (
              <span className="font-mono text-[10px] text-cyan-300 font-semibold mt-0.5">
                {isPPO ? `π: ${(aiQValues[2] * 100).toFixed(1)}%` : `Q: ${aiQValues[2]?.toFixed(2)}`}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 3. Live Step Reward Stream Log */}
      <div className="glass-panel p-4 flex flex-col h-64">
        <div className="flex items-center justify-between mb-2.5 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
            </span>
            <h3 className="text-xs uppercase font-bold tracking-wider text-slate-200">
              Live Reward Stream
            </h3>
          </div>
          <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-slate-900/90 border border-slate-800">
            <span className="text-[10px] text-slate-400 font-mono">r_t Total:</span>
            <strong className="text-xs font-mono text-cyan-300">{(episodeReward || 0).toFixed(2)}</strong>
          </div>
        </div>

        {/* Live Category Summary Badges */}
        <div className="grid grid-cols-2 gap-2 mb-2.5">
          <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-950/60 border border-emerald-500/20">
            <div className="flex items-center gap-1.5 text-[11px] text-emerald-400 font-medium">
              <Shield className="w-3.5 h-3.5" />
              <span>Catches</span>
            </div>
            <span className="font-mono text-xs font-bold text-emerald-300">
              {rewardEvents?.filter((e) => e.items.some((i) => i.type.includes('PADDLE_CATCH'))).length || 0}
            </span>
          </div>
          <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-950/60 border border-cyan-500/20">
            <div className="flex items-center gap-1.5 text-[11px] text-cyan-400 font-medium">
              <Zap className="w-3.5 h-3.5" />
              <span>Breaks</span>
            </div>
            <span className="font-mono text-xs font-bold text-cyan-300">
              {rewardEvents?.filter((e) => e.items.some((i) => i.type.includes('BRICK_BREAK'))).length || 0}
            </span>
          </div>
        </div>

        {/* Event Cards Ticker */}
        <div className="flex-1 overflow-y-auto space-y-1.5 pr-1 font-mono text-xs scrollbar-thin scrollbar-thumb-slate-800">
          {rewardEvents && rewardEvents.length > 0 ? (
            rewardEvents.map((evt, idx) => {
              const meta = getEventMeta(evt);
              const Icon = meta.icon;
              return (
                <div
                  key={`${evt.step}-${idx}`}
                  className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 transition-all shadow-sm"
                >
                  <div className="flex items-center gap-2">
                    <div className={`w-5 h-5 rounded flex items-center justify-center border ${meta.iconColor}`}>
                      <Icon className="w-3 h-3" />
                    </div>
                    <div className="flex flex-col">
                      <span className="text-[11px] font-semibold text-slate-200 leading-tight">
                        {meta.label}
                      </span>
                      <span className="text-[9px] font-mono text-slate-500">
                        Step #{evt.step}
                      </span>
                    </div>
                  </div>
                  <div className={`px-2 py-0.5 rounded border text-[11px] font-mono font-bold ${meta.badgeColor}`}>
                    {evt.total > 0 ? `+${evt.total.toFixed(2)}` : evt.total.toFixed(2)}
                  </div>
                </div>
              );
            })
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-500 text-xs py-4">
              <Activity className="w-5 h-5 mb-1.5 text-slate-600 animate-pulse" />
              <span className="text-[11px] text-slate-400">Listening to RL transitions...</span>
              <span className="text-[9px] text-slate-600 mt-0.5">Catches, breaks & milestones stream here live</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
