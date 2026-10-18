import React, { useMemo } from 'react';
import { TrendingUp } from 'lucide-react';

export function RewardChart({
  rewardHistory = [],
  history = [],
  currentSteps = 0,
  steps = 0,
  currentReward = 0,
  episodeReward = 0,
  score = 0,
}) {
  const dataHistory = rewardHistory && rewardHistory.length > 0 ? rewardHistory : history || [];
  const stepCount = steps || currentSteps || 0;
  const totalReward = episodeReward !== undefined && episodeReward !== 0 ? episodeReward : (currentReward || 0);

  // Generate SVG path for cumulative reward progression
  const points = useMemo(() => {
    if (!dataHistory || dataHistory.length < 2) return '';
    const maxVal = Math.max(...dataHistory.map((p) => p.reward || 0), 10);
    const minVal = Math.min(...dataHistory.map((p) => p.reward || 0), -20);
    const range = maxVal - minVal || 1;

    const width = 380;
    const height = 90;

    return dataHistory
      .map((pt, idx) => {
        const rwd = pt.reward || 0;
        const x = (idx / (dataHistory.length - 1)) * width;
        const y = height - ((rwd - minVal) / range) * (height - 10) - 5;
        return `${idx === 0 ? 'M' : 'L'} ${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [dataHistory]);

  return (
    <div className="glass-panel p-4 flex flex-col gap-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs uppercase font-bold tracking-wider text-slate-200">
            Real-Time Episode Performance
          </h3>
        </div>
        <div className="flex items-center gap-3 text-xs font-mono">
          <span className="text-slate-400">
            Steps: <strong className="text-slate-200">{stepCount}</strong>
          </span>
          <span className="text-slate-400">
            Rwd: <strong className="text-emerald-400">{Number(totalReward || 0).toFixed(1)}</strong>
          </span>
        </div>
      </div>

      {/* Mini SVG Curve */}
      <div className="w-full h-24 bg-slate-950/70 border border-slate-800/80 rounded-xl relative overflow-hidden flex items-center justify-center p-2">
        {dataHistory.length > 2 ? (
          <svg viewBox="0 0 380 90" className="w-full h-full overflow-visible">
            <defs>
              <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#00f0ff" />
                <stop offset="50%" stopColor="#38bdf8" />
                <stop offset="100%" stopColor="#10b981" />
              </linearGradient>
            </defs>
            {/* Zero line */}
            <line x1="0" y1="50" x2="380" y2="50" stroke="rgba(255,255,255,0.07)" strokeDasharray="3 3" />
            {/* Reward Path */}
            <path
              d={points}
              fill="none"
              stroke="url(#lineGrad)"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="filter drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]"
            />
          </svg>
        ) : (
          <span className="text-slate-500 text-xs font-mono">
            Accumulating reward steps...
          </span>
        )}
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-3 gap-2 text-center font-mono">
        <div className="p-2 rounded-lg bg-slate-950/40 border border-slate-800/60">
          <span className="text-[10px] text-slate-400 block mb-0.5">Score</span>
          <span className="text-sm font-bold text-cyan-300">{score || 0}</span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/40 border border-slate-800/60">
          <span className="text-[10px] text-slate-400 block mb-0.5">Avg Rwd/Step</span>
          <span className="text-sm font-bold text-emerald-400">
            {stepCount > 0 ? (totalReward / stepCount).toFixed(3) : '0.000'}
          </span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/40 border border-slate-800/60">
          <span className="text-[10px] text-slate-400 block mb-0.5">Efficiency</span>
          <span className="text-sm font-bold text-pink-400">
            {score > 0 && stepCount > 0 ? (score / (stepCount / 60)).toFixed(1) + '/s' : '0/s'}
          </span>
        </div>
      </div>
    </div>
  );
}
