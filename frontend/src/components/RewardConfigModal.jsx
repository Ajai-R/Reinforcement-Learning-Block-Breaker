import React from 'react';
import { X, Sliders, RotateCcw, Check } from 'lucide-react';
import { DEFAULT_REWARDS } from '../engine/rlEnvironment';

export function RewardConfigModal({ isOpen, onClose, weights, onChangeWeights }) {
  if (!isOpen) return null;

  const handleSlider = (key, val) => {
    onChangeWeights({ ...weights, [key]: parseFloat(val) });
  };

  const applyPreset = (presetName) => {
    if (presetName === 'default') {
      onChangeWeights({ ...DEFAULT_REWARDS });
    } else if (presetName === 'survival') {
      onChangeWeights({
        ...DEFAULT_REWARDS,
        stepSurvival: 0.05,
        paddleCatch: 10.0,
        brickBreak: 5.0,
        ballDropPenalty: -20.0,
      });
    } else if (presetName === 'smasher') {
      onChangeWeights({
        ...DEFAULT_REWARDS,
        stepSurvival: 0.0,
        paddleCatch: 2.0,
        brickBreak: 25.0,
        stageClear: 200.0,
      });
    } else if (presetName === 'dense') {
      onChangeWeights({
        ...DEFAULT_REWARDS,
        proximityShaping: 0.02,
      });
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="glass-panel-elevated p-6 w-full max-w-xl text-white shadow-2xl relative"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-700/60 pb-3 mb-5">
          <div className="flex items-center gap-2.5">
            <Sliders className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold font-mono tracking-wide">
              Reward Function Tuning (R_t)
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Quick Presets */}
        <div className="mb-5">
          <label className="text-xs uppercase font-bold text-slate-400 tracking-wider block mb-2">
            Reward Presets:
          </label>
          <div className="grid grid-cols-4 gap-2">
            <button
              onClick={() => applyPreset('default')}
              className="btn-secondary text-xs py-1.5 px-2"
            >
              Balanced
            </button>
            <button
              onClick={() => applyPreset('survival')}
              className="btn-secondary text-xs py-1.5 px-2"
            >
              Survival Max
            </button>
            <button
              onClick={() => applyPreset('smasher')}
              className="btn-secondary text-xs py-1.5 px-2"
            >
              High Break
            </button>
            <button
              onClick={() => applyPreset('dense')}
              className="btn-secondary text-xs py-1.5 px-2"
            >
              Dense Shaping
            </button>
          </div>
        </div>

        {/* Sliders Grid */}
        <div className="space-y-4 font-mono text-xs">
          {/* Step Survival */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Step Survival Reward (per frame)</span>
              <span className="text-cyan-300 font-bold">+{weights.stepSurvival.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="0.10"
              step="0.005"
              value={weights.stepSurvival}
              onChange={(e) => handleSlider('stepSurvival', e.target.value)}
            />
          </div>

          {/* Paddle Catch */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Paddle Interception / Catch</span>
              <span className="text-emerald-400 font-bold">+{weights.paddleCatch.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="20"
              step="0.5"
              value={weights.paddleCatch}
              onChange={(e) => handleSlider('paddleCatch', e.target.value)}
            />
          </div>

          {/* Brick Break */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Brick Destroyed Reward</span>
              <span className="text-pink-400 font-bold">+{weights.brickBreak.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="1"
              max="30"
              step="1"
              value={weights.brickBreak}
              onChange={(e) => handleSlider('brickBreak', e.target.value)}
            />
          </div>

          {/* Stage Clear */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Stage Clear Bonus (Terminal Win)</span>
              <span className="text-amber-400 font-bold">+{weights.stageClear.toFixed(0)}</span>
            </div>
            <input
              type="range"
              min="20"
              max="300"
              step="10"
              value={weights.stageClear}
              onChange={(e) => handleSlider('stageClear', e.target.value)}
            />
          </div>

          {/* Ball Missed Penalty */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Ball Missed / Drop Penalty (Terminal)</span>
              <span className="text-rose-400 font-bold">{weights.ballDropPenalty.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min="-40"
              max="0"
              step="1"
              value={weights.ballDropPenalty}
              onChange={(e) => handleSlider('ballDropPenalty', e.target.value)}
            />
          </div>

          {/* Proximity Shaping */}
          <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800">
            <div className="flex justify-between mb-1.5">
              <span className="text-slate-300">Proximity Shaping Gradient (Ball Descent)</span>
              <span className="text-violet-400 font-bold">+{weights.proximityShaping.toFixed(3)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="0.05"
              step="0.005"
              value={weights.proximityShaping}
              onChange={(e) => handleSlider('proximityShaping', e.target.value)}
            />
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between mt-6 pt-3 border-t border-slate-800">
          <button
            onClick={() => applyPreset('default')}
            className="btn-secondary text-xs py-2 px-3 flex items-center gap-1.5"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>
          <button onClick={onClose} className="btn-primary text-xs py-2 px-4">
            <Check className="w-4 h-4" />
            <span>Apply Weights</span>
          </button>
        </div>
      </div>
    </div>
  );
}
