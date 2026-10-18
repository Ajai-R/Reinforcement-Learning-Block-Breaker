import React from 'react';
import { X, BookOpen, Target, Cpu, Award, Zap, Code2 } from 'lucide-react';

export function RulesModal({ isOpen, onClose }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="glass-panel-elevated p-6 w-full max-w-2xl text-white shadow-2xl relative max-h-[85vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-700/60 pb-3 mb-4 sticky top-0 bg-slate-900/90 backdrop-blur-md z-10">
          <div className="flex items-center gap-2.5">
            <BookOpen className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold font-mono tracking-wide">
              Game Rules & RL Mathematical Framework
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-6 text-sm text-slate-300">
          {/* Section 1: Classic Game Rules */}
          <div>
            <div className="flex items-center gap-2 text-cyan-400 font-bold mb-2">
              <Target className="w-4 h-4" />
              <h3 className="uppercase text-xs tracking-wider">1. Authentic Block Breaker Physics & Rules</h3>
            </div>
            <ul className="space-y-2 list-disc list-inside text-xs leading-relaxed text-slate-300 bg-slate-950/50 p-3.5 rounded-xl border border-slate-800">
              <li>
                <strong className="text-white">Angular Paddle Deflection:</strong> The ball reflection angle depends on where it hits the paddle. Hitting the center bounces the ball straight up, while edges cause sharp angle deflections ($up\ to\ \pm 65^\circ$).
              </li>
              <li>
                <strong className="text-white">Brick Types:</strong>
                <ul className="pl-5 pt-1 space-y-1">
                  <li><span className="text-cyan-300 font-bold">1-Hit Normal (Cyan/Green/Amber):</span> Break in 1 hit, worth 10-30 points.</li>
                  <li><span className="text-pink-400 font-bold">2-Hit Reinforced (Magenta):</span> Cracks on first hit, shatters on second hit (50 pts).</li>
                  <li><span className="text-slate-400 font-bold">Steel Unbreakable (Grey):</span> Reflects the ball without breaking.</li>
                </ul>
              </li>
              <li>
                <strong className="text-white">Combos:</strong> Destroying consecutive bricks without hitting the paddle accumulates a multiplier (up to $5\times$).
              </li>
              <li>
                <strong className="text-white">Episodic Objective:</strong> Prevent the ball from dropping below the paddle while clearing all breakable bricks with maximum reward efficiency.
              </li>
            </ul>
          </div>

          {/* Section 2: RL Formulation */}
          <div>
            <div className="flex items-center gap-2 text-emerald-400 font-bold mb-2">
              <Cpu className="w-4 h-4" />
              <h3 className="uppercase text-xs tracking-wider">2. Markov Decision Process (MDP) Formulation</h3>
            </div>
            <div className="bg-slate-950/50 p-3.5 rounded-xl border border-slate-800 space-y-3 font-mono text-xs">
              <div>
                <span className="text-cyan-300 font-bold block mb-1">State Observation Space S_t:</span>
                <p className="text-slate-300 text-[11px] leading-relaxed">
                  Continuous 134-dimensional vector (6 kinematics + 128 spatial brick layout cells):
                </p>
                <div className="p-2 rounded bg-slate-900 border border-slate-800 mt-1 text-slate-200">
                  <code>s_t = [paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_left_ratio, ...brick_grid_8x16]</code>
                </div>
              </div>

              <div>
                <span className="text-emerald-400 font-bold block mb-1">Action Space A_t:</span>
                <div className="grid grid-cols-3 gap-2 text-center text-[11px] mt-1">
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <strong>0: LEFT</strong> (-paddle_speed)
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <strong>1: STAY</strong> (0 velocity)
                  </div>
                  <div className="p-1.5 rounded bg-slate-900 border border-slate-800">
                    <strong>2: RIGHT</strong> (+paddle_speed)
                  </div>
                </div>
              </div>

              <div>
                <span className="text-pink-400 font-bold block mb-1">Reward Function R_t:</span>
                <div className="p-2 rounded bg-slate-900 border border-slate-800 text-[11px] space-y-1">
                  <div>+0.001 : Step survival reward (incentivizes keeping the ball alive)</div>
                  <div>+1.000 : Successful paddle interception catch</div>
                  <div>+10.00 : Brick destroyed (scaled by 10 * [1 + (combo - 1) * 0.1])</div>
                  <div>+3.000 : Reinforced brick cracked</div>
                  <div>+100.0 : Terminal stage cleared bonus</div>
                  <div>-10.00 : Terminal ball drop / miss penalty</div>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: Next Steps - DQN Training */}
          <div>
            <div className="flex items-center gap-2 text-amber-400 font-bold mb-2">
              <Code2 className="w-4 h-4" />
              <h3 className="uppercase text-xs tracking-wider">3. Training Deep Q-Networks (DQN)</h3>
            </div>
            <div className="bg-slate-950/50 p-3.5 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-2">
              <p>
                The environment runs a high-precision physics simulation with continuous collision detection and continuous state vector observations.
              </p>
              <p>
                In the next phase, we can train a DQN / Double DQN model to play the game:
              </p>
              <div className="p-2 bg-slate-900 rounded font-mono text-[11px] text-emerald-400">
                Q(s, a) ← Q(s, a) + α [ r + γ max Q(s', a') - Q(s, a) ]
              </div>
              <p className="text-slate-400 text-[11px]">
                Features like Replay Buffers, $\epsilon$-greedy exploration decay, and Target Networks ensure robust convergence.
              </p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end mt-6 pt-3 border-t border-slate-800">
          <button onClick={onClose} className="btn-primary text-xs py-2 px-4">
            Got It, Let's Play!
          </button>
        </div>
      </div>
    </div>
  );
}
