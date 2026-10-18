// High-Speed Node.js IPC Runner for Breakout RL Environment
// Directly imports the EXACT JavaScript BreakoutEnv simulation to ensure 100% parity with React.

import readline from 'readline';
import { BreakoutEnv, ACTIONS } from '../frontend/src/engine/rlEnvironment.js';

const env = new BreakoutEnv({ trainingMode: true });

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false,
});

rl.on('line', (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;

  try {
    const req = JSON.parse(trimmed);
    const cmd = req.cmd;

    if (cmd === 'reset') {
      const maxSteps = req.max_steps || 8000;
      const fullReset = req.full_reset !== undefined ? req.full_reset : true;
      const obs = env.reset(fullReset, { maxSteps });
      const res = {
        obs: obs.vector,
        info: {
          score: env.score,
          breakable_count: env.breakableCount,
          broken_count: env.brokenCount,
          rows: env.rows,
          cols: env.cols,
          max_steps: env.maxSteps,
        },
      };
      if (req.render_frame) {
        res.frame = Buffer.from(env.renderFrameGrayscale(84, 84)).toString('base64');
      }
      process.stdout.write(JSON.stringify(res) + '\n');
    } else if (cmd === 'step') {
      const action = req.action !== undefined ? req.action : ACTIONS.STAY;
      const frameSkip = req.frame_skip || 1;
      let totalReward = 0;
      let terminated = false;
      let truncated = false;
      let lastObs = null;

      for (let i = 0; i < frameSkip; i++) {
        const stepRes = env.step(action);
        totalReward += stepRes.reward;
        lastObs = stepRes.obs;
        if (stepRes.terminated) terminated = true;
        if (stepRes.truncated) truncated = true;
        if (terminated || truncated) break;
      }

      const res = {
        obs: lastObs ? lastObs.vector : env.getObservation().vector,
        reward: totalReward,
        terminated,
        truncated,
        info: {
          score: env.score,
          steps: env.steps,
          episode_reward: env.episodeReward,
          broken_count: env.brokenCount,
          breakable_count: env.breakableCount,
          bricks_left: env.breakableCount - env.brokenCount,
        },
      };
      if (req.render_frame) {
        res.frame = Buffer.from(env.renderFrameGrayscale(84, 84)).toString('base64');
      }
      process.stdout.write(JSON.stringify(res) + '\n');
    } else if (cmd === 'close') {
      process.exit(0);
    }
  } catch (err) {
    const errRes = { error: err.message };
    process.stdout.write(JSON.stringify(errRes) + '\n');
  }
});
