import React, { useRef, useEffect, useCallback } from 'react';
import { BRICK_TYPES, BRICK_COLORS } from '../engine/levels';
import { Play, RotateCcw, Award, AlertCircle } from 'lucide-react';
import confetti from 'canvas-confetti';

export function GameBoard({
  env,
  isPaused,
  playerMode = 'human',
  showTrajectory,
  isTerminated,
  onRestart,
  onTogglePause,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const trailRef = useRef([]);
  const canvasDimRef = useRef({ width: 700, height: 600 });

  // Coordinate transformation: World coordinates (0-10) to Canvas pixels
  const worldToCanvas = useCallback((wx, wy, width, height) => {
    const scaleX = width / env.width;
    const scaleY = height / env.height;
    const cx = wx * scaleX;
    const cy = height - wy * scaleY;
    return { cx, cy, scaleX, scaleY };
  }, [env.width, env.height]);

  // Main Canvas Render Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId;

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvasDimRef.current.width || 700;
      const height = canvasDimRef.current.height || 600;
      if (width <= 0 || height <= 0) return;

      // Reset transform with DPR
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Handle screen shake
      const shake = env.particles.getShakeOffset();

      ctx.save();
      ctx.clearRect(0, 0, width, height);
      ctx.translate(shake.x, shake.y);

      // 1. Draw Arcade Grid Background
      ctx.fillStyle = '#080c16';
      ctx.fillRect(0, 0, width, height);

      // Background grid lines
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.05)';
      ctx.lineWidth = 1;
      const gridSize = Math.max(24, Math.floor(width / 24));
      for (let x = 0; x < width; x += gridSize) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += gridSize) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Bottom Danger Zone
      const dangerPt = worldToCanvas(0, env.paddleY - 0.25, width, height);
      const dangerY = isFinite(dangerPt.cy) ? dangerPt.cy : height - 30;
      try {
        const dangerGrad = ctx.createLinearGradient(0, Math.max(0, dangerY), 0, height);
        dangerGrad.addColorStop(0, 'rgba(244, 63, 94, 0.0)');
        dangerGrad.addColorStop(1, 'rgba(244, 63, 94, 0.25)');
        ctx.fillStyle = dangerGrad;
        ctx.fillRect(0, dangerY, width, height - dangerY);
      } catch {}

      // Bottom Paddle Track Rail
      const railPt = worldToCanvas(0, env.paddleY, width, height);
      const railY = isFinite(railPt.cy) ? railPt.cy : height - 40;
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.25)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 8]);
      ctx.beginPath();
      ctx.moveTo(12, railY);
      ctx.lineTo(width - 12, railY);
      ctx.stroke();
      ctx.setLineDash([]);

      // 2. Draw Bricks (Full Width & Upper Half Coverage)
      for (let r = 0; r < env.rows; r++) {
        for (let c = 0; c < env.cols; c++) {
          const brick = env.bricks[r][c];
          if (!brick.intact || brick.hp <= 0) continue;

          const bounds = env.getBrickBounds(r, c);
          const p1 = worldToCanvas(bounds.xMin, bounds.yMax, width, height);
          const p2 = worldToCanvas(bounds.xMax, bounds.yMin, width, height);

          const bw = Math.max(2, Math.abs(p2.cx - p1.cx) - 2.5);
          const bh = Math.max(2, Math.abs(p2.cy - p1.cy) - 2.5);
          const bx = p1.cx + 1.25;
          const by = p1.cy + 1.25;

          const colorInfo = BRICK_COLORS[brick.type] || BRICK_COLORS[BRICK_TYPES.NORMAL_BLUE];

          ctx.save();
          ctx.shadowColor = colorInfo.glow;
          ctx.shadowBlur = brick.type === BRICK_TYPES.UNBREAKABLE ? 3 : 8;

          const brickGrad = ctx.createLinearGradient(bx, by, bx, by + bh);
          if (brick.type === BRICK_TYPES.UNBREAKABLE) {
            brickGrad.addColorStop(0, '#64748b');
            brickGrad.addColorStop(1, '#334155');
          } else if (brick.hp > 1) {
            brickGrad.addColorStop(0, '#ec4899');
            brickGrad.addColorStop(1, '#be185d');
          } else {
            brickGrad.addColorStop(0, colorInfo.fill);
            brickGrad.addColorStop(1, colorInfo.stroke);
          }

          ctx.fillStyle = brickGrad;
          ctx.beginPath();
          ctx.roundRect(bx, by, bw, bh, 3);
          ctx.fill();

          ctx.strokeStyle = colorInfo.stroke;
          ctx.lineWidth = 1;
          ctx.stroke();

          // Highlight bevel on top edge
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(bx + 2, by + 1.5);
          ctx.lineTo(bx + bw - 2, by + 1.5);
          ctx.stroke();

          // Crack effect for damaged reinforced bricks
          if (brick.type === BRICK_TYPES.REINFORCED && brick.hp === 1) {
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(bx + bw * 0.3, by + 2);
            ctx.lineTo(bx + bw * 0.5, by + bh * 0.6);
            ctx.lineTo(bx + bw * 0.75, by + bh - 2);
            ctx.stroke();
          }

          // Unbreakable steel rivets
          if (brick.type === BRICK_TYPES.UNBREAKABLE) {
            ctx.fillStyle = '#cbd5e1';
            ctx.beginPath();
            ctx.arc(bx + 4, by + 4, 1.5, 0, Math.PI * 2);
            ctx.arc(bx + bw - 4, by + 4, 1.5, 0, Math.PI * 2);
            ctx.fill();
          }

          ctx.restore();
        }
      }

      // 3. Draw Trajectory Forecast Guide
      if (showTrajectory && env.ballVy < 0 && env.ballLaunched) {
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.55)';
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 2.5;

        const start = worldToCanvas(env.ballX, env.ballY, width, height);
        ctx.beginPath();
        ctx.moveTo(start.cx, start.cy);

        let simX = env.ballX;
        let simY = env.ballY;
        let simVx = env.ballVx;
        let simVy = env.ballVy;
        let dtSim = 0.02;

        for (let s = 0; s < 80; s++) {
          simX += simVx * dtSim;
          simY += simVy * dtSim;

          if (simX <= env.ballRadius || simX >= env.width - env.ballRadius) {
            simVx = -simVx;
            const bouncePt = worldToCanvas(simX, simY, width, height);
            ctx.lineTo(bouncePt.cx, bouncePt.cy);
          }

          if (simY <= env.paddleY) {
            const hitPt = worldToCanvas(simX, env.paddleY, width, height);
            ctx.lineTo(hitPt.cx, hitPt.cy);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(hitPt.cx, hitPt.cy, 8, 0, Math.PI * 2);
            ctx.strokeStyle = '#00f0ff';
            ctx.setLineDash([]);
            ctx.stroke();
            break;
          }
        }
        ctx.restore();
      }

      // 4. Draw Ball Trails
      if (env.ballLaunched) {
        trailRef.current.push({ x: env.ballX, y: env.ballY, alpha: 1.0 });
        if (trailRef.current.length > 8) trailRef.current.shift();

        trailRef.current.forEach((pt, idx) => {
          pt.alpha *= 0.85;
          const pos = worldToCanvas(pt.x, pt.y, width, height);
          const r = (env.ballRadius * (width / env.width)) * (0.4 + (idx / trailRef.current.length) * 0.6);
          ctx.save();
          ctx.fillStyle = `rgba(0, 240, 255, ${pt.alpha * 0.35})`;
          ctx.beginPath();
          ctx.arc(pos.cx, pos.cy, r, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        });
      }

      // 5. Draw Paddle (Vibrant, Prominent, Highly Visible)
      const padPos = worldToCanvas(env.paddleX, env.paddleY, width, height);
      const padX = isFinite(padPos.cx) ? padPos.cx : width / 2;
      const padY = isFinite(padPos.cy) ? padPos.cy : height - 40;
      const padW = isFinite(env.paddleWidth) && width > 0 ? env.paddleWidth * (width / env.width) : 120;
      const padH = isFinite(env.paddleHeight) && height > 0 ? env.paddleHeight * (height / env.height) : 22;

      ctx.save();
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 24;

      // Outer bezel container
      ctx.fillStyle = 'rgba(2, 6, 23, 0.95)';
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.roundRect(padX - padW / 2, padY - padH / 2, padW, padH, padH / 2);
      ctx.fill();
      ctx.stroke();

      // Glowing Neon Core
      try {
        const padGrad = ctx.createLinearGradient(
          padX - padW / 2,
          padY,
          padX + padW / 2,
          padY
        );
        padGrad.addColorStop(0, '#38bdf8');
        padGrad.addColorStop(0.3, '#00f0ff');
        padGrad.addColorStop(0.5, '#ffffff');
        padGrad.addColorStop(0.7, '#00f0ff');
        padGrad.addColorStop(1, '#38bdf8');

        ctx.fillStyle = padGrad;
        ctx.beginPath();
        ctx.roundRect(padX - padW / 2 + 4, padY - padH / 4, Math.max(2, padW - 8), padH / 2, padH / 4);
        ctx.fill();
      } catch {}

      // Red Center Aim Marker
      ctx.fillStyle = '#f43f5e';
      ctx.shadowColor = '#f43f5e';
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(padX, padY, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.restore();

      // 6. Draw Ball
      const rawBallPos = worldToCanvas(env.ballX, env.ballY, width, height);
      const ballX = isFinite(rawBallPos.cx) ? rawBallPos.cx : width / 2;
      const ballY = isFinite(rawBallPos.cy) ? rawBallPos.cy : height / 2;
      const ballR = isFinite(env.ballRadius) && width > 0 ? Math.max(3, env.ballRadius * (width / env.width)) : 10;

      ctx.save();
      ctx.shadowColor = '#00f0ff';
      ctx.shadowBlur = 18;

      try {
        const ballGrad = ctx.createRadialGradient(
          ballX - ballR * 0.3,
          ballY - ballR * 0.3,
          Math.max(0.1, ballR * 0.1),
          ballX,
          ballY,
          ballR
        );
        ballGrad.addColorStop(0, '#ffffff');
        ballGrad.addColorStop(0.4, '#a5f3fc');
        ballGrad.addColorStop(1, '#00f0ff');

        ctx.fillStyle = ballGrad;
      } catch {
        ctx.fillStyle = '#00f0ff';
      }
      ctx.beginPath();
      ctx.arc(ballX, ballY, ballR, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      ctx.restore();

      // 7. Draw Particles & Floating Scores
      env.particles.draw(ctx);

      // 8. Launch Prompt Overlay
      if (!env.ballLaunched && !isTerminated) {
        ctx.save();
        ctx.fillStyle = '#00f0ff';
        ctx.font = "800 14px 'JetBrains Mono', monospace";
        ctx.textAlign = 'center';
        ctx.shadowColor = '#00f0ff';
        ctx.shadowBlur = 12;
        ctx.fillText('PRESS SPACE OR CLICK TO LAUNCH', width / 2, height - 75);
        ctx.restore();
      }

      ctx.restore();

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [env, showTrajectory, isTerminated, worldToCanvas]);

  // Handle Dynamic Full-Container Canvas Resizing
  useEffect(() => {
    const handleResize = () => {
      if (!canvasRef.current || !containerRef.current) return;
      const container = containerRef.current;
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();

      const targetW = Math.max(360, Math.floor(rect.width));
      const targetH = Math.max(480, Math.min(800, Math.floor(window.innerHeight - 220)));

      canvasDimRef.current = { width: targetW, height: targetH };

      const canvas = canvasRef.current;
      canvas.width = Math.floor(targetW * dpr);
      canvas.height = Math.floor(targetH * dpr);
      canvas.style.width = `${targetW}px`;
      canvas.style.height = `${targetH}px`;
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Launch Ball on Click
  const handlePointerDown = () => {
    if (!env.ballLaunched) {
      env.launchBall();
    }
  };

  const isWon = env.breakableCount > 0 && env.brokenCount >= env.breakableCount;
  const isLoss = isTerminated && !isWon;

  useEffect(() => {
    if (isWon) {
      confetti({
        particleCount: 90,
        spread: 80,
        origin: { y: 0.6 },
        colors: ['#00f0ff', '#ec4899', '#10b981', '#f59e0b'],
      });
    }
  }, [isWon]);

  return (
    <div className="relative flex flex-col items-center select-none w-full h-full" ref={containerRef}>
      {/* Canvas Container with Full Width & Height */}
      <div className="relative w-full rounded-2xl overflow-hidden shadow-2xl border-2 border-cyan-500/40 bg-slate-950/90 backdrop-blur-md flex items-center justify-center">
        <canvas
          ref={canvasRef}
          onClick={handlePointerDown}
          className="cursor-default block w-full"
        />

        {/* Victory Overlay */}
        {isWon && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center animate-fadeIn z-20">
            <div className="w-16 h-16 rounded-full bg-emerald-500/20 border border-emerald-500 flex items-center justify-center mb-3 shadow-lg shadow-emerald-500/30">
              <Award className="w-9 h-9 text-emerald-400" />
            </div>
            <h2 className="text-3xl font-black text-white tracking-wide mb-1 font-mono">
              STAGE CLEARED!
            </h2>
            <p className="text-emerald-400 font-medium mb-4 text-sm">
              Terminal Reward: +{env.rewardWeights.stageClear} pts
            </p>
            <div className="grid grid-cols-2 gap-3 mb-5 w-full max-w-xs text-left">
              <div className="metric-box">
                <span className="text-xs text-slate-400">Total Score</span>
                <span className="text-xl font-bold font-mono text-cyan-300">{env.score}</span>
              </div>
              <div className="metric-box">
                <span className="text-xs text-slate-400">Episode Reward</span>
                <span className="text-xl font-bold font-mono text-emerald-400">
                  {env.episodeReward.toFixed(1)}
                </span>
              </div>
            </div>
            <button onClick={onRestart} className="btn-primary">
              <RotateCcw className="w-4 h-4" /> Next Level / Restart
            </button>
          </div>
        )}

        {/* Loss Overlay */}
        {isLoss && (
          <div className="absolute inset-0 bg-slate-950/85 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center animate-fadeIn z-20">
            <div className="w-16 h-16 rounded-full bg-rose-500/20 border border-rose-500 flex items-center justify-center mb-3 shadow-lg shadow-rose-500/30">
              <AlertCircle className="w-9 h-9 text-rose-400" />
            </div>
            <h2 className="text-3xl font-black text-rose-500 tracking-wide mb-1 font-mono">
              BALL MISSED!
            </h2>
            <p className="text-slate-400 text-sm mb-4">
              Terminal Penalty: {env.rewardWeights.ballDropPenalty.toFixed(1)} • Episode Ended
            </p>
            <div className="grid grid-cols-2 gap-3 mb-5 w-full max-w-xs text-left">
              <div className="metric-box">
                <span className="text-xs text-slate-400">Final Score</span>
                <span className="text-xl font-bold font-mono text-cyan-300">{env.score}</span>
              </div>
              <div className="metric-box">
                <span className="text-xs text-slate-400">Total Reward</span>
                <span className="text-xl font-bold font-mono text-amber-400">
                  {env.episodeReward.toFixed(1)}
                </span>
              </div>
            </div>
            <button onClick={onRestart} className="btn-primary bg-rose-600 hover:bg-rose-500">
              <RotateCcw className="w-4 h-4" /> Play Again (Space / R)
            </button>
          </div>
        )}

        {/* Paused Overlay */}
        {isPaused && !isWon && !isLoss && (
          <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex flex-col items-center justify-center p-6 z-10">
            <span className="badge-neon badge-amber mb-3 text-sm px-4 py-1">PAUSED</span>
            <p className="text-slate-300 text-sm mb-4">Game is currently paused</p>
            <button onClick={onTogglePause} className="btn-primary">
              <Play className="w-4 h-4 fill-current" /> Resume Game
            </button>
          </div>
        )}
      </div>

      {/* Level Status Footer */}
      <div className="mt-2.5 flex items-center justify-between w-full px-2 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full animate-pulse ${playerMode !== 'human' ? 'bg-pink-400' : 'bg-cyan-400'}`} />
          <span>
            Player: <strong className={playerMode !== 'human' ? 'text-pink-300' : 'text-slate-200'}>
              {playerMode === 'agent1_vector' ? 'AI AGENT 1 (Vector-DQN 134D)' :
               playerMode === 'agent2_cnn' ? 'AI AGENT 2 (Spatial CNN 88/88 Winner)' :
               (playerMode === 'agent3_dqn' || playerMode === 'agent3_ppo') ? 'AI AGENT 3 (Rainbow DQN 🌈)' :
               'HUMAN (A / D / Arrow Keys)'}
            </strong>
          </span>
        </div>
        <div className="flex items-center gap-2.5">
          <span>Layout: <strong className="text-cyan-300">{env.presetName}</strong></span>
          {env.preset?.badge && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-bold border ${env.preset.badgeColor || 'border-slate-700'}`}>
              {env.preset.badge}
            </span>
          )}
          <span>Bricks: <strong className="text-emerald-400">{env.breakableCount - env.brokenCount}/{env.breakableCount}</strong></span>
        </div>
      </div>
    </div>
  );
}
