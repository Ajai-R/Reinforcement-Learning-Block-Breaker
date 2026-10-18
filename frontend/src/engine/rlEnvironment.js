// Reinforcement Learning Environment & Breakout Physics Engine
// Mathematically Robust Sub-Stepped Collision & High-Precision Angular Deflection

import { BRICK_TYPES, LEVEL_PRESETS, createBrickGrid } from './levels.js';
import { sound } from './audio.js';
import { ParticleSystem } from './particles.js';

export const ACTIONS = {
  LEFT: 0,
  STAY: 1,
  RIGHT: 2,
};

export const ACTION_NAMES = ['LEFT', 'STAY', 'RIGHT'];

export const DEFAULT_REWARDS = {
  stepSurvival: 0.002,
  paddleCatch: 12.0,
  brickBreak: 20.0,
  reinforcedDamage: 8.0,
  stageClear: 500.0,
  ballDropPenalty: -12.0,
  proximityShaping: 0.06,
};



export class BreakoutEnv {
  constructor(options = {}) {
    // World coordinates (0 to 10.0 units)
    this.width = 10.0;
    this.height = 10.0;
    this.fixedDt = options.fixedDt || (1.0 / 60.0); // Standardized fixed 60Hz physics timestep

    // Paddle specs (Compact, agile, sleek arcade size)
    this.paddleWidth = 1.1; // Reduced size as requested
    this.paddleHeight = 0.20;
    this.paddleY = 0.85;
    this.paddleSpeed = 8.0;
    this.targetPaddleX = 5.0;

    // Ball specs (Small, crisp, smooth motion)
    this.ballRadius = 0.11;
    this.baseSpeed = 4.0;
    this.maxSpeed = 6.8;

    // High Brick Placement (Leaves > 55% open flight space for human reaction)
    this.gridTop = 9.6;
    this.gridBottom = 6.5;
    this.brickMarginX = 0.2;

    // Level & Bricks
    this.levelKey = options.levelKey || 'invader';
    this.rewardWeights = { ...DEFAULT_REWARDS };
    this.trainingMode = Boolean(options.trainingMode);

    this.particles = new ParticleSystem();

    // Stats & Episode state
    this.score = 0;
    this.combo = 0;
    this.steps = 0;
    this.maxSteps = options.maxSteps || 8000; // Increased to 8000 steps (~133 seconds at 60Hz) to allow full board clear
    this.episodeReward = 0;
    this.totalBricks = 0;
    this.breakableCount = 0;
    this.brokenCount = 0;

    this.ballLaunched = false;
    this.rewardEvents = [];

    this.reset();
  }

  setLevel(levelKey) {
    this.levelKey = levelKey;
    this.reset(true);
  }

  resetLevel() {
    const { bricks, rows, cols, presetName, preset } = createBrickGrid(this.levelKey);
    this.bricks = bricks;
    this.rows = rows;
    this.cols = cols;
    this.presetName = presetName;
    this.preset = preset;

    const usableWidth = this.width - (this.brickMarginX * 2);
    this.brickW = usableWidth / this.cols;
    this.brickH = (this.gridTop - this.gridBottom) / this.rows;

    this.totalBricks = 0;
    this.breakableCount = 0;
    this.brokenCount = 0;

    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const b = this.bricks[r][c];
        if (b.type !== BRICK_TYPES.EMPTY) {
          this.totalBricks++;
          if (b.type !== BRICK_TYPES.UNBREAKABLE) {
            this.breakableCount++;
          }
        }
      }
    }
  }

  reset(fullReset = true, options = {}) {
    if (options.maxSteps) {
      this.maxSteps = options.maxSteps;
    }
    if (fullReset) {
      this.score = 0;
      this.steps = 0;
      this.episodeReward = 0;
      this.combo = 0;
      this.resetLevel();
      this.particles.clear();
      this.rewardEvents = [];
      this.milestones = { q1: false, q2: false, q3: false };
      this.stepsSinceBreak = 0;
    } else {
      this.combo = 0;
      this.particles.clear();
      this.rewardEvents = [];
    }

    this.paddleX = this.width / 2.0;

    this.targetPaddleX = this.paddleX;
    this.paddleVx = 0;

    // Ball on top of paddle ready for launch
    this.ballX = this.paddleX;
    this.ballY = this.paddleY + this.paddleHeight / 2 + this.ballRadius + 0.03;
    this.ballVx = 0;
    this.ballVy = 0;
    this.ballLaunched = false;

    if (this.trainingMode || options.autoLaunch) {
      this.launchBall();
    }

    return this.getObservation();
  }

  launchBall(vx = null, vy = null) {
    if (this.ballLaunched) return;
    this.ballLaunched = true;
    const initialVx = vx !== null ? vx : (Math.random() > 0.5 ? 1.8 : -1.8);
    const initialVy = vy !== null ? vy : this.baseSpeed;
    this.ballVx = initialVx;
    this.ballVy = initialVy;
  }

  getBrickBounds(r, c) {
    const xMin = this.brickMarginX + c * this.brickW;
    const xMax = xMin + this.brickW;
    const yMax = this.gridTop - r * this.brickH;
    const yMin = yMax - this.brickH;
    return { xMin, xMax, yMin, yMax };
  }

  // Smooth paddle position updater
  setTargetPaddleX(targetX) {
    const halfP = this.paddleWidth / 2;
    this.targetPaddleX = Math.max(halfP, Math.min(this.width - halfP, targetX));
  }

  step(action, dt = null) {
    const frameDt = dt || this.fixedDt;
    this.steps++;
    let stepReward = this.rewardWeights.stepSurvival;
    const rewardBreakdown = [];

    if (this.rewardWeights.stepSurvival !== 0) {
      rewardBreakdown.push({ type: 'SURVIVAL', val: this.rewardWeights.stepSurvival });
    }

    // 1. Smooth Paddle Movement & Target Interpolation
    const halfP = this.paddleWidth / 2;

    if (action === ACTIONS.LEFT) {
      this.targetPaddleX -= this.paddleSpeed * frameDt;
    } else if (action === ACTIONS.RIGHT) {
      this.targetPaddleX += this.paddleSpeed * frameDt;
    }
    this.targetPaddleX = Math.max(halfP, Math.min(this.width - halfP, this.targetPaddleX));

    if (this.trainingMode) {
      // Direct, responsive control for RL training without artificial smoothing delay
      this.paddleX = this.targetPaddleX;
      this.paddleVx = action === ACTIONS.LEFT ? -this.paddleSpeed : (action === ACTIONS.RIGHT ? this.paddleSpeed : 0);
    } else {
      // Exponential smoothing for fluid human control
      const smoothFactor = 1.0 - Math.exp(-40.0 * frameDt);
      const oldPaddleX = this.paddleX;
      this.paddleX += (this.targetPaddleX - this.paddleX) * smoothFactor;
      this.paddleVx = (this.paddleX - oldPaddleX) / Math.max(0.0001, frameDt);
    }

    // Dense proximity shaping when ball is descending towards the paddle
    if (this.ballLaunched && this.ballVy < 0 && this.ballY < 6.5 && this.rewardWeights.proximityShaping > 0) {
      const dist = Math.abs(this.paddleX - this.ballX);
      const maxRange = this.width / 2; // 5.0 units
      const proxScore = Math.max(0, 1.0 - dist / maxRange);
      const proxReward = proxScore * this.rewardWeights.proximityShaping;
      stepReward += proxReward;
    }

    // Ball held before launch (if not training mode)
    if (!this.ballLaunched) {
      this.ballX = this.paddleX;
      this.ballY = this.paddleY + this.paddleHeight / 2 + this.ballRadius + 0.03;
      this.episodeReward += stepReward;
      return {
        obs: this.getObservation(),
        reward: stepReward,
        terminated: false,
        truncated: false,
        breakdown: rewardBreakdown,
      };
    }

    // 2. Micro-Substepping for Mathematically Perfect CCD
    const numSubSteps = 4;
    const subDt = frameDt / numSubSteps;

    let lifeLostTerminated = false;
    let stageClearTerminated = false;

    for (let sub = 0; sub < numSubSteps; sub++) {
      // Kinematic update
      this.ballX += this.ballVx * subDt;
      this.ballY += this.ballVy * subDt;

      // Wall Collisions
      if (this.ballX <= this.ballRadius) {
        this.ballX = this.ballRadius;
        this.ballVx = Math.abs(this.ballVx);
        if (!this.trainingMode) sound.wallBounce();
      } else if (this.ballX >= this.width - this.ballRadius) {
        this.ballX = this.width - this.ballRadius;
        this.ballVx = -Math.abs(this.ballVx);
        if (!this.trainingMode) sound.wallBounce();
      }

      if (this.ballY >= this.height - this.ballRadius) {
        this.ballY = this.height - this.ballRadius;
        this.ballVy = -Math.abs(this.ballVy);
        if (!this.trainingMode) sound.wallBounce();
      }

      // Exact Minkowski Circle-AABB Collision & Normal Reflection
      let hitBrick = null;
      let hitRow = -1;
      let hitCol = -1;
      let minPenetration = Infinity;
      let collisionNormal = { x: 0, y: 0 };
      let closestPt = { x: 0, y: 0 };

      for (let r = 0; r < this.rows; r++) {
        for (let c = 0; c < this.cols; c++) {
          const brick = this.bricks[r][c];
          if (!brick.intact || brick.hp <= 0) continue;

          const bounds = this.getBrickBounds(r, c);
          const cx = Math.max(bounds.xMin, Math.min(this.ballX, bounds.xMax));
          const cy = Math.max(bounds.yMin, Math.min(this.ballY, bounds.yMax));

          const dx = this.ballX - cx;
          const dy = this.ballY - cy;
          const distSq = dx * dx + dy * dy;

          if (distSq < this.ballRadius * this.ballRadius) {
            const dist = Math.sqrt(distSq);
            let nx, ny;

            if (dist > 0.0001) {
              nx = dx / dist;
              ny = dy / dist;
            } else {
              const penLeft = this.ballX - bounds.xMin;
              const penRight = bounds.xMax - this.ballX;
              const penBottom = this.ballY - bounds.yMin;
              const penTop = bounds.yMax - this.ballY;
              const minPen = Math.min(penLeft, penRight, penBottom, penTop);

              if (minPen === penLeft) { nx = -1; ny = 0; }
              else if (minPen === penRight) { nx = 1; ny = 0; }
              else if (minPen === penBottom) { nx = 0; ny = -1; }
              else { nx = 0; ny = 1; }
            }

            const dot = this.ballVx * nx + this.ballVy * ny;
            if (dot < 0) {
              hitBrick = brick;
              hitRow = r;
              hitCol = c;
              collisionNormal = { x: nx, y: ny };
              closestPt = { x: cx, y: cy };
              break;
            }
          }
        }
        if (hitBrick) break;
      }

      // Resolve Brick Collision
      if (hitBrick) {
        const nx = collisionNormal.x;
        const ny = collisionNormal.y;
        const dot = this.ballVx * nx + this.ballVy * ny;

        this.ballVx = this.ballVx - 2 * dot * nx;
        this.ballVy = this.ballVy - 2 * dot * ny;

        this.ballX = closestPt.x + nx * (this.ballRadius + 0.002);
        this.ballY = closestPt.y + ny * (this.ballRadius + 0.002);

        // Process Brick Damage
        const bounds = this.getBrickBounds(hitRow, hitCol);
        if (hitBrick.type !== BRICK_TYPES.UNBREAKABLE) {
          hitBrick.hp--;
          if (hitBrick.hp <= 0) {
            hitBrick.intact = false;
            this.brokenCount++;
            this.combo++;
            const pts = hitBrick.points * Math.min(this.combo, 5);
            this.score += pts;

            const brickRwd = this.rewardWeights.brickBreak * (1 + (this.combo - 1) * 0.1);
            stepReward += brickRwd;
            rewardBreakdown.push({ type: `BRICK_BREAK (x${this.combo})`, val: brickRwd });

            // Milestone Clearing Bonuses
            this.stepsSinceBreak = 0;
            const ratio = this.breakableCount > 0 ? this.brokenCount / this.breakableCount : 0;
            if (ratio >= 0.25 && !this.milestones.q1) {
              this.milestones.q1 = true;
              stepReward += 25.0;
              rewardBreakdown.push({ type: 'MILESTONE_25%', val: 25.0 });
            }
            if (ratio >= 0.50 && !this.milestones.q2) {
              this.milestones.q2 = true;
              stepReward += 50.0;
              rewardBreakdown.push({ type: 'MILESTONE_50%', val: 50.0 });
            }
            if (ratio >= 0.75 && !this.milestones.q3) {
              this.milestones.q3 = true;
              stepReward += 100.0;
              rewardBreakdown.push({ type: 'MILESTONE_75%', val: 100.0 });
            }

            if (!this.trainingMode) {
              sound.brickDestroy(this.combo);
              this.particles.triggerShake(3, 6);
              this.particles.emitBrickShatter(bounds.xMin, bounds.yMin, this.brickW, this.brickH, { fill: '#38bdf8', glow: 'rgba(56,189,248,0.8)' }, 12);
              this.particles.addFloatingText(`+${pts}`, this.ballX, this.ballY, '#00f0ff', 12);
            }
          } else {
            stepReward += this.rewardWeights.reinforcedDamage;
            rewardBreakdown.push({ type: 'BRICK_CRACK', val: this.rewardWeights.reinforcedDamage });
            if (!this.trainingMode) {
              sound.reinforcedCrack();
              this.particles.triggerShake(2, 4);
              this.particles.addFloatingText('CRACK!', this.ballX, this.ballY, '#ec4899', 11);
            }
          }
        } else {
          if (!this.trainingMode) {
            sound.unbreakableHit();
            this.particles.triggerShake(1, 3);
          }
        }
      }

      // Paddle Catch Interception
      const paddleTop = this.paddleY + this.paddleHeight / 2;
      const paddleBottom = this.paddleY - this.paddleHeight / 2;
      const paddleLeft = this.paddleX - this.paddleWidth / 2;
      const paddleRight = this.paddleX + this.paddleWidth / 2;

      if (this.ballY - this.ballRadius <= paddleTop &&
          this.ballY + this.ballRadius >= paddleBottom &&
          this.ballVy < 0) {
        if (this.ballX >= paddleLeft - 0.08 && this.ballX <= paddleRight + 0.08) {
          const offset = (this.ballX - this.paddleX) / (this.paddleWidth / 2);
          const clampedOffset = Math.max(-0.92, Math.min(0.92, offset));
          const maxAngle = (60 * Math.PI) / 180;
          const bounceAngle = clampedOffset * maxAngle;

          const currentSpeed = Math.min(this.maxSpeed, Math.sqrt(this.ballVx * this.ballVx + this.ballVy * this.ballVy) * 1.008);
          this.ballVx = currentSpeed * Math.sin(bounceAngle);
          this.ballVy = currentSpeed * Math.cos(bounceAngle);

          this.ballY = paddleTop + this.ballRadius + 0.002;
          this.combo = 0;

          stepReward += this.rewardWeights.paddleCatch;
          rewardBreakdown.push({ type: 'PADDLE_CATCH', val: this.rewardWeights.paddleCatch });

          if (!this.trainingMode) {
            sound.paddleHit(clampedOffset);
            this.particles.emitPaddleSpark(this.ballX, paddleTop, 8);
            this.particles.addFloatingText('CATCH +5', this.paddleX, paddleTop + 0.3, '#10b981', 11);
          }
        }
      }

      // Check Ball Missed
      if (this.ballY < this.paddleY - 0.3) {
        lifeLostTerminated = true;
        break;
      }
    }

    // Check Terminations
    let terminated = false;
    let truncated = false;

    if (lifeLostTerminated) {
      stepReward += this.rewardWeights.ballDropPenalty;
      rewardBreakdown.push({ type: 'BALL_MISSED', val: this.rewardWeights.ballDropPenalty });
      if (!this.trainingMode) {
        sound.lifeLost();
        sound.gameOver();
        this.particles.triggerShake(7, 14);
      }
      terminated = true;
    }

    if (this.breakableCount > 0 && this.brokenCount >= this.breakableCount) {
      stepReward += this.rewardWeights.stageClear;
      rewardBreakdown.push({ type: 'STAGE_CLEAR', val: this.rewardWeights.stageClear });
      if (!this.trainingMode) sound.victory();
      terminated = true;
    }

    if (this.steps >= this.maxSteps && !terminated) {
      truncated = true;
    }

    this.episodeReward += stepReward;


    const meaningfulEvents = rewardBreakdown.filter((it) => it.type !== 'SURVIVAL');
    if (meaningfulEvents.length > 0) {
      this.rewardEvents.unshift({
        step: this.steps,
        total: stepReward,
        items: meaningfulEvents,
      });
      if (this.rewardEvents.length > 40) this.rewardEvents.pop();
    }

    if (!this.trainingMode) {
      this.particles.update();
    }

    return {
      obs: this.getObservation(),
      reward: stepReward,
      terminated,
      truncated,
      breakdown: rewardBreakdown,
    };
  }

  getObservation() {
    const normPaddleX = (this.paddleX - this.width / 2) / (this.width / 2);
    const normBallX = (this.ballX - this.width / 2) / (this.width / 2);
    const normBallY = (this.ballY - this.height / 2) / (this.height / 2);
    const normBallVx = this.ballVx / this.maxSpeed;
    const normBallVy = this.ballVy / this.maxSpeed;
    const normBricksLeft = this.breakableCount > 0 ? (this.breakableCount - this.brokenCount) / this.breakableCount : 0;

    // Flattened spatial brick grid layout (8 rows * 16 cols = 128 elements)
    // 0.0: destroyed/empty, >0: intact/damaged breakable hp ratio (1.0, 0.5), -1.0: unbreakable steel
    const brickLayout = [];
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const b = this.bricks[r][c];
        if (!b.intact || b.hp <= 0 || b.type === BRICK_TYPES.EMPTY) {
          brickLayout.push(0.0);
        } else if (b.type === BRICK_TYPES.UNBREAKABLE) {
          brickLayout.push(-1.0);
        } else {
          brickLayout.push(Number((b.hp / b.maxHp).toFixed(2)));
        }
      }
    }

    return {
      vector: [
        normPaddleX,
        normBallX,
        normBallY,
        normBallVx,
        normBallVy,
        normBricksLeft,
        ...brickLayout,
      ],
      brickLayout,
      rows: this.rows,
      cols: this.cols,
      raw: {
        paddleX: this.paddleX,
        ballX: this.ballX,
        ballY: this.ballY,
        ballVx: this.ballVx,
        ballVy: this.ballVy,
        score: this.score,
        steps: this.steps,
        episodeReward: this.episodeReward,
      },
    };
  }

  renderFrameGrayscale(targetW = 84, targetH = 84) {
    const frame = new Uint8Array(targetW * targetH);
    const scaleX = targetW / this.width;
    const scaleY = targetH / this.height;

    // 1. Render Intact Bricks
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.cols; c++) {
        const brick = this.bricks[r][c];
        if (!brick.intact || brick.hp <= 0 || brick.type === BRICK_TYPES.EMPTY) continue;

        const bounds = this.getBrickBounds(r, c);
        const x0 = Math.max(0, Math.floor(bounds.xMin * scaleX));
        const x1 = Math.min(targetW - 1, Math.ceil(bounds.xMax * scaleX) - 1);
        const y0 = Math.max(0, Math.floor((this.height - bounds.yMax) * scaleY));
        const y1 = Math.min(targetH - 1, Math.ceil((this.height - bounds.yMin) * scaleY) - 1);

        let pixelVal = 200;
        if (brick.type === BRICK_TYPES.UNBREAKABLE) pixelVal = 100;
        else if (brick.type === BRICK_TYPES.REINFORCED) {
          pixelVal = brick.hp > 1 ? 240 : 160;
        }

        for (let py = y0; py <= y1; py++) {
          const rowOffset = py * targetW;
          for (let px = x0; px <= x1; px++) {
            frame[rowOffset + px] = pixelVal;
          }
        }
      }
    }

    // 2. Render Paddle
    const padHalfW = this.paddleWidth / 2;
    const padHalfH = this.paddleHeight / 2;
    const px0 = Math.max(0, Math.floor((this.paddleX - padHalfW) * scaleX));
    const px1 = Math.min(targetW - 1, Math.ceil((this.paddleX + padHalfW) * scaleX));
    const py0 = Math.max(0, Math.floor((this.height - (this.paddleY + padHalfH)) * scaleY));
    const py1 = Math.min(targetH - 1, Math.ceil((this.height - (this.paddleY - padHalfH)) * scaleY));

    for (let py = py0; py <= py1; py++) {
      const rowOffset = py * targetW;
      for (let px = px0; px <= px1; px++) {
        frame[rowOffset + px] = 255;
      }
    }

    // 3. Render Ball
    const bx = Math.floor(this.ballX * scaleX);
    const by = Math.floor((this.height - this.ballY) * scaleY);
    const br = Math.max(1, Math.round(this.ballRadius * scaleX));

    for (let dy = -br; dy <= br; dy++) {
      const py = by + dy;
      if (py < 0 || py >= targetH) continue;
      const rowOffset = py * targetW;
      for (let dx = -br; dx <= br; dx++) {
        const px = bx + dx;
        if (px < 0 || px >= targetW) continue;
        if (dx * dx + dy * dy <= br * br + 1) {
          frame[rowOffset + px] = 255;
        }
      }
    }

    return frame;
  }
}
