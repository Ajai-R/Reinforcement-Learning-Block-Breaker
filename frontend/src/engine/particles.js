// Particle Emitter & Floating Text System for Block Breaker Canvas

export class ParticleSystem {
  constructor() {
    this.particles = [];
    this.floatingTexts = [];
    this.shakeTime = 0;
    this.shakeMagnitude = 0;
  }

  triggerShake(magnitude = 6, duration = 12) {
    this.shakeMagnitude = magnitude;
    this.shakeTime = duration;
  }

  getShakeOffset() {
    if (this.shakeTime <= 0) return { x: 0, y: 0 };
    const factor = this.shakeTime / 12;
    const x = (Math.random() * 2 - 1) * this.shakeMagnitude * factor;
    const y = (Math.random() * 2 - 1) * this.shakeMagnitude * factor;
    return { x, y };
  }

  emitBrickShatter(x, y, width, height, color, count = 18) {
    for (let i = 0; i < count; i++) {
      const px = x + Math.random() * width;
      const py = y + Math.random() * height;
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.5 + Math.random() * 4.5;
      this.particles.push({
        x: px,
        y: py,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed - 1.0, // slight upward bias
        color: color.fill,
        glow: color.glow,
        size: 2.5 + Math.random() * 3.5,
        alpha: 1.0,
        life: 1.0,
        decay: 0.02 + Math.random() * 0.03,
        gravity: 0.12,
      });
    }
  }

  emitPaddleSpark(x, y, count = 10) {
    for (let i = 0; i < count; i++) {
      const angle = -Math.PI / 2 + (Math.random() - 0.5) * 1.2;
      const speed = 2.0 + Math.random() * 3.5;
      this.particles.push({
        x: x + (Math.random() - 0.5) * 20,
        y: y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        color: '#38bdf8',
        glow: 'rgba(56, 189, 248, 0.8)',
        size: 2 + Math.random() * 2,
        alpha: 1.0,
        life: 1.0,
        decay: 0.04 + Math.random() * 0.03,
        gravity: 0.08,
      });
    }
  }

  addFloatingText(text, x, y, color = '#38bdf8', size = 14) {
    this.floatingTexts.push({
      text,
      x,
      y,
      vy: -1.2,
      alpha: 1.0,
      decay: 0.025,
      color,
      size,
    });
  }

  update() {
    if (this.shakeTime > 0) {
      this.shakeTime -= 1;
    }

    // Update particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vy += p.gravity;
      p.life -= p.decay;
      p.alpha = Math.max(0, p.life);
      if (p.life <= 0) {
        this.particles.splice(i, 1);
      }
    }

    // Update floating texts
    for (let i = this.floatingTexts.length - 1; i >= 0; i--) {
      const t = this.floatingTexts[i];
      t.y += t.vy;
      t.alpha -= t.decay;
      if (t.alpha <= 0) {
        this.floatingTexts.splice(i, 1);
      }
    }
  }

  draw(ctx) {
    // Draw particles
    for (const p of this.particles) {
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.glow || p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Draw floating texts
    for (const t of this.floatingTexts) {
      ctx.save();
      ctx.globalAlpha = t.alpha;
      ctx.font = `700 ${t.size}px 'JetBrains Mono', monospace`;
      ctx.fillStyle = t.color;
      ctx.shadowColor = t.color;
      ctx.shadowBlur = 6;
      ctx.textAlign = 'center';
      ctx.fillText(t.text, t.x, t.y);
      ctx.restore();
    }
  }

  clear() {
    this.particles = [];
    this.floatingTexts = [];
    this.shakeTime = 0;
  }
}
