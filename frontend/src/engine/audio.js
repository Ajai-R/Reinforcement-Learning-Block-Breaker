// Web Audio API Retro 8-bit Sound Synthesizer (Zero external dependencies)

class SoundFX {
  constructor() {
    this.ctx = null;
    this.enabled = true;
    this.masterGain = null;
  }

  init() {
    if (typeof window === 'undefined') return;
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
        this.masterGain = this.ctx.createGain();
        this.masterGain.gain.setValueAtTime(0.15, this.ctx.currentTime);
        this.masterGain.connect(this.ctx.destination);
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  toggleSound() {
    this.enabled = !this.enabled;
    return this.enabled;
  }

  playTone(freq, type = 'square', duration = 0.08, endFreq = null) {
    if (!this.enabled || !this.ctx) return;
    try {
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      if (endFreq) {
        osc.frequency.exponentialRampToValueAtTime(Math.max(10, endFreq), this.ctx.currentTime + duration);
      }

      gain.gain.setValueAtTime(0.3, this.ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(this.masterGain);

      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch {
      // Ignore audio interruptions
    }
  }

  paddleHit(hitOffset = 0) {
    this.init();
    // Frequency shifts slightly with hit angle
    const base = 280 + Math.abs(hitOffset) * 120;
    this.playTone(base, 'triangle', 0.07, base * 1.5);
  }

  brickHit() {
    this.init();
    this.playTone(480, 'square', 0.06, 620);
  }

  brickDestroy(combo = 1) {
    this.init();
    const comboFactor = Math.min(combo, 8);
    const freq = 440 + comboFactor * 60;
    this.playTone(freq, 'square', 0.09, freq * 1.6);
  }

  reinforcedCrack() {
    this.init();
    this.playTone(320, 'sawtooth', 0.08, 180);
  }

  unbreakableHit() {
    this.init();
    this.playTone(180, 'sawtooth', 0.05, 120);
  }

  wallBounce() {
    this.init();
    this.playTone(360, 'sine', 0.04, 300);
  }

  lifeLost() {
    this.init();
    if (!this.enabled || !this.ctx) return;
    try {
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(320, now);
      osc.frequency.linearRampToValueAtTime(80, now + 0.35);

      gain.gain.setValueAtTime(0.4, now);
      gain.gain.linearRampToValueAtTime(0.01, now + 0.35);

      osc.connect(gain);
      gain.connect(this.masterGain);
      osc.start(now);
      osc.stop(now + 0.35);
    } catch {}
  }

  victory() {
    this.init();
    if (!this.enabled || !this.ctx) return;
    const notes = [523.25, 659.25, 783.99, 1046.5]; // C5, E5, G5, C6
    notes.forEach((freq, i) => {
      setTimeout(() => {
        this.playTone(freq, 'square', 0.16, freq * 1.05);
      }, i * 110);
    });
  }

  gameOver() {
    this.init();
    if (!this.enabled || !this.ctx) return;
    const notes = [440, 415.3, 392, 349.2];
    notes.forEach((freq, i) => {
      setTimeout(() => {
        this.playTone(freq, 'sawtooth', 0.2, freq * 0.9);
      }, i * 140);
    });
  }
}

export const sound = new SoundFX();
