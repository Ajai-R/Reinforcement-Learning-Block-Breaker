// WebSocket Training Bridge to connect React Breakout Environment to Python RL Trainer
// Communicates over ws://localhost:8765

export class TrainingBridge {
  constructor(env, options = {}) {
    this.env = env;
    this.url = options.url || 'ws://localhost:8765';
    this.ws = null;
    this.isConnected = false;
    this.onStatusChange = options.onStatusChange || (() => {});
    this.frameWidth = options.frameWidth || 84;
    this.frameHeight = options.frameHeight || 84;
  }

  connect() {
    if (this.ws) {
      this.disconnect();
    }

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.isConnected = true;
        this.onStatusChange({ connected: true, error: null });
        console.log(`[TrainingBridge] Connected to Python RL Server at ${this.url}`);
        this.sendMessage({ type: 'READY', info: { rows: this.env.rows, cols: this.env.cols } });
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          this.handleCommand(data);
        } catch (err) {
          console.error('[TrainingBridge] Failed to parse message:', err);
        }
      };

      this.ws.onerror = (err) => {
        this.onStatusChange({ connected: false, error: 'Connection Error' });
      };

      this.ws.onclose = () => {
        this.isConnected = false;
        this.onStatusChange({ connected: false, error: null });
        console.log('[TrainingBridge] Disconnected from Python RL Server');
      };
    } catch (err) {
      this.onStatusChange({ connected: false, error: err.message });
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    this.onStatusChange({ connected: false, error: null });
  }

  sendMessage(msg) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  handleCommand(cmd) {
    if (cmd.type === 'RESET') {
      const obs = this.env.reset();
      const frameArray = Array.from(this.env.renderFrameGrayscale(this.frameWidth, this.frameHeight));
      this.sendMessage({
        type: 'RESET_RESPONSE',
        frame: frameArray,
        obs: obs.vector,
        shape: [this.frameHeight, this.frameWidth],
        breakableCount: this.env.breakableCount,
      });
    } else if (cmd.type === 'STEP') {
      const action = cmd.action ?? 1;
      const frameSkip = cmd.frameSkip || 4;
      let totalReward = 0;
      let terminated = false;
      let truncated = false;
      let lastObs = null;

      for (let i = 0; i < frameSkip; i++) {
        const stepRes = this.env.step(action);
        totalReward += stepRes.reward;
        lastObs = stepRes.obs;
        if (stepRes.terminated) terminated = true;
        if (stepRes.truncated) truncated = true;
        if (terminated || truncated) break;
      }

      const frameArray = Array.from(this.env.renderFrameGrayscale(this.frameWidth, this.frameHeight));
      this.sendMessage({
        type: 'STEP_RESPONSE',
        frame: frameArray,
        reward: totalReward,
        terminated,
        truncated,
        obs: lastObs?.vector || this.env.getObservation().vector,
        score: this.env.score,
        brokenCount: this.env.brokenCount,
      });
    }
  }
}
