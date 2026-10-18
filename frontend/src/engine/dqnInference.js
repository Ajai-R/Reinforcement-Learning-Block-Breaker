// Pure JavaScript In-Browser Inference Engine for Vector-DQN (134-dim input)
// Supports Two-Stream Dueling MLP, Standard Dueling MLP, and expert heuristic fallback.

export class DQNInference {
  constructor(modelJson = null) {
    this.modelData = null;
    this.isModelLoaded = false;
    if (modelJson) {
      this.loadWeights(modelJson);
    }
  }

  loadWeights(modelJson) {
    try {
      this.modelData = typeof modelJson === 'string' ? JSON.parse(modelJson) : modelJson;
      this.weights = this.modelData.weights;
      if (!this.weights || typeof this.weights !== 'object') {
        throw new Error('Model JSON does not contain exported weights');
      }
      this.dueling = Boolean(this.modelData.dueling);
      this.stateDim = this.modelData.state_dim || 134;
      this.numActions = this.modelData.num_actions || 3;
      this.isModelLoaded = true;
      console.log('[DQNInference] Model weights loaded successfully! Dueling:', this.dueling);
      return true;
    } catch (err) {
      console.error('[DQNInference] Failed to load model weights:', err);
      this.isModelLoaded = false;
      return false;
    }
  }

  // 2D Convolution: (C_in, H, W) -> (C_out, H, W) with ReLU
  static conv2d(input, inC, H, W, weight, bias, outC, kernelSize = 3, padding = 1) {
    const outH = H;
    const outW = W;
    const output = new Float32Array(outC * outH * outW);
    const halfK = Math.floor(kernelSize / 2);

    for (let oc = 0; oc < outC; oc++) {
      const ocOffset = oc * outH * outW;
      const b = bias ? bias[oc] : 0.0;
      const wOc = weight[oc];

      for (let r = 0; r < outH; r++) {
        const rowOffset = ocOffset + r * outW;
        for (let c = 0; c < outW; c++) {
          let sum = b;
          for (let ic = 0; ic < inC; ic++) {
            const inOffset = ic * H * W;
            const wIc = wOc[ic];
            for (let kr = 0; kr < kernelSize; kr++) {
              const inR = r + kr - halfK;
              if (inR >= 0 && inR < H) {
                const inRowOffset = inOffset + inR * W;
                const wRow = wIc[kr];
                for (let kc = 0; kc < kernelSize; kc++) {
                  const inCol = c + kc - halfK;
                  if (inCol >= 0 && inCol < W) {
                    sum += input[inRowOffset + inCol] * wRow[kc];
                  }
                }
              }
            }
          }
          output[rowOffset + c] = sum > 0.0 ? sum : 0.0;
        }
      }
    }
    return output;
  }

  // 2x2 Max Pooling: (C, H, W) -> (C, H/2, W/2)
  static maxpool2d(input, C, H, W) {
    const outH = Math.floor(H / 2);
    const outW = Math.floor(W / 2);
    const output = new Float32Array(C * outH * outW);

    for (let c = 0; c < C; c++) {
      const inCOffset = c * H * W;
      const outCOffset = c * outH * outW;
      for (let r = 0; r < outH; r++) {
        const inR2 = r * 2;
        const outRowOffset = outCOffset + r * outW;
        for (let col = 0; col < outW; col++) {
          const inC2 = col * 2;
          const idx0 = inCOffset + inR2 * W + inC2;
          const m0 = input[idx0];
          const m1 = input[idx0 + 1];
          const m2 = input[idx0 + W];
          const m3 = input[idx0 + W + 1];
          output[outRowOffset + col] = Math.max(m0, m1, m2, m3);
        }
      }
    }
    return output;
  }

  // 2D Convolution with arbitrary Stride and Padding:
  static conv2d_strided(input, inC, inH, inW, weight, bias, outC, kernelSize, stride, padding = 0) {
    const outH = Math.floor((inH - kernelSize + 2 * padding) / stride) + 1;
    const outW = Math.floor((inW - kernelSize + 2 * padding) / stride) + 1;
    const output = new Float32Array(outC * outH * outW);

    for (let oc = 0; oc < outC; oc++) {
      const ocOffset = oc * outH * outW;
      const b = bias ? bias[oc] : 0.0;
      const wOc = weight[oc];

      for (let r = 0; r < outH; r++) {
        const inR0 = r * stride - padding;
        const rowOffset = ocOffset + r * outW;
        for (let c = 0; c < outW; c++) {
          const inC0 = c * stride - padding;
          let sum = b;

          for (let ic = 0; ic < inC; ic++) {
            const inChannelOffset = ic * inH * inW;
            const wIc = wOc[ic];

            for (let kr = 0; kr < kernelSize; kr++) {
              const inR = inR0 + kr;
              if (inR >= 0 && inR < inH) {
                const inRowOffset = inChannelOffset + inR * inW;
                const wRow = wIc[kr];
                for (let kc = 0; kc < kernelSize; kc++) {
                  const inCol = inC0 + kc;
                  if (inCol >= 0 && inCol < inW) {
                    sum += input[inRowOffset + inCol] * wRow[kc];
                  }
                }
              }
            }
          }
          output[rowOffset + c] = sum > 0.0 ? sum : 0.0;
        }
      }
    }
    return output;
  }

  // Dense Layer: y = ReLU(W * x + b) or Linear(W * x + b)
  static linear(x, weight, bias, activation = 'relu') {
    const outDim = weight.length;
    const inDim = x.length;
    const out = new Float32Array(outDim);

    for (let i = 0; i < outDim; i++) {
      let sum = bias ? bias[i] : 0.0;
      const row = weight[i];
      for (let j = 0; j < inDim; j++) {
        sum += row[j] * x[j];
      }
      if (activation === 'relu') {
        out[i] = sum > 0.0 ? sum : 0.0;
      } else if (activation === 'tanh') {
        out[i] = Math.tanh(sum);
      } else {
        out[i] = sum;
      }
    }
    return out;
  }

  predict(stateVector, env = null) {
    if (!this.isModelLoaded || !this.weights) {
      return this.heuristicPredict(stateVector);
    }

    const x = stateVector;

    try {
      // 0. Check for DeepMind Nature DQN (84x84x4 raw pixels)
      if (this.weights['conv1.weight']) {
        if (!this.frameRingBuffer) this.frameRingBuffer = [];
        const rawFrame = env ? env.renderFrameGrayscale(84, 84) : null;
        if (rawFrame) {
          this.frameRingBuffer.push(rawFrame);
          if (this.frameRingBuffer.length > 4) this.frameRingBuffer.shift();
          while (this.frameRingBuffer.length < 4) this.frameRingBuffer.unshift(rawFrame);
        }

        const stacked = new Float32Array(4 * 84 * 84);
        if (this.frameRingBuffer && this.frameRingBuffer.length === 4) {
          for (let ch = 0; ch < 4; ch++) {
            const f = this.frameRingBuffer[ch];
            const chOffset = ch * 84 * 84;
            for (let i = 0; i < 7056; i++) {
              stacked[chOffset + i] = f[i] / 255.0;
            }
          }
        }

        // Conv 1: 4 -> 16, 8x8, stride 4 (out: 16x20x20)
        const c1 = DQNInference.conv2d_strided(stacked, 4, 84, 84, this.weights['conv1.weight'], this.weights['conv1.bias'], 16, 8, 4, 0);
        // Conv 2: 16 -> 32, 4x4, stride 2 (out: 32x9x9 = 2592)
        const c2 = DQNInference.conv2d_strided(c1, 16, 20, 20, this.weights['conv2.weight'], this.weights['conv2.bias'], 32, 4, 2, 0);
        // FC1: 2592 -> 256
        const f1 = DQNInference.linear(c2, this.weights['fc1.weight'], this.weights['fc1.bias'], 'relu');
        // FC2: 256 -> 3
        const qVals = DQNInference.linear(f1, this.weights['fc2.weight'], this.weights['fc2.bias'], 'linear');

        const qValues = [0, 0, 0];
        let bestAction = 1;
        let maxQ = -Infinity;
        for (let i = 0; i < this.numActions; i++) {
          qValues[i] = qVals[i];
          if (qValues[i] > maxQ) {
            maxQ = qValues[i];
            bestAction = i;
          }
        }
        return { action: bestAction, qValues, isTrained: true, modelType: 'dqn' };
      }

      // 1. Check for Spatial Grid CNN Architecture (DQN or PPO)
      if (this.weights['conv.0.weight']) {
        const kinX = x.slice(0, 6);
        const grid = new Float32Array(3 * 16 * 16);

        // Channel 0: Bricks (top 8 rows of 16x16)
        if (x.length >= 134) {
          for (let i = 0; i < 128; i++) {
            grid[i] = x[6 + i];
          }
        }

        // Channel 1: Paddle (Row 14)
        const padNormX = kinX[0];
        const padCol = Math.max(1, Math.min(14, Math.round((padNormX + 1.0) * 0.5 * 15.0)));
        const ch1Offset = 16 * 16 + 14 * 16;
        grid[ch1Offset + padCol] = 1.0;
        if (padCol > 0) grid[ch1Offset + padCol - 1] = 0.8;
        if (padCol < 15) grid[ch1Offset + padCol + 1] = 0.8;

        // Channel 2: Ball
        const ballNormX = kinX[1];
        const ballNormY = kinX[2];
        const ballCol = Math.max(0, Math.min(15, Math.round((ballNormX + 1.0) * 0.5 * 15.0)));
        const ballRow = Math.max(0, Math.min(15, Math.round((1.0 - ballNormY) * 0.5 * 15.0)));
        const ch2Offset = 2 * 16 * 16 + ballRow * 16 + ballCol;
        grid[ch2Offset] = 1.0;

        // Conv 1 -> MaxPool
        const c1 = DQNInference.conv2d(grid, 3, 16, 16, this.weights['conv.0.weight'], this.weights['conv.0.bias'], 32, 3, 1);
        const p1 = DQNInference.maxpool2d(c1, 32, 16, 16);

        // Conv 2 -> MaxPool
        const c2 = DQNInference.conv2d(p1, 32, 8, 8, this.weights['conv.3.weight'], this.weights['conv.3.bias'], 64, 3, 1);
        const p2 = DQNInference.maxpool2d(c2, 64, 8, 8);

        // Conv 3
        const c3 = DQNInference.conv2d(p2, 64, 4, 4, this.weights['conv.6.weight'], this.weights['conv.6.bias'], 64, 3, 1);

        // Spatial Dense (1024 -> 128)
        const spatialFeat = DQNInference.linear(c3, this.weights['conv.9.weight'], this.weights['conv.9.bias'], 'relu');

        // Kinematic Dense (6 -> 64 -> 64)
        const k0 = DQNInference.linear(kinX, this.weights['kin_net.0.weight'], this.weights['kin_net.0.bias'], 'relu');
        const kinFeat = DQNInference.linear(k0, this.weights['kin_net.2.weight'], this.weights['kin_net.2.bias'], 'relu');

        // Combined (128 + 64 = 192)
        const combined = new Float32Array(spatialFeat.length + kinFeat.length);
        combined.set(spatialFeat, 0);
        combined.set(kinFeat, spatialFeat.length);

        // Case A: Full Rainbow DQN (C51 Distributional + Dueling + NoisyNet)
        if (this.weights['val_noisy1.weight_mu'] || this.modelData?.format === 'rainbow_c51_dqn') {
          const numAtoms = this.modelData?.num_atoms || 51;
          const vMin = this.modelData?.v_min ?? -150.0;
          const vMax = this.modelData?.v_max ?? 1000.0;
          const support = this.modelData?.support || Array.from({ length: numAtoms }, (_, i) => vMin + i * (vMax - vMin) / (numAtoms - 1));

          // Value stream (51 atom logits): combined (192) -> 128 -> 51
          const v0 = DQNInference.linear(combined, this.weights['val_noisy1.weight_mu'], this.weights['val_noisy1.bias_mu'], 'relu');
          const valLogits = DQNInference.linear(v0, this.weights['val_noisy2.weight_mu'], this.weights['val_noisy2.bias_mu'], 'linear');

          // Advantage stream (num_actions * 51 atom logits): combined (192) -> 128 -> 153
          const a0 = DQNInference.linear(combined, this.weights['adv_noisy1.weight_mu'], this.weights['adv_noisy1.bias_mu'], 'relu');
          const advLogits = DQNInference.linear(a0, this.weights['adv_noisy2.weight_mu'], this.weights['adv_noisy2.bias_mu'], 'linear');

          // Mean advantage per atom across actions
          const meanAdv = new Float32Array(numAtoms);
          for (let i = 0; i < numAtoms; i++) {
            let sum = 0.0;
            for (let a = 0; a < this.numActions; a++) {
              sum += advLogits[a * numAtoms + i];
            }
            meanAdv[i] = sum / this.numActions;
          }

          // Dueling combination & Softmax per action over 51 atoms -> expected Q
          const qValues = [0, 0, 0];
          let bestAction = 1;
          let maxQ = -Infinity;

          for (let a = 0; a < this.numActions; a++) {
            const aOffset = a * numAtoms;
            let maxLogit = -Infinity;
            for (let i = 0; i < numAtoms; i++) {
              const qLogit = valLogits[i] + (advLogits[aOffset + i] - meanAdv[i]);
              if (qLogit > maxLogit) maxLogit = qLogit;
            }

            let sumExp = 0.0;
            const probs = new Float32Array(numAtoms);
            for (let i = 0; i < numAtoms; i++) {
              const qLogit = valLogits[i] + (advLogits[aOffset + i] - meanAdv[i]);
              probs[i] = Math.exp(qLogit - maxLogit);
              sumExp += probs[i];
            }

            let expQ = 0.0;
            for (let i = 0; i < numAtoms; i++) {
              const p = probs[i] / sumExp;
              expQ += p * support[i];
            }

            qValues[a] = expQ;
            if (expQ > maxQ) {
              maxQ = expQ;
              bestAction = a;
            }
          }

          return { action: bestAction, qValues, isTrained: true, modelType: 'rainbow' };
        }

        // Case B: Spatial CNN-PPO (Actor-Critic)
        if (this.weights['actor.0.weight'] || this.modelData?.format === 'ppo_cnn_v1') {
          const a0 = DQNInference.linear(combined, this.weights['actor.0.weight'], this.weights['actor.0.bias'], 'relu');
          const logits = DQNInference.linear(a0, this.weights['actor.2.weight'], this.weights['actor.2.bias'], 'linear');

          // Softmax action probabilities
          let maxL = -Infinity;
          for (let i = 0; i < logits.length; i++) if (logits[i] > maxL) maxL = logits[i];
          let sumExp = 0.0;
          const probs = new Float32Array(logits.length);
          for (let i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - maxL);
            sumExp += probs[i];
          }
          let bestAction = 1;
          let maxP = -Infinity;
          for (let i = 0; i < logits.length; i++) {
            probs[i] /= sumExp;
            if (probs[i] > maxP) {
              maxP = probs[i];
              bestAction = i;
            }
          }
          return { action: bestAction, qValues: Array.from(probs), isTrained: true, modelType: 'ppo' };
        }

        // Case B: Spatial CNN Dueling DQN
        const v0 = DQNInference.linear(combined, this.weights['value_stream.0.weight'], this.weights['value_stream.0.bias'], 'relu');
        const v1 = DQNInference.linear(v0, this.weights['value_stream.2.weight'], this.weights['value_stream.2.bias'], 'linear');
        const stateValue = v1[0];

        const a0 = DQNInference.linear(combined, this.weights['advantage_stream.0.weight'], this.weights['advantage_stream.0.bias'], 'relu');
        const adv = DQNInference.linear(a0, this.weights['advantage_stream.2.weight'], this.weights['advantage_stream.2.bias'], 'linear');

        let meanAdv = 0.0;
        for (let i = 0; i < adv.length; i++) meanAdv += adv[i];
        meanAdv /= adv.length;

        const qValues = [0, 0, 0];
        let bestAction = 1;
        let maxQ = -Infinity;

        for (let i = 0; i < this.numActions; i++) {
          qValues[i] = stateValue + (adv[i] - meanAdv);
          if (qValues[i] > maxQ) {
            maxQ = qValues[i];
            bestAction = i;
          }
        }
        return { action: bestAction, qValues, isTrained: true, modelType: 'cnn' };
      }

      // 1. Check for Legacy Vector PPO Architecture (Actor-Critic)
      if (this.weights['actor.0.weight'] || this.modelData?.format === 'ppo_actor_v1') {
        const kinX = x.slice(0, 6);
        const k0 = DQNInference.linear(kinX, this.weights['kin_stream.0.weight'], this.weights['kin_stream.0.bias'], 'tanh');
        const k1 = DQNInference.linear(k0, this.weights['kin_stream.2.weight'], this.weights['kin_stream.2.bias'], 'tanh');

        const brickX = x.slice(6);
        const b0 = DQNInference.linear(brickX, this.weights['brick_stream.0.weight'], this.weights['brick_stream.0.bias'], 'tanh');
        const b1 = DQNInference.linear(b0, this.weights['brick_stream.2.weight'], this.weights['brick_stream.2.bias'], 'tanh');

        const combined = new Float32Array(k1.length + b1.length);
        combined.set(k1, 0);
        combined.set(b1, k1.length);

        const a0 = DQNInference.linear(combined, this.weights['actor.0.weight'], this.weights['actor.0.bias'], 'tanh');
        const logits = DQNInference.linear(a0, this.weights['actor.2.weight'], this.weights['actor.2.bias'], 'linear');

        // Softmax action probabilities
        let maxL = -Infinity;
        for (let i = 0; i < logits.length; i++) if (logits[i] > maxL) maxL = logits[i];
        let sumExp = 0.0;
        const probs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
          probs[i] = Math.exp(logits[i] - maxL);
          sumExp += probs[i];
        }
        let bestAction = 1;
        let maxP = -Infinity;
        for (let i = 0; i < logits.length; i++) {
          probs[i] /= sumExp;
          if (probs[i] > maxP) {
            maxP = probs[i];
            bestAction = i;
          }
        }
        return { action: bestAction, qValues: Array.from(probs), isTrained: true, modelType: 'ppo' };
      }

      // 2. Check for Two-Stream Architecture (kin_stream + brick_stream)
      if (this.weights['kin_stream.0.weight']) {

        const kinX = x.slice(0, 6);
        const k0 = DQNInference.linear(kinX, this.weights['kin_stream.0.weight'], this.weights['kin_stream.0.bias'], 'relu');
        const k1 = DQNInference.linear(k0, this.weights['kin_stream.2.weight'], this.weights['kin_stream.2.bias'], 'relu');

        let combined = k1;
        if (this.weights['brick_stream.0.weight'] && x.length > 6) {
          const brickX = x.slice(6);
          const b0 = DQNInference.linear(brickX, this.weights['brick_stream.0.weight'], this.weights['brick_stream.0.bias'], 'relu');
          const b1 = DQNInference.linear(b0, this.weights['brick_stream.2.weight'], this.weights['brick_stream.2.bias'], 'relu');

          combined = new Float32Array(k1.length + b1.length);
          combined.set(k1, 0);
          combined.set(b1, k1.length);
        }

        if (this.dueling) {
          const v0 = DQNInference.linear(combined, this.weights['value_stream.0.weight'], this.weights['value_stream.0.bias'], 'relu');
          const v1 = DQNInference.linear(v0, this.weights['value_stream.2.weight'], this.weights['value_stream.2.bias'], 'linear');
          const stateValue = v1[0];

          const a0 = DQNInference.linear(combined, this.weights['advantage_stream.0.weight'], this.weights['advantage_stream.0.bias'], 'relu');
          const adv = DQNInference.linear(a0, this.weights['advantage_stream.2.weight'], this.weights['advantage_stream.2.bias'], 'linear');

          let meanAdv = 0.0;
          for (let i = 0; i < adv.length; i++) meanAdv += adv[i];
          meanAdv /= adv.length;

          const qValues = [0, 0, 0];
          let bestAction = 1;
          let maxQ = -Infinity;

          for (let i = 0; i < this.numActions; i++) {
            qValues[i] = stateValue + (adv[i] - meanAdv);
            if (qValues[i] > maxQ) {
              maxQ = qValues[i];
              bestAction = i;
            }
          }
          return { action: bestAction, qValues, isTrained: true };
        } else {
          const h0 = DQNInference.linear(combined, this.weights['head.0.weight'], this.weights['head.0.bias'], 'relu');
          const qValues = DQNInference.linear(h0, this.weights['head.2.weight'], this.weights['head.2.bias'], 'linear');
          let bestAction = 1;
          let maxQ = -Infinity;
          for (let i = 0; i < this.numActions; i++) {
            if (qValues[i] > maxQ) {
              maxQ = qValues[i];
              bestAction = i;
            }
          }
          return { action: bestAction, qValues: Array.from(qValues), isTrained: true };
        }
      }

      // 2. Fallback for Legacy Single-Stream Architecture
      if (this.dueling && this.weights['feature_extractor.0.weight']) {
        const f0 = DQNInference.linear(x, this.weights['feature_extractor.0.weight'], this.weights['feature_extractor.0.bias'], 'relu');
        const f1 = DQNInference.linear(f0, this.weights['feature_extractor.2.weight'], this.weights['feature_extractor.2.bias'], 'relu');

        const v0 = DQNInference.linear(f1, this.weights['value_stream.0.weight'], this.weights['value_stream.0.bias'], 'relu');
        const v1 = DQNInference.linear(v0, this.weights['value_stream.2.weight'], this.weights['value_stream.2.bias'], 'linear');
        const stateValue = v1[0];

        const a0 = DQNInference.linear(f1, this.weights['advantage_stream.0.weight'], this.weights['advantage_stream.0.bias'], 'relu');
        const adv = DQNInference.linear(a0, this.weights['advantage_stream.2.weight'], this.weights['advantage_stream.2.bias'], 'linear');

        let meanAdv = 0.0;
        for (let i = 0; i < adv.length; i++) meanAdv += adv[i];
        meanAdv /= adv.length;

        const qValues = [0, 0, 0];
        let bestAction = 1;
        let maxQ = -Infinity;

        for (let i = 0; i < this.numActions; i++) {
          qValues[i] = stateValue + (adv[i] - meanAdv);
          if (qValues[i] > maxQ) {
            maxQ = qValues[i];
            bestAction = i;
          }
        }
        return { action: bestAction, qValues, isTrained: true };
      }

      return this.heuristicPredict(stateVector);
    } catch (err) {
      console.warn('[DQNInference] Forward pass error, using heuristic fallback:', err);
      return this.heuristicPredict(stateVector);
    }
  }

  // Smart baseline heuristic when no custom weights are active
  heuristicPredict(stateVector) {
    const paddleNormX = stateVector[0];
    const ballNormX = stateVector[1];
    const ballNormVx = stateVector[3];
    const ballNormVy = stateVector[4];

    // Predict intercept based on ball position and velocity
    let targetX = ballNormX;
    if (ballNormVy < 0) {
      targetX = ballNormX + ballNormVx * 0.35;
    }

    let action = 1; // STAY
    const diff = targetX - paddleNormX;

    if (diff < -0.04) {
      action = 0; // LEFT
    } else if (diff > 0.04) {
      action = 2; // RIGHT
    }

    return {
      action,
      qValues: [action === 0 ? 1.0 : 0.0, action === 1 ? 1.0 : 0.0, action === 2 ? 1.0 : 0.0],
      isTrained: false,
    };
  }
}
