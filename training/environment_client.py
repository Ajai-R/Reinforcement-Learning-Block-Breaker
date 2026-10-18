"""
Breakout Environment Client running the EXACT JavaScript Environment via Node.js IPC.

Guarantees 100% physics, collision, and reward parity with the React application.
Returns the 134-dimensional observation vector:
[paddle_x, ball_x, ball_y, ball_vx, ball_vy, bricks_left_ratio, ...brick_layout_8x16]
"""

import os
import sys
import json
import base64
import subprocess
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BreakoutClientEnv:
    """
   environment interface executing the real JavaScript BreakoutEnv via Node.js IPC.
    """

    def __init__(self, frame_skip: int = 4, render_frame: bool = False):
        self.frame_skip = frame_skip
        self.render_frame = render_frame
        self.state_dim = 134
        self.action_dim = 3  # 0: LEFT, 1: STAY, 2: RIGHT

        script_dir = os.path.dirname(os.path.abspath(__file__))
        runner_path = os.path.join(script_dir, "node_env_runner.js")
        project_root = os.path.dirname(script_dir)

        # Launch Node.js environment runner subprocess
        self.proc = subprocess.Popen(
            ["node", runner_path],
            cwd=project_root,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered
        )

        if self.proc.poll() is not None:
            err = self.proc.stderr.read()
            raise RuntimeError(f"Failed to start Node.js environment runner: {err}")

    def _send_cmd(self, payload: dict) -> dict:
        """Sends a JSON command to the Node runner and parses the response."""
        line = json.dumps(payload) + "\n"
        self.proc.stdin.write(line)
        self.proc.stdin.flush()

        resp_line = self.proc.stdout.readline()
        if not resp_line:
            err = self.proc.stderr.read() if self.proc.stderr else "Unknown error"
            raise RuntimeError(f"Node.js environment runner closed unexpectedly: {err}")

        data = json.loads(resp_line)
        if "error" in data:
            raise RuntimeError(f"Error from JS environment: {data['error']}")
        return data

    def reset(self, max_steps: Optional[int] = None, full_reset: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Resets the JS Breakout environment.

        Args:
            max_steps: Optional maximum steps before episode truncation (default 8000).
            full_reset: Whether to reset the brick layout (True) or preserve broken bricks (False).

        Returns:
            obs: np.ndarray of shape (84, 84) uint8 (if render_frame=True) or shape (134,) float32
            info: dict containing environment stats
        """
        payload = {"cmd": "reset", "render_frame": self.render_frame, "full_reset": full_reset}
        if max_steps is not None:
            payload["max_steps"] = int(max_steps) * self.frame_skip
        else:
            payload["max_steps"] = 16000
        res = self._send_cmd(payload)

        info = res.get("info", {})
        if self.render_frame and "frame" in res:
            raw_bytes = base64.b64decode(res["frame"])
            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((84, 84))
            info["vector_obs"] = np.array(res["obs"], dtype=np.float32)
            return frame, info

        obs = np.array(res["obs"], dtype=np.float32)
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Steps the JS Breakout environment with the given action.

        Args:
            action: 0 (LEFT), 1 (STAY), or 2 (RIGHT)

        Returns:
            obs: np.ndarray of shape (84, 84) uint8 (if render_frame=True) or shape (134,) float32
            reward: float
            terminated: bool (ball lost or all breakable bricks destroyed)
            truncated: bool (max episode steps reached)
            info: dict containing score, bricks remaining, etc.
        """
        res = self._send_cmd({
            "cmd": "step",
            "action": int(action),
            "frame_skip": self.frame_skip,
            "render_frame": self.render_frame,
        })

        info = res.get("info", {})
        reward = float(res["reward"])
        terminated = bool(res["terminated"])
        truncated = bool(res["truncated"])

        if self.render_frame and "frame" in res:
            raw_bytes = base64.b64decode(res["frame"])
            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((84, 84))
            info["vector_obs"] = np.array(res["obs"], dtype=np.float32)
            return frame, reward, terminated, truncated, info

        obs = np.array(res["obs"], dtype=np.float32)
        return obs, reward, terminated, truncated, info

    def close(self):
        """Terminates the Node.js runner process."""
        if self.proc and self.proc.poll() is None:
            try:
                self._send_cmd({"cmd": "close"})
            except Exception:
                pass
            self.proc.terminate()
            self.proc.wait()

    def __del__(self):
        self.close()
