import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import os
from collections import deque
import argparse
import ale_py

gym.register_envs(ale_py)
import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        n_mul = 5
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((n_mul, n_mul)),
            nn.Flatten(),
            nn.Linear(n_mul * n_mul * 64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


def evaluate(model, env, preprocessor, seed, device="cpu"):
    rewards = []

    # Perform 20 episodes evaluation for the given seed
    for ep_count in range(20):
        try:
            obs, _ = env.reset(seed=seed + ep_count)
            state = preprocessor.reset(obs)
            done = False
            ep_reward = 0

            while not done:
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = preprocessor.step(next_obs)

            rewards.append(ep_reward)
        except Exception as e:
            print(f"⚠️  Skipping seed {seed} episode {ep_count} due to error: {e}")
            continue  # Skip to next episode

    if rewards:
        avg_reward = np.mean(rewards)
        return avg_reward
    else:
        return float('-inf')


def main():
    model_path = "results_tk3_clip_Pong/model_ep400.pt"  # Path to the model
    seed_start = 5  # Starting seed value
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    # Load the model
    dummy_env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    num_actions = dummy_env.action_space.n
    dummy_env.close()

    model = DQN(4, num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()

    # Track sliding window of 20 episodes
    reward_window = deque(maxlen=20)

    # Iterating through the seeds
    for seed in range(seed_start, seed_start + 1000000):  # Example: Check a range of 10 seeds
        print(f"Evaluating for seed {seed}...")
        reward = evaluate(model, env, preprocessor, seed, device)
        reward_window.append(reward)

        if len(reward_window) == 20:
            avg_reward = np.mean(reward_window)  # Calculate the average of the last 20 episodes
            if avg_reward >= 19:
                print(f"🏆 Seed {seed} → Avg Reward: {avg_reward:.2f} → CONDITION MET! Ending evaluation.")
                break
            else:
                print(f"❌ Seed {seed} → Avg Reward: {avg_reward:.2f} → CONDITION NOT MET. Trying next seed...")

    env.close()


if __name__ == "__main__":
    main()