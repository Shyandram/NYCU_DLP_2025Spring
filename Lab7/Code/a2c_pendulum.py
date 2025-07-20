#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm
from typing import Tuple
import os

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer

def init_weights(m, gain=np.sqrt(2), bias_const=0.0):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain) # Orthogonal initialization
        nn.init.constant_(m.bias, bias_const)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############A
        # Remeber to initialize the layer weights

        # Define network layers
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_out = nn.Linear(64, out_dim)
        self.log_std_out = nn.Linear(64, out_dim)
        
        self.apply(init_weights)
        self.mean_out.apply(init_layer_uniform)
        self.log_std_out.apply(init_layer_uniform)
        
        self.log_std_min = -20
        self.log_std_max = 2
        
        #############################
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""

        ############TODO#############
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        
        mean = self.mean_out(x)
        log_std = self.log_std_out(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        
        # Create a Normal distribution object
        dist = Normal(mean, std)
        action = dist.sample() 
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 64)  
        self.fc2 = nn.Linear(64, 64)    
        self.value_out = nn.Linear(64, 1)  
        
        self.apply(init_weights)
        init_weights(self.value_out, gain=1.)
        
        
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        value = self.value_out(x)

        #############################

        return value

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]
        self.entropy = dist.entropy().mean()
        
        selected_action = torch.clamp(selected_action, self.env.action_space.low[0], self.env.action_space.high[0])
        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition.extend([next_state, reward, done])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the model by gradient descent."""
        state, log_prob, next_state, reward, done = self.transition
        state = state.to(self.device)
        log_prob = log_prob.to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - done
        
        ############TODO#############
        next_value = self.critic(next_state)
        Q_t = reward + self.gamma * next_value * mask
        current_value = self.critic(state)
        value_loss = F.mse_loss(Q_t.detach(), current_value)
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5)
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        advantage = Q_t - current_value

        policy_loss = -(advantage.detach() * log_prob) - self.entropy_weight * self.entropy
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_score = -np.inf
        
        episode_progress = tqdm(range(self.num_episodes))
        for ep in episode_progress: 
            actor_losses, critic_losses, scores = [], [], []
            seed = random.randint(0, 1e20) 
            state, _ = self.env.reset(seed=seed)
            score = 0
            done = False
            while not done:
                # self.env.render()  # Render the environment
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "episode": ep,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    }) 
                # if episode ends
                if done:
                    scores.append(score)
                    episode_progress.set_description_str(f"Episode {ep}: Total Reward = {score}")
                    # W&B logging
                    wandb.log({
                        "step": step_count,
                        "episode": ep,
                        "return": score
                        })  
                    # Save the model every 50 episodes
                    if ep % 50 == 0:
                        torch.save(self.actor.state_dict(), os.path.join(args.save, f"actor_{ep}.pth"))
                        torch.save(self.critic.state_dict(), os.path.join(args.save, f"critic_{ep}.pth"))
                    # Save the latest model
                    torch.save(self.actor.state_dict(), os.path.join(args.save, f"actor_latest.pth"))
                    torch.save(self.critic.state_dict(), os.path.join(args.save, f"critic_latest.pth"))
                    
                    # Save the model if the score is better than the best score
                    if ep % 10 == 0:
                        if score > best_score:
                            best_score = score
                            # Save the model
                            torch.save(self.actor.state_dict(), os.path.join(args.save, f"actor_best.pth"))
                            torch.save(self.critic.state_dict(), os.path.join(args.save, f"critic_best.pth"))
                    
                    self.actor_scheduler.step()
                    self.critic_scheduler.step()
    def test(self, video_folder: str, num_episodes: int = 20):
        """Test the agent for a specified number of episodes."""
        self.is_test = True

        tmp_env = self.env
        
        # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda t: True)
        total_score = []

        for ep in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed+ep)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

            print(f"Episode {ep + 1}: score = {score}")
            total_score.append(score)

        self.env.close()
        self.env = tmp_env
        print(f"Average score over {num_episodes} episodes: {np.mean(total_score)}")

def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="pendulum-a2c-run")
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=4000)
    parser.add_argument("--seed", type=int, default=648039789)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--save", type=str, default="task1-2")
    # Simulate command-line arguments in a notebook environment
    args = parser.parse_args([]) # Pass an empty list to avoid reading from command line
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 90
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # '''training'''
    # wandb.init(project="DLP-Lab7-A2C-Pendulum", name=args.wandb_run_name, save_code=True)
    # if not os.path.exists(args.save):
    #     os.makedirs(args.save)    
    # agent = A2CAgent(env, args)
    # agent.train()

    '''Load the best model'''
    agent = A2CAgent(env, args)
    agent.actor.load_state_dict(torch.load(f"LAB7_413551036_task1_a2c_pendulum.pth"))
    agent.test(video_folder="videos_Task1")