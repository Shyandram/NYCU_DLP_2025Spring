#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2025, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import random, os
from collections import deque
from typing import Deque, List, Tuple

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
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean_out = nn.Linear(64, out_dim)
        self.log_std_out = nn.Linear(64, out_dim)
        
        self.apply(init_weights)
        self.mean_out.apply(init_layer_uniform)
        self.log_std_out.apply(init_layer_uniform)

        # Set the min and max for log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
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
        
        # Normal distribution
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
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    advantages = []
    adv_next = torch.tensor(0.0, device=rewards[0].device)

    for i in reversed(range(len(rewards))):
        reward = rewards[i]
        current_value = values[i]
        mask = masks[i]

        if i == len(rewards) - 1:
            next_val = next_value * mask
        else:
            next_val = values[i + 1] * mask

        # delta = r + gamma * V(s') - V(s)
        delta = reward + gamma * next_val - current_value
        # A_t = delta_t + gamma * tau * A_{t+1}
        adv_t = delta + gamma * tau * adv_next * mask
        adv_next = adv_t

        advantages.insert(0, adv_t)
    # G_t = A_t + V(s_t)
    gae_returns = [adv + val for adv, val in zip(advantages, values)]

    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.save = args.save
        
        # device: cpu / gpu
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(self.device)

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.005)

        self.actor.scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.9)
        self.critic.scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            entropy = dist.entropy().mean()

            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            actor_loss = -torch.min(ratio*adv, clipped_ratio*adv).mean() - self.entropy_weight * entropy
            
            
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            critic_loss = F.mse_loss(return_, self.critic(state))

            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss

    def train(self):
        """Train the PPO agent."""
        self.is_test = False


        actor_losses, critic_losses = [], []
        scores = []
        step_count =0
        episode_iterator = tqdm(range(1, self.num_episodes))
        for ep in episode_iterator:
            score = 0
            seed = random.randint(0, 1e30)
            # print("\n")
            # state, _ = self.env.reset()
            state, _ = self.env.reset(seed=seed)
            state = np.expand_dims(state, axis=0)
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]
                step_count += 1

                # if episode ends
                if done[0][0]:
                    # state, _ = self.env.reset()
                    state, _ = self.env.reset(seed=seed)
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    episode_iterator.set_description_str(f"Episode {ep}: Total Reward = {score}")
                    wandb.log({
                        "episode": ep,
                        "step": step_count,
                        "score": score,
                    })
                    score = 0

            actor_loss, critic_loss = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            self.actor.scheduler.step()
            self.critic.scheduler.step()

            # W&B logging
            wandb.log({
                "step": step_count,
                "actor loss": actor_loss,
                "critic loss": critic_loss,
                }) 
            
            # save model
            if ep % 50 == 0:
                torch.save(self.actor.state_dict(), f"{self.save}/ppo_actor_{ep}.pth")
                torch.save(self.critic.state_dict(), f"{self.save}/ppo_critic_{ep}.pth")

            # save latest model
            torch.save(self.actor.state_dict(), f"{self.save}/ppo_actor_latest.pth")
            torch.save(self.critic.state_dict(), f"{self.save}/ppo_critic_latest.pth")


        # termination
        self.env.close()

    def test(self, video_folder: str, num_episodes: int = 20):
        """Test the agent for a specified number of episodes."""
        self.is_test = True

        tmp_env = self.env        
        # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda t: True)
        total_score = []

        for ep in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed+ep)
            state = np.expand_dims(state, axis=0)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
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
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=1400)
    parser.add_argument("--entropy-weight", type=int, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=int, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)  
    parser.add_argument("--update-epoch", type=float, default=10)
    parser.add_argument("--save", type=str, default="task3-6")
    args = parser.parse_args([])

    if not os.path.exists(args.save):
        os.makedirs(args.save)
 
    # environment
    env = gym.make("Walker2d-v4", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)

    # wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    # agent = PPOAgent(env, args)
    # agent.train()


    '''Load the best model'''
    agent = PPOAgent(env, args)
    agent.actor.load_state_dict(torch.load(f"LAB7_413551036_task3_ppo_2m.pth", map_location=agent.device))
    agent.test(video_folder="videos_Task3")