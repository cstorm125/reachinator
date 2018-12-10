from collections import namedtuple
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils import OUNoise
from network import Actor, Critic

class Agent():        
    def __init__(self, 
        state_size, action_size, replay_memory, random_seed=0, nb_agent = 20, bs = 128,
        gamma=0.99, tau=1e-3, lr_actor=1e-4, lr_critic=1e-4, wd_actor=0, wd_critic=0,
        clip_actor = None, clip_critic=None, update_interval = 20, update_times = 10): 

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.nb_agent = nb_agent
        self.bs = bs
        self.update_interval = update_interval
        self.update_times = update_times
        self.timestep = 0

        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.wd_critic = wd_critic
        self.wd_actor = wd_actor
        self.clip_critic=clip_critic
        self.clip_actor = clip_actor
        self.actor_losses = []
        self.critic_losses = []

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor,weight_decay=self.wd_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic,weight_decay=self.wd_critic)

        # Noise process
        self.noise = OUNoise((self.nb_agent, action_size), random_seed)

        # Replay memory
        self.memory = replay_memory
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        #increment timestep
        self.timestep+=1
        
        # Save experience / reward
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory  
        if self.timestep % self.update_interval == 0:
            for i in range(self.update_times):
                if len(self.memory) > self.bs:
                    experiences = self.memory.sample(self.bs)
                    self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()        
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset_noise(self):
        self.noise.reset()

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.clip_critic: torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), self.clip_critic)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.clip_actor: torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), self.clip_actor)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)   
           
        self.actor_losses.append(actor_loss.cpu().data.numpy())
        self.critic_losses.append(critic_loss.cpu().data.numpy())        

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)