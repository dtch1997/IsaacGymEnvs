import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Actor(nn.Module):
#     def __init__(self, state_dim, number_of_llps):
#         super(Actor, self).__init__()
#         # actor
#         self.actor = nn.Sequential(
#             nn.Linear(state_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, number_of_llps),
#             nn.Softmax(dim=-1),
#         )
#
#     def forward(self, state):
#         return self.actor(state)
#


class Actor(nn.Module):
    def __init__(self, state_dim, number_of_llps):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_llps),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state, exploration_noise=None):
        logits = self.actor(state)

        print(exploration_noise)

        if exploration_noise is not None:
            logits = logits + exploration_noise
        action_probs = self.softmax(logits)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_dim, number_of_llps, H):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + number_of_llps, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.H = H

    def forward(self, state, action):
        # rewards are in range [-H, 0]
        return -self.critic(torch.cat([state, action], 1)) * self.H


class PPO:
    def __init__(self, state_dim, number_of_llps, lr, H, players, gamma):
        self.actor = Actor(state_dim, number_of_llps).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, number_of_llps, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.F = torch.nn.functional
        self.epsilon_clip = 0.1

        self.mseLoss = torch.nn.MSELoss()
        self.CrossEntropy = torch.nn.CrossEntropyLoss()
        self.players = players

        self.gamma = gamma

        self.running_mean_std = RunningMeanStd(state_dim)

    def select_action(self, state,noise,eps):
        # eps =eps
        # rand_num = torch.rand(1).item()
        #
        # if rand_num < eps:
        #     # Explore: Randomly pick an action
        #     random_action = torch.randint(high=7, size=(state.shape[0],7), dtype=torch.long).cuda(0)
        #     chosen_action = random_action.detach().cpu().data.numpy().flatten()
        # else:
        # Exploit: Use action from the actor network
        chosen_action = self.actor(state,noise).detach().cpu().data.numpy().flatten()

        return chosen_action

    def update(self, buffer, n_iter, batch_size):
        for i in range(n_iter):

            # Sample a batch of transitions from replay buffer:
            output_dict = buffer.sample(batch_size)

            # Unpack buffer states
            next_state = output_dict['next_obs']
            state = output_dict['obs']
            reward = output_dict['reward']
            done = output_dict['done']
            action = output_dict['action']

            hl_obs = self.norm_obs(output_dict['hl_obs'])
            hl_next_obs = self.norm_obs(output_dict['hl_next_obs'])
            hl_action = output_dict['hl_action']
            hl_last_action = output_dict['hl_last_action']

            exploration_noise = torch.normal(mean=hl_action.float(), std=0.5).cuda(0)

            # Compute advantages
            with torch.no_grad():
                next_action_llp_dist = self.actor(hl_next_obs, exploration_noise)
                next_action_llp = next_action_llp_dist.argmax(dim=-1)
                next_state_values = self.critic(hl_next_obs, next_action_llp_dist)

                target_Q = reward + ((1 - done) * self.gamma * next_state_values)
                advantages = target_Q - self.critic(hl_obs, hl_action)

            # Update the critic
            critic_loss = self.F.smooth_l1_loss(self.critic(hl_obs, hl_action), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the actor using PPO objective
            action_probs = self.actor(hl_obs, exploration_noise)

            epsilon = 1e-8
            ratio = torch.exp(action_probs.log() - (hl_last_action+epsilon).log())
            surr1 = ratio * advantages.detach()
            surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages.detach()
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        return critic_loss, actor_loss


    def norm_obs(self, observation):

        with torch.no_grad():
            norm_obs = self.running_mean_std(observation)
            clamped_obs = torch.clamp(norm_obs,-1.0,1.0)
            return clamped_obs

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, name))
        torch.save(self.critic.state_dict(), "%s/%s_crtic.pth" % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, name), map_location="cpu")
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_crtic.pth" % (directory, name), map_location="cpu")
        )
        return self.actor, self.critic
