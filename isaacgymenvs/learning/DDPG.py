import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
            nn.Softmax(),
        )

    def forward(self, state):
        return self.actor(state)


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


class DDPG:
    def __init__(self, state_dim, number_of_llps, lr, H, players, gamma):
        self.actor = Actor(state_dim, number_of_llps).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, number_of_llps, H).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()

        self.players = players

        self.gamma = gamma

    def select_action(self, state):
        # state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu().data.numpy().flatten()

    def update(self, buffer, n_iter, batch_size):
        for i in range(n_iter):

            # Sample a batch of transitions from replay buffer:
            output_dict = buffer.sample(batch_size)

            #unpack buffer states
            next_state = output_dict['next_obs']
            state = output_dict['obs']
            reward = output_dict['reward']
            done = output_dict['done']
            action = output_dict['action']

            hl_obs = output_dict['hl_obs']
            hl_next_obs = output_dict['hl_next_obs']
            hl_action = output_dict['hl_action']
            hl_last_action = output_dict ['hl_last_action']


            # select next action
            with torch.no_grad():
                next_action_llp_dist = self.actor(hl_next_obs)
                next_action_llp = next_action_llp_dist.argmax(dim=-1)


            # Compute target Q-value:
            target_Q = self.critic(hl_next_obs, next_action_llp_dist)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

            # Optimize Critic:
            critic_loss = self.mseLoss(self.critic(hl_obs, hl_action), target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss:
            action_llp_dist = self.actor(hl_obs)
            action_llp = action_llp_dist.argmax(dim=1)
            actor_loss = -self.critic(hl_obs, action_llp_dist).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()





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
