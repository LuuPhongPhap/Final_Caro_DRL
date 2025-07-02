import os
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --- Hyperparameters ---
BOARD_SIZE = 10
WIN_LEN = 5
EPISODES = 2000
MAX_STEPS = BOARD_SIZE * BOARD_SIZE
LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPS_CLIP = 0.2
K_EPOCH = 4
STEP_PENALTY = -0.05        # stronger step penalty
BLOCK_REWARD = 3
CHAIN3_REWARD = 1          # intermediate reward for creating or blocking chain-3
WIN_REWARD = 10
LOSE_PENALTY = -10
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02        # increased entropy bonus
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Game Environment ---
class CaroEnv:
    def __init__(self, size=BOARD_SIZE, win_len=WIN_LEN, render=False):
        self.size = size
        self.win_len = win_len
        self.render = render
        if render:
            pygame.init()
            self.cell_size = 50
            self.screen = pygame.display.set_mode((size*self.cell_size, size*self.cell_size))
            pygame.display.set_caption('Caro PPO')
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        return self._get_state()

    def step(self, action):
        r, c = divmod(action, self.size)
        if self.board[r, c] != 0:
            return self._get_state(), 0.0, False, {}
        reward = STEP_PENALTY
        self.board[r, c] = self.current_player
        if self._check_win(r, c, self.current_player):
            reward += WIN_REWARD
            self.done = True
            return self._get_state(), reward, True, {}
        opp = 3 - self.current_player
        if self._check_block(r, c, opp):
            reward += BLOCK_REWARD
        if self._check_chain(r, c, self.current_player, 3):
            reward += CHAIN3_REWARD
        if self._check_chain(r, c, opp, 3):
            reward += CHAIN3_REWARD
        self.current_player = opp
        return self._get_state(), reward, False, {}

    def _get_state(self):
        state = np.zeros((self.size, self.size, 3), dtype=np.float32)
        state[..., 0][self.board == 1] = 1.0
        state[..., 1][self.board == 2] = 1.0
        state[..., 2][self.board == 0] = 1.0
        return state

    def _count_dir(self, r, c, dr, dc, player):
        cnt = 0
        i, j = r + dr, c + dc
        while 0 <= i < self.size and 0 <= j < self.size and self.board[i, j] == player:
            cnt += 1; i += dr; j += dc
        return cnt

    def _check_win(self, r, c, player):
        for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
            if 1 + self._count_dir(r,c,dr,dc,player) + self._count_dir(r,c,-dr,-dc,player) >= self.win_len:
                return True
        return False

    def _check_block(self, r, c, player):
        for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
            if self._count_dir(r,c,dr,dc,player) + self._count_dir(r,c,-dr,-dc,player) >= self.win_len-1:
                return True
        return False

    def _check_chain(self, r, c, player, length):
        for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
            if 1 + self._count_dir(r,c,dr,dc,player) + self._count_dir(r,c,-dr,-dc,player) >= length:
                return True
        return False

# --- PPO Agent ---
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim[0], 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 128 * state_dim[1] * state_dim[2]
        self.fc_pi = nn.Sequential(nn.Linear(conv_out, 1024), nn.ReLU(), nn.Linear(1024, action_dim), nn.Softmax(dim=-1))
        self.fc_v  = nn.Sequential(nn.Linear(conv_out, 1024), nn.ReLU(), nn.Linear(1024, 1))

    def forward(self, x):
        x = self.conv(x)
        return self.fc_pi(x), self.fc_v(x)

class PPO:
    def __init__(self, state_dim, action_dim, total_episodes=EPISODES):
        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda ep: max(0.0, 1 - ep/total_episodes)
        )

    def select_action(self, state, memory):
        state_t = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(DEVICE)
        probs, value = self.policy(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        # store data for update (detach to free graph)
        memory.states.append(state_t)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action).detach())
        memory.values.append(value.squeeze().detach())
        return action.item()

    def update(self, memory, episode_num=None):
        # compute GAE advantages
        rewards, dones = memory.rewards, memory.dones
        values = memory.values + [torch.tensor(0.0).to(DEVICE)]
        gae = 0; advantages = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1-dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1-dones[i]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages).to(DEVICE)
        returns = advantages + torch.stack(memory.values)
        returns = (returns - returns.mean()) / (returns.std()+1e-5)

        old_states = torch.cat(memory.states)
        old_actions = torch.stack(memory.actions)
        old_log = torch.stack(memory.logprobs)

        losses = []
        for _ in range(K_EPOCH):
            probs, vals = self.policy(old_states)
            dist = torch.distributions.Categorical(probs)
            new_log = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()

            ratios = (new_log - old_log.detach()).exp()
            s1 = ratios * advantages
            s2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            policy_loss = -torch.min(s1, s2).mean()
            value_loss = (returns - vals.squeeze()).pow(2).mean()
            loss = policy_loss + VALUE_COEF*value_loss - ENTROPY_COEF*entropy
            losses.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if episode_num is not None:
            self.scheduler.step()
        memory.clear()
        return np.mean(losses)

class Memory:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.rewards, self.values, self.entropies, self.dones = [], [], [], [], [], [], []
    def clear(self):
        self.__init__()

# --- Training Loop ---
def train():
    env = CaroEnv(render=False)
    ppo = PPO((3, BOARD_SIZE, BOARD_SIZE), BOARD_SIZE*BOARD_SIZE)
    mem = Memory()
    all_r, avg_r, all_l, avg_l = [], [], [], []

    for ep in range(1, EPISODES+1):
        state = env.reset()
        ep_r = 0
        for _ in range(MAX_STEPS):
            act = ppo.select_action(state, mem)
            state, r, done, _ = env.step(act)
            mem.rewards.append(r)
            mem.dones.append(done)
            ep_r += r
            if done: break
        l = ppo.update(mem, episode_num=ep)
        all_r.append(ep_r); all_l.append(l)
        avg_r.append(np.mean(all_r[-100:])); avg_l.append(np.mean(all_l[-100:]))
        print(f"Episode {ep}\tReward: {ep_r:.2f}\tLoss: {l:.4f}")

    plt.figure(); plt.plot(all_r, label='Reward'); plt.legend(); plt.show()
    plt.figure(); plt.plot(all_l, label='Loss'); plt.legend(); plt.show()

if __name__ == '__main__':
    train()
