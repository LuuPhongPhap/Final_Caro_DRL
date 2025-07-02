import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *

# Hyperparameters
BOARD_SIZE = 10
WIN_LEN = 5
EPISODES = 1000
GAMMA = 0.99
LR = 1e-4
ENTROPY_BETA = 0.005
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment with enhanced reward shaping
class CaroEnv:
    def __init__(self, size=BOARD_SIZE, win_len=WIN_LEN):
        self.size = size
        self.win_len = win_len
        self.board = np.zeros((size, size), dtype=np.int8)
        self.current_player = 1

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        return self._get_image()

    def _get_image(self):
        img = np.zeros((self.size, self.size), dtype=np.uint8)
        img[self.board == 1] = 255
        img[self.board == -1] = 127
        img = cv2.resize(img, (224, 224))
        return img.astype(np.float32) / 255.0

    def step(self, action):
        x, y = divmod(action, self.size)
        reward = 0.0
        done = False
        if not (0 <= x < self.size and 0 <= y < self.size) or self.board[x, y] != 0:
            return self._get_image(), -10.0, True, {}
        reward += 0.2
        self.board[x, y] = self.current_player
        if self._check_win(x, y):
            return self._get_image(), reward + 10.0, True, {}
        if self._blocked_opponent_four(x, y):
            reward += 5.0
        chain_len = self._longest_chain(self.current_player)
        reward += (chain_len ** 2) * 0.02
        reward += self._count_open_threes(self.current_player) * 0.1
        reward -= 0.005
        self.current_player *= -1
        return self._get_image(), reward, done, {}

    def _check_win(self, x, y):
        dirs = [(1,0), (0,1), (1,1), (1,-1)]
        for dx, dy in dirs:
            count = 1
            for sign in [1, -1]:
                nx, ny = x + sign*dx, y + sign*dy
                while 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == self.current_player:
                    count += 1; nx += sign*dx; ny += sign*dy
            if count >= self.win_len:
                return True
        return False

    def _blocked_opponent_four(self, x, y):
        opp = -self.current_player
        for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
            seq = []
            for i in range(-4,5):
                nx, ny = x + i*dx, y + i*dy
                seq.append(self.board[nx, ny] if 0<=nx<self.size and 0<=ny<self.size else None)
            for i in range(len(seq)-4):
                window = seq[i:i+5]
                if window.count(opp)==4 and window.count(0)==1:
                    return True
        return False

    def _longest_chain(self, player):
        max_len = 0
        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] == player:
                    for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
                        length=1; nx, ny = x+dx, y+dy
                        while 0<=nx<self.size and 0<=ny<self.size and self.board[nx,ny]==player:
                            length+=1; nx+=dx; ny+=dy
                        max_len=max(max_len, length)
        return max_len

    def _count_open_threes(self, player):
        count=0; opp=-player
        for dx, dy in [(1,0),(0,1),(1,1),(1,-1)]:
            for x in range(self.size):
                for y in range(self.size):
                    seq=[]
                    for i in range(-1,4):
                        nx, ny = x+i*dx, y+i*dy
                        seq.append(self.board[nx,ny] if 0<=nx<self.size and 0<=ny<self.size else None)
                    if seq[0]==0 and seq[-1]==0 and seq[1:4].count(player)==3:
                        count+=1
        return count

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * (input_shape[1] // 4) * (input_shape[2] // 4)
        self.fc = nn.Linear(conv_out, 256)
        self.policy = nn.Linear(256, n_actions)
        self.value = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.fc(x))
        return self.policy(x), self.value(x)

def train():
    print(f"Using device: {DEVICE}")
    env = CaroEnv()
    model = ActorCritic((1,224,224), BOARD_SIZE*BOARD_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

    rewards_history, loss_history = [], []
    for ep in range(1, EPISODES+1):
        obs = torch.tensor(env.reset(), dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)
        log_probs, values, rets, entropies = [], [], [], []
        done = False
        while not done:
            logits, val = model(obs)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            act = dist.sample()
            log_probs.append(dist.log_prob(act))
            entropies.append(dist.entropy())
            values.append(val)
            img, r, done, _ = env.step(act.item())
            rets.append(torch.tensor([r], device=DEVICE))
            obs = torch.tensor(img, dtype=torch.float32, device=DEVICE).unsqueeze(0).unsqueeze(0)

        R = torch.zeros(1,1,device=DEVICE); advs = []
        for i in reversed(range(len(rets))):
            R = rets[i] + GAMMA * R
            advs.insert(0, R - values[i])
        advs = torch.cat(advs)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        policy_loss = 0; value_loss = 0
        for lp, adv, ent, ret, val in zip(log_probs, advs, entropies, rets, values):
            policy_loss -= lp * adv + ENTROPY_BETA * ent
            value_loss += VALUE_LOSS_COEF * (ret - val).pow(2)
        loss = policy_loss + value_loss

        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step(); scheduler.step()

        total_r = float(sum(r.item() for r in rets))
        rewards_history.append(total_r); loss_history.append(float(loss))
        print(f"Ep {ep}/{EPISODES} | R: {total_r:.2f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    plt.figure(); plt.plot(rewards_history); plt.title('Rewards'); plt.savefig('rewards.png')
    plt.figure(); plt.plot(loss_history); plt.title('Loss'); plt.savefig('loss.png')

if __name__ == '__main__':
    train()