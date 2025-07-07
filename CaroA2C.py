import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import cv2
import pandas as pd

# === Hyperparameters ===
BOARD_SIZE       = 10
WIN_LEN          = 5
EPISODES         = 2000
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
LR               = 0.005
ENTROPY_COEF     = 0.01
ENTROPY_END      = 1e-4
VALUE_COEF       = 0.5
MAX_GRAD_NORM    = 0.5
PRINT_FREQ       = 50
MA_WINDOW        = 100
CURRICULUM_WIN   = 10
CURRICULUM_THRESH= 0.8
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Environment ===
CELL_SIZE    = 40
SCREEN_SIZE  = BOARD_SIZE * CELL_SIZE

class CaroEnv:
    def __init__(self, opponent='random'):
        pygame.init()
        self.screen   = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        self.opponent= opponent
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), int)
        self.current_player = 1
        self.done = False
        self.last_opp = (0,0)
        return self._get_image()

    def _get_image(self):
        self.screen.fill((255,255,255))
        # draw grid
        for i in range(BOARD_SIZE+1):
            pygame.draw.line(self.screen,(0,0,0),(i*CELL_SIZE,0),(i*CELL_SIZE,SCREEN_SIZE))
            pygame.draw.line(self.screen,(0,0,0),(0,i*CELL_SIZE),(SCREEN_SIZE,i*CELL_SIZE))
        # draw stones
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y,x]!=0:
                    color = (0,0,0) if self.board[y,x]==1 else (255,0,0)
                    pygame.draw.circle(self.screen,color,(x*CELL_SIZE+CELL_SIZE//2,y*CELL_SIZE+CELL_SIZE//2),15)
        img = pygame.surfarray.array3d(self.screen)
        img = cv2.resize(np.transpose(img,(1,0,2)),(84,84))
        return img.astype(np.float32)/255.0

    def step(self, action):
        x,y = action % BOARD_SIZE, action // BOARD_SIZE
        # invalid move
        if self.board[y,x] != 0:
            return self._get_image(), -1.0, False, {}
        reward = -0.05
        # agent move
        self.board[y,x] = 1
        # win check
        if self._current_chain(x,y,1) >= WIN_LEN:
            self.done = True
            return self._get_image(), 10.0, True, {}
        # block rewards
        if self._check_block(x,y,4): reward += 5.0
        elif self._check_block(x,y,3): reward += 2.0
        # draw
        if np.all(self.board != 0):
            self.done = True
            return self._get_image(), reward, True, {}
        # opponent move
        self._opp_move()
        ox,oy = self.last_opp
        if self._current_chain(ox,oy,-1) >= WIN_LEN:
            self.done = True
            return self._get_image(), reward - 10.0, True, {}
        return self._get_image(), reward, False, {}

    def _opp_move(self):
        opp = -1
        # heuristic block
        if self.opponent=='heuristic':
            for L in (4,3):
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        if self.board[y,x]==0 and self._check_block(x,y,L):
                            self.board[y,x]=opp
                            self.last_opp=(x,y)
                            return
        # random
        empties = list(zip(*np.where(self.board==0)))
        y,x = empties[np.random.randint(len(empties))]
        self.board[y,x]=opp
        self.last_opp=(x,y)

    def _check_line(self,x,y,dx,dy,L,p):
        cnt=0
        for o in range(-L+1, L):
            nx,ny = x+o*dx, y+o*dy
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and self.board[ny,nx]==p:
                cnt+=1
                if cnt>=L: return True
            else:
                cnt=0
        return False

    def _check_block(self,x,y,L):
        return any(self._check_line(x,y,dx,dy,L,-1) for dx,dy in [(1,0),(0,1),(1,1),(1,-1)])

    def _current_chain(self,x,y,p):
        m=1
        for dx,dy in [(1,0),(0,1),(1,1),(1,-1)]:
            c=1; i=1
            while 0<=x+i*dx<BOARD_SIZE and 0<=y+i*dy<BOARD_SIZE and self.board[y+i*dy,x+i*dx]==p:
                c+=1; i+=1
            i=1
            while 0<=x-i*dx<BOARD_SIZE and 0<=y-i*dy<BOARD_SIZE and self.board[y-i*dy,x-i*dx]==p:
                c+=1; i+=1
            m=max(m,c)
        return m

# === A2C Network & Agent ===
class A2CNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,2), nn.ReLU(),
            nn.Conv2d(32,64,3,2), nn.ReLU(),
            nn.Conv2d(64,64,3,2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1,*input_shape)).shape[1]
        self.fc_pi = nn.Sequential(nn.Linear(out,256), nn.ReLU(), nn.Linear(256,n_actions))
        self.fc_v  = nn.Sequential(nn.Linear(out,256), nn.ReLU(), nn.Linear(256,1))

    def forward(self,x):
        feat = self.conv(x)
        return self.fc_pi(feat), self.fc_v(feat)

class A2CAgent:
    def __init__(self, input_shape, n_actions):
        self.net = A2CNet(input_shape, n_actions).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)

    def select(self, state, mask):
        st = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(DEVICE)
        logits, val = self.net(st)
        logits = logits.masked_fill(mask.to(DEVICE), -1e9)
        dist = Categorical(F.softmax(logits, dim=-1))
        a = dist.sample()
        return a.item(), dist.log_prob(a), dist.entropy(), val.squeeze()

    def compute_gae(self, rewards, values, dones, next_val):
        values = values + [next_val]
        gae = 0; returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA*values[i+1]*(1-dones[i]) - values[i]
            gae = delta + GAMMA*GAE_LAMBDA*(1-dones[i])*gae
            returns.insert(0, gae + values[i])
        return returns

    def update(self, memories, next_val):
        rws, lps, vals, ents, dns = memories
        returns = self.compute_gae(rws, vals, dns, next_val)
        returns = torch.tensor(returns, device=DEVICE)
        logps = torch.stack(lps)
        vals   = torch.stack(vals)
        ents   = torch.stack(ents)
        advs   = (returns - vals.squeeze()).detach()

        p_loss = -(logps * advs).mean() - ENTROPY_COEF*ents.mean()
        v_loss = VALUE_COEF * advs.pow(2).mean()
        loss = p_loss + v_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
        self.opt.step()
        return p_loss.item(), v_loss.item()

# === Training ===
env    = CaroEnv()
agent  = A2CAgent((3,84,84), BOARD_SIZE*BOARD_SIZE)
win_buf= []
all_rewards, p_losses, v_losses = [], [], []

for ep in range(1, EPISODES+1):
    state = env.reset(); done=False; total_r=0
    memories = ([],[],[],[],[])

    while not done:
        mask = torch.tensor(env.board.reshape(-1)!=0)
        a, lp, ent, val = agent.select(state, mask)
        nxt, r, done, _ = env.step(a)
        for buf, x in zip(memories, (r, lp, val, ent, done)):
            buf.append(x)
        state = nxt; total_r += r

    # curriculum switch
    win_buf.append(1 if total_r>0 else 0)
    if len(win_buf)>=CURRICULUM_WIN and np.mean(win_buf[-CURRICULUM_WIN:])>=CURRICULUM_THRESH:
        env.opponent = 'heuristic'

    # last value = 0 when done
    next_val = 0.0
    p_l, v_l = agent.update(memories, next_val)

    all_rewards.append(total_r)
    p_losses.append(p_l)
    v_losses.append(v_l)

    if ep % PRINT_FREQ == 0:
        ma = np.mean(all_rewards[-MA_WINDOW:])
        print(f"Ep {ep}/{EPISODES} | MA{MA_WINDOW} Reward: {ma:.2f} | P_loss: {np.mean(p_losses[-PRINT_FREQ:]):.3f} | V_loss: {np.mean(v_losses[-PRINT_FREQ:]):.3f}")

# Plot reward curve
ma = pd.Series(all_rewards).rolling(MA_WINDOW).mean()
plt.figure(figsize=(10,6))
plt.plot(ma, linewidth=2)
plt.title(f"A2C Reward (MA{MA_WINDOW})")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)
plt.show()
