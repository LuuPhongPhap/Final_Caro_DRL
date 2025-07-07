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
BOARD_SIZE      = 10
WIN_LEN         = 5
EPISODES        = 2000
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
LR              = 0.001
CLIP_START      = 0.2
CLIP_END        = 0.05
ENTROPY_COEF    = 0.01
ENTROPY_END     = 1e-4
VALUE_COEF      = 0.5
KL_COEF         = 0.5
KL_THRESHOLD    = 0.01
MAX_GRAD_NORM   = 0.5
UPDATE_EPOCHS   = 4
BATCH_SIZE      = 64
PRINT_FREQ      = 50
MA_WINDOW       = 100
CURRICULUM_WIN  = 10
CURRICULUM_THRESH = 0.8
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CaroEnv with opponent mode ===
CELL_SIZE   = 40
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE

class CaroEnv:
    def __init__(self, opponent='random'):
        pygame.init()
        self.screen   = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE))
        self.opponent = opponent
        self.reset()

    def reset(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), int)
        self.player = 1
        self.done = False
        return self._get_image()

    def _get_image(self):
        self.screen.fill((255,255,255))
        for i in range(BOARD_SIZE+1):
            pygame.draw.line(self.screen,(0,0,0),(i*CELL_SIZE,0),(i*CELL_SIZE,SCREEN_SIZE))
            pygame.draw.line(self.screen,(0,0,0),(0,i*CELL_SIZE),(SCREEN_SIZE,i*CELL_SIZE))
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if self.board[y,x]!=0:
                    color = (0,0,0) if self.board[y,x]==1 else (255,0,0)
                    pygame.draw.circle(self.screen,color,(x*CELL_SIZE+20,y*CELL_SIZE+20),15)
        img = pygame.surfarray.array3d(self.screen)
        img = cv2.resize(np.transpose(img,(1,0,2)),(84,84))
        return img.astype(np.float32)/255.0

    def step(self,action):
        x,y = action % BOARD_SIZE, action // BOARD_SIZE
        # Agent move
        if self.board[y,x]!=0:
            reward = -1.0
        else:
            self.board[y,x]=1
            reward = -0.05
            if self._check_win(x,y):
                self.done=True
                return self._get_image(),10.0,True,{}
            # block reward
            if self._check_block(x,y,4): reward+=5
            elif self._check_block(x,y,3): reward+=2
        # Opponent move
        if not self.done:
            self._opp_move()
            ox,oy = self.last_opp
            if self._check_win(ox,oy):
                reward -= 10.0
                self.done = True
        return self._get_image(), reward, self.done, {}

    def _opp_move(self):
        opp=-1; self.last_opp=(0,0)
        # heuristic if enabled
        if self.opponent=='heuristic':
            for length in (4,3):
                for y in range(BOARD_SIZE):
                    for x in range(BOARD_SIZE):
                        if self.board[y,x]==0 and self._check_block(x,y,length):
                            self.board[y,x]=opp
                            self.last_opp=(x,y)
                            return
        # random fallback
        empties = list(zip(*np.where(self.board==0)))
        y,x = empties[np.random.randint(len(empties))]
        self.board[y,x]=opp
        self.last_opp=(x,y)

    def _check_line(self,x,y,dx,dy,L,p):
        cnt=0
        for o in range(-L+1,L):
            nx,ny = x+o*dx, y+o*dy
            if 0<=nx<BOARD_SIZE and 0<=ny<BOARD_SIZE and self.board[ny,nx]==p:
                cnt+=1
                if cnt>=L: return True
            else: cnt=0
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

    def _check_win(self,x,y):
        return self._current_chain(x,y,self.board[y,x])>=WIN_LEN

# === PPO Network & Agent ===
class PPOPolicy(nn.Module):
    def __init__(self,input_shape,n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,3,2), nn.ReLU(),
            nn.Conv2d(32,64,3,2), nn.ReLU(),
            nn.Conv2d(64,64,3,2), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            out = self.conv(torch.zeros(1,*input_shape)).shape[1]
        self.fc_pi = nn.Sequential(nn.Linear(out,256),nn.ReLU(),nn.Linear(256,n_actions))
        self.fc_v  = nn.Sequential(nn.Linear(out,256),nn.ReLU(),nn.Linear(256,1))
        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)):
                nn.init.orthogonal_(m.weight,np.sqrt(2)); nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.conv(x)
        return self.fc_pi(x), self.fc_v(x)

class PPOAgent:
    def __init__(self, input_shape, n_actions):
        self.net = PPOPolicy(input_shape, n_actions).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.logps = []
        self.values = []
        self.rews = []
        self.dones = []

    def select(self, state, mask):
        st = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(DEVICE)
        logits, val = self.net(st)
        logits = logits.masked_fill(mask.to(DEVICE), -1e9)
        dist = Categorical(F.softmax(logits, dim=-1))
        a = dist.sample()

        # **detach** ngay khi lưu vào memory
        self.states.append(st.detach())
        self.actions.append(a.detach())
        self.logps.append(dist.log_prob(a).detach())
        self.values.append(val.squeeze().detach())
        return a.item()

    def store(self, r, d):
        self.rews.append(r)
        self.dones.append(d)

    def compute_gae(self, next_val):
        values = self.values + [next_val]
        gae = 0
        returns = []
        for i in reversed(range(len(self.rews))):
            delta = self.rews[i] + GAMMA * values[i+1] * (1 - self.dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - self.dones[i]) * gae
            returns.insert(0, gae + values[i])
        return torch.stack(returns).to(DEVICE)

    def update(self, last_state, ep):
        # Lấy giá trị V(s_{T})
        with torch.no_grad():
            _, next_val = self.net(last_state)

        returns = self.compute_gae(next_val.squeeze())
        values  = torch.stack(self.values)
        advs    = returns - values
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)

        S = torch.cat(self.states)
        A = torch.stack(self.actions)
        oldLP = torch.stack(self.logps)
        N = len(returns)

        clip_eps = CLIP_START - (CLIP_START-CLIP_END)*(ep/EPISODES)
        ent_coef = ENTROPY_COEF - (ENTROPY_COEF-ENTROPY_END)*(ep/EPISODES)

        for _ in range(UPDATE_EPOCHS):
            idxs = np.random.permutation(N)
            for i in range(0, N, BATCH_SIZE):
                b = idxs[i:i+BATCH_SIZE]
                bs, ba = S[b], A[b]
                br, ba_adv, bo_lp = returns[b], advs[b], oldLP[b]

                logits, vals = self.net(bs)
                dist = Categorical(F.softmax(logits, dim=-1))
                new_lp = dist.log_prob(ba)
                ent    = dist.entropy().mean()

                # Tính KL-penalty bằng approx log-probs
                old_p = bo_lp.exp()
                new_p = new_lp.exp()
                kl    = (old_p * (bo_lp - new_lp)).mean()
                kl_pen = KL_COEF * max(0, kl.item() - KL_THRESHOLD)

                # PPO losses
                ratio = (new_lp - bo_lp).exp()
                s1 = ratio * ba_adv
                s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * ba_adv
                actor_loss  = -torch.min(s1, s2).mean()
                critic_loss = F.mse_loss(vals.squeeze(), br)

                loss = actor_loss + VALUE_COEF * critic_loss - ent_coef * ent + kl_pen

                # **Zero gradients trước, backward rồi step**
                self.opt.zero_grad()
                loss.backward()  # retain_graph=False
                nn.utils.clip_grad_norm_(self.net.parameters(), MAX_GRAD_NORM)
                self.opt.step()

        self.clear()

# === Training ===
env = CaroEnv()
agent = PPOAgent((3,84,84), BOARD_SIZE*BOARD_SIZE)
win_buf = []
all_rewards = []

for ep in range(1, EPISODES+1):
    state = env.reset(); done=False; total_r=0
    while not done:
        mask = torch.tensor(env.board.reshape(-1)!=0)
        a = agent.select(state,mask)
        nxt,r,done,_ = env.step(a)
        agent.store(r,done)
        state = nxt; total_r+=r

    # curriculum
    win_buf.append(1 if total_r>0 else 0)
    if len(win_buf)>=CURRICULUM_WIN and np.mean(win_buf[-CURRICULUM_WIN:])>=CURRICULUM_THRESH:
        env.opponent = 'heuristic'

    last_s = torch.FloatTensor(state).permute(2,0,1).unsqueeze(0).to(DEVICE)
    agent.update(last_s, ep)
    all_rewards.append(total_r)

    if ep % PRINT_FREQ == 0:
        ma = pd.Series(all_rewards).rolling(MA_WINDOW).mean().iloc[-1]
        print(f"Ep {ep}/{EPISODES} | MA{MA_WINDOW} Reward: {ma:.2f}")

# Plot
ma = pd.Series(all_rewards).rolling(MA_WINDOW).mean()
plt.figure(figsize=(10,6))
plt.plot(ma,linewidth=2)
plt.title(f"PPO Reward (MA{MA_WINDOW})")
plt.xlabel("Episode"); plt.ylabel("Reward"); plt.grid(True)
plt.show()
