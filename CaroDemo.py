import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import time
import matplotlib.pyplot as plt

# --- Gomoku Environment with Pygame ---
class GomokuEnv:
    def __init__(self, size=10, win_len=5, cell_size=40, delay=0.3):
        pygame.init()
        self.size = size
        self.win_len = win_len
        self.cell_size = cell_size
        self.delay = delay
        self.window_size = self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Caro DQN Agent Training")
        self.font = pygame.font.SysFont(None, self.cell_size - 10)
        self.reset()

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.done = False
        self._draw_board()
        return self._get_state()

    def step(self, action):
        x, y = divmod(action, self.size)
        if self.board[x, y] != 0 or self.done:
            return self._get_state(), -10, True, {}
        self.board[x, y] = self.current_player
        reward, self.done = self._check_game(x, y)
        
        # Adjust reward
        if self.done:
            if reward == 1.0:
                reward = 10 if self.current_player == 1 else -10
            elif reward == 0.5:
                reward = 5
            else:
                reward = -10
        else:
            reward = -0.01  # penalty for each move

        self.current_player *= -1
        self._draw_board()
        time.sleep(self.delay)
        return self._get_state(), reward, self.done, {}

    def legal_actions(self):
        return [i for i in range(self.size * self.size) if self.board.flat[i] == 0]

    def _get_state(self):
        return self.board.flatten().astype(np.float32)

    def _check_game(self, x, y):
        player = self.board[x, y]
        dirs = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dx, dy in dirs:
            count = 1
            for sign in [1, -1]:
                nx, ny = x, y
                while True:
                    nx += dx * sign
                    ny += dy * sign
                    if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == player:
                        count += 1
                    else:
                        break
            if count >= self.win_len:
                return 1.0, True
        if np.all(self.board != 0):
            return 0.5, True
        return 0.0, False

    def _draw_board(self):
        self.screen.fill((230, 185, 139))
        for i in range(self.size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (i * self.cell_size, 0), (i * self.cell_size, self.window_size))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * self.cell_size), (self.window_size, i * self.cell_size))
        for x in range(self.size):
            for y in range(self.size):
                val = self.board[x, y]
                if val != 0:
                    text = 'X' if val == 1 else 'O'
                    color = (255, 0, 0) if val == 1 else (0, 0, 255)
                    img = self.font.render(text, True, color)
                    rect = img.get_rect(center=(y * self.cell_size + self.cell_size // 2,
                                                x * self.cell_size + self.cell_size // 2))
                    self.screen.blit(img, rect)
        pygame.display.flip()

    def close(self):
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# --- DQN Agent ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.e_min = 0.1
        self.e_decay = 0.9995
        self.batch_size = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.update_target_every = 1000
        self.steps = 0

    def act(self, state, legal_actions):
        if random.random() < self.epsilon:
            return random.choice(legal_actions)
        state_v = torch.tensor([state], device=self.device, dtype=torch.float32)
        q_vals = self.policy_net(state_v).detach().cpu().numpy()[0]
        mask = np.full(q_vals.shape, -np.inf)
        mask[legal_actions] = 0
        return int(np.argmax(q_vals + mask))

    def remember(self, s, a, r, s2, d):
        self.memory.append((s, a, r, s2, d))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        states_v = torch.tensor(states, device=self.device, dtype=torch.float32)
        next_v = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        actions_v = torch.tensor(actions, device=self.device, dtype=torch.int64)
        rewards_v = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.bool, device=self.device)

        q_vals = self.policy_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self.target_net(next_v).max(dim=1)[0]
            q_next[dones_t] = 0.0
        target = rewards_v + self.gamma * q_next
        loss = nn.MSELoss()(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.e_min, self.epsilon * self.e_decay)
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps += 1


# --- Training Loop ---
def train(agent, env, episodes=1000, display_every=100):
    stats = {'wins': 0, 'loses': 0, 'draws': 0}
    win_hist, lose_hist, draw_hist, ep_idx = [], [], [], []
    running = True

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        while not done and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            legal = env.legal_actions()
            action = agent.act(state, legal)
            next_state, reward, done, _ = env.step(action)

            if not done:
                opp = random.choice(env.legal_actions())
                _, opp_r, done, _ = env.step(opp)
                reward -= opp_r

            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state

        if not running:
            break

        if reward > 1:
            stats['wins'] += 1
        elif reward < 0:
            stats['loses'] += 1
        else:
            stats['draws'] += 1

        print(f"Episode {ep}: Reward={reward:.2f}, Wins={stats['wins']}, Loses={stats['loses']}, Draws={stats['draws']}")

        if ep % display_every == 0:
            ep_idx.append(ep)
            win_hist.append(stats['wins'])
            lose_hist.append(stats['loses'])
            draw_hist.append(stats['draws'])

    plt.figure()
    plt.plot(ep_idx, win_hist, label='Wins')
    plt.plot(ep_idx, lose_hist, label='Loses')
    plt.plot(ep_idx, draw_hist, label='Draws')
    plt.xlabel('Episode')
    plt.ylabel('Count')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    env.close()
    return stats


if __name__ == '__main__':
    SIZE = 10
    env = GomokuEnv(size=SIZE, delay=0.3)
    agent = DQNAgent(state_dim=SIZE * SIZE, action_dim=SIZE * SIZE)
    results = train(agent, env, episodes=2000, display_every=200)
    print("Training complete.", results)
