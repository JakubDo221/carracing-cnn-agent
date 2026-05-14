import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from matplotlib import animation
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# CONFIG
SEED = 42
MAX_STEPS = 500        # krótszy epizod
FRAME_SKIP = 4         # liczymy co 4 klatki
POP_SIZE = 20
ELITE_FRAC = 0.1
N_ITER = 80
SIGMA = 0.5
DEVICE = torch.device("cpu")  # CPU często szybciej dla małych sieci

np.random.seed(SEED)
torch.manual_seed(SEED)

# PREPROCESSING
def preprocess(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84))
    obs = obs / 255.0
    return obs

# CNN POLICY NETWORK (mniejsza)

class CNNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 9 * 9, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        steering = torch.tanh(x[:, 0])
        gas = torch.sigmoid(x[:, 1])
        brake = torch.sigmoid(x[:, 2])
        return torch.stack([steering, gas, brake], dim=1)

agent = CNNAgent().to(DEVICE)

# CEM HELPERS
def get_weights_dim(model):
    return sum(p.numel() for p in model.parameters())

def set_weights(model, weights):
    idx = 0
    for p in model.parameters():
        size = p.numel()
        p.data = torch.tensor(weights[idx:idx+size], dtype=torch.float32, device=DEVICE).view(p.shape)
        idx += size

def evaluate(weights, model, env):
    set_weights(model, weights)
    obs, _ = env.reset(seed=SEED)
    total_reward = 0.0

    for _ in range(MAX_STEPS):
        obs_p = preprocess(obs)
        obs_t = torch.tensor(obs_p).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            action = model(obs_t).cpu().numpy()[0]

        # FRAME SKIP
        reward_sum = 0.0
        for _ in range(FRAME_SKIP):
            obs, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break

        total_reward += reward_sum
        if terminated or truncated:
            break

    return total_reward

# CEM TRAINING
def cem():
    env = gym.make("CarRacing-v3", continuous=True)
    n_elite = max(1, int(POP_SIZE * ELITE_FRAC))
    dim = get_weights_dim(agent)
    mean = np.zeros(dim)
    scores = []

    for i in range(1, N_ITER+1):
        population = [mean + SIGMA * np.random.randn(dim) for _ in range(POP_SIZE)]
        rewards = np.array([evaluate(w, agent, env) for w in population])
        elite_idx = rewards.argsort()[-n_elite:]
        elite_weights = [population[i] for i in elite_idx]
        mean = np.mean(elite_weights, axis=0)
        best_reward = rewards[elite_idx[-1]]
        scores.append(best_reward)
        print(f"Iteration {i:3d} | Best reward: {best_reward:8.2f}")

    env.close()
    set_weights(agent, mean)
    return scores

if __name__ == "__main__":
    # TRAIN
    scores = cem()

    # PLOT
    plt.figure(figsize=(8,4))
    plt.plot(scores)
    plt.xlabel("Iteration")
    plt.ylabel("Best episode reward")
    plt.title("CEM + CNN on CarRacing-v3 (FAST)")
    plt.grid()
    plt.show()

    # SAVE MODEL
    torch.save(agent.state_dict(), "cnn_cem_carracing.pth")