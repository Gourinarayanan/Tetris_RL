import os
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# 1. DQN MODEL
# -------------------------
class DQN(nn.Module):
    def __init__(self, input_dim=27):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)

        return x.squeeze()


# -------------------------
# 2. REPLAY BUFFER
# -------------------------
Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])

class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, reward, next_state, done):
        self.memory.append(Transition(state, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# -------------------------
# 3. INITIALIZE MODEL
# -------------------------
model = DQN()
target_model = DQN()
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.SmoothL1Loss()

replay_buffer = ReplayBuffer()

gamma = 0.99  # discount factor


# -------------------------
# 4. TRAINING HELPERS
# -------------------------

def update_target_model():
    target_model.load_state_dict(model.state_dict())


def save_model(path="dqn_model.pth"):
    torch.save(model.state_dict(), path)


def load_model(path="dqn_model.pth"):
    if not os.path.exists(path):
        return False

    checkpoint = torch.load(path, map_location="cpu")
    try:
        model.load_state_dict(checkpoint)
        print(f"Loaded model from {path}")
    except RuntimeError as e:
        print(f"Warning: model load mismatch: {e}")
        print("Attempting partial load from checkpoint...")
        try:
            model.load_state_dict(checkpoint, strict=False)
            print("Partial model load completed. New layers initialized randomly.")
        except RuntimeError as e2:
            print(f"Failed to partially load checkpoint: {e2}")
            print("Skipping incompatible checkpoint and using a fresh model.")
            return False

    target_model.load_state_dict(model.state_dict())
    return True


def train_step(state, reward, next_state, done, batch_size=64):
    replay_buffer.push(state, reward, next_state, done)

    if len(replay_buffer) < batch_size:
        return 0.0

    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))

    states = torch.tensor(batch.state, dtype=torch.float32)
    rewards = torch.tensor(batch.reward, dtype=torch.float32)
    next_states = torch.tensor(batch.next_state, dtype=torch.float32)
    dones = torch.tensor(batch.done, dtype=torch.float32)

    q_pred = model(states)
    with torch.no_grad():
        q_next = target_model(next_states)

    target = rewards + gamma * q_next * (1.0 - dones)

    loss = loss_fn(q_pred, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()