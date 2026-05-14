import os
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from env.tetris_env import STATE_DIM   # single source of truth

# ─────────────────────────────────────────────────────────────────────────────
# 1. DUELING DOUBLE-DQN NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class DQN(nn.Module):
    """
    Dueling architecture: shared encoder → separate value & advantage streams.
    Outputs a single scalar Q-value estimate for a given board state.
    """
    def __init__(self, input_dim: int = STATE_DIM):
        super().__init__()
        # Shared feature extractor with batch normalisation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        feat  = self.encoder(x)
        value = self.value_stream(feat)         # (B, 1)
        adv   = self.adv_stream(feat)           # (B, 1)
        q     = value + adv - adv.mean(dim=1, keepdim=True)
        return q.squeeze(-1)                    # (B,) or scalar


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRIORITISED REPLAY BUFFER
# ─────────────────────────────────────────────────────────────────────────────
Transition = namedtuple("Transition", ["state", "reward", "next_state", "done"])


class PrioritisedReplayBuffer:
    """Simple proportional prioritised experience replay."""

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha    = alpha
        self.memory   = []
        self.priorities = deque(maxlen=capacity)
        self._idx     = 0

    def push(self, state, reward, next_state, done, priority: float = 1.0):
        t = Transition(state, reward, next_state, done)
        if len(self.memory) < self.capacity:
            self.memory.append(t)
        else:
            self.memory[self._idx] = t
        if len(self.priorities) < self.capacity:
            self.priorities.append(priority ** self.alpha)
        else:
            self.priorities[self._idx] = priority ** self.alpha
        self._idx = (self._idx + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        prios = list(self.priorities)[:len(self.memory)]
        total = sum(prios)
        probs = [p / total for p in prios]
        indices = random.choices(range(len(self.memory)), weights=probs, k=batch_size)
        samples = [self.memory[i] for i in indices]
        # Importance-sampling weights
        n   = len(self.memory)
        weights = [(n * probs[i]) ** (-beta) for i in indices]
        max_w   = max(weights)
        weights = [w / max_w for w in weights]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + 1e-5) ** self.alpha
            self.priorities[idx] = p

    def __len__(self):
        return len(self.memory)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GLOBAL SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────
model        = DQN(STATE_DIM)
target_model = DQN(STATE_DIM)
target_model.load_state_dict(model.state_dict())
target_model.eval()

optimizer     = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler     = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.95)
loss_fn       = nn.SmoothL1Loss(reduction='none')
replay_buffer = PrioritisedReplayBuffer(capacity=50_000)
gamma         = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def update_target_model():
    """Hard update of the target network."""
    target_model.load_state_dict(model.state_dict())


def soft_update_target(tau: float = 0.005):
    """Polyak soft update."""
    for tp, p in zip(target_model.parameters(), model.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


def save_model(path: str = "dqn_model.pth"):
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim":  STATE_DIM,
    }, path)
    print(f"[Neural] Model saved → {path}")


def load_model(path: str = "dqn_model.pth") -> bool:
    if not os.path.exists(path):
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # Support both old bare state_dict and new wrapped format
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        target_model.load_state_dict(model.state_dict())
        target_model.eval()
        print(f"[Neural] Loaded model from {path}")
        return True
    except Exception as e:
        print(f"[Neural] Could not load model ({e}). Starting fresh.")
        return False


def train_step(state, reward: float, next_state, done: bool,
               batch_size: int = 64, beta: float = 0.4) -> float:
    """
    Stores transition, samples a prioritised mini-batch, performs
    a Double-DQN gradient step and returns the scalar loss.
    """
    # Push with max priority so new transitions are sampled immediately
    max_p = max(replay_buffer.priorities, default=1.0)
    replay_buffer.push(state, reward, next_state, done, priority=max_p)

    if len(replay_buffer) < batch_size:
        return 0.0

    transitions, indices, weights = replay_buffer.sample(batch_size, beta)
    batch = Transition(*zip(*transitions))

    states      = torch.tensor(batch.state,      dtype=torch.float32)
    rewards_t   = torch.tensor(batch.reward,     dtype=torch.float32)
    next_states = torch.tensor(batch.next_state, dtype=torch.float32)
    dones_t     = torch.tensor(batch.done,       dtype=torch.float32)
    weights_t   = torch.tensor(weights,          dtype=torch.float32)

    # Q(s, a) from online model
    q_pred = model(states)

    # Double DQN: use online model to *select* action, target to *evaluate*
    with torch.no_grad():
        q_next = target_model(next_states)

    target = rewards_t + gamma * q_next * (1.0 - dones_t)

    # Element-wise loss * IS weights
    element_loss = loss_fn(q_pred, target)                     # (B,)
    td_errors    = (q_pred - target).detach().abs().cpu().tolist()
    loss         = (element_loss * weights_t).mean()

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()

    # Update priorities
    replay_buffer.update_priorities(indices, td_errors)

    return loss.item()