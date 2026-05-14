import random
import os
import time

import torch
from env.tetris_env import TetrisEnv
from models.neural import model, target_model, train_step, soft_update_target, save_model, load_model, scheduler, optimizer

MODEL_PATH = "dqn_model.pth"
NUM_EPISODES = 3000
BATCH_SIZE = 512
EPS_START = 1.0
EPS_END = 0.01
# Slower decay to allow more exploration of state space
EPS_DECAY = 0.998


def choose_action(env, epsilon=0.1):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None

    if random.random() < epsilon:
        return random.choice(valid_actions)

    best_action = None
    best_score = float("-inf")

    # Evaluate all possible placement states
    for action in valid_actions:
        sim_state, sim_reward, sim_done = env.simulate_step(action)
        with torch.no_grad():
            st = torch.tensor(sim_state, dtype=torch.float32).unsqueeze(0)
            q_value = model(st).item()

        # Evaluate the state: reward received + discounted future expected reward
        score = sim_reward if sim_done else sim_reward + 0.99 * q_value
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def run_training():
    print("=" * 60)
    print("🚀 Tetris RL Agent - Training Mode")
    print("=" * 60)
    
    epsilon = EPS_START
    env = TetrisEnv()
    best_reward = float("-inf")
    episode_rewards = []
    episode_lines = []

    if os.path.exists(MODEL_PATH):
        if load_model(MODEL_PATH):
            epsilon = 0.2 # Reduce epsilon if loading pre-trained

    start_time = time.time()

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        game_over = False
        total_reward = 0
        
        while not game_over:
            action = choose_action(env, epsilon)
            if action is None:
                game_over = True
                break

            next_state, reward, game_over = env.step(action)
            loss = train_step(state, reward, next_state, game_over, batch_size=BATCH_SIZE)
            soft_update_target()
            
            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_lines.append(env._total_lines)
        
        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        scheduler.step() # Decay learning rate

        # Logging
        if episode % 20 == 0:
            avg_reward = sum(episode_rewards[-20:]) / min(20, len(episode_rewards))
            avg_lines = sum(episode_lines[-20:]) / min(20, len(episode_lines))
            lr = optimizer.param_groups[0]['lr']
            print(f"Ep {episode:4d} | AvgReward: {avg_reward:7.1f} | AvgLines: {avg_lines:5.1f} | MaxLines: {max(episode_lines[-20:]):4d} | Eps: {epsilon:.3f} | LR: {lr:.5f}")
            save_model(MODEL_PATH)

        if total_reward > best_reward:
            best_reward = total_reward
            save_model(MODEL_PATH)

    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes.")
    save_model(MODEL_PATH)


if __name__ == "__main__":
    run_training()