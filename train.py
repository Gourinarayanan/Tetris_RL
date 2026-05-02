import random
import os

import torch
from env.tetris_env import TetrisEnv
from models.neural import model, train_step, update_target_model, save_model, load_model

MODEL_PATH = "dqn_model.pth"
NUM_EPISODES = 5000
BATCH_SIZE = 128
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.992
TARGET_UPDATE_EVERY = 5


def choose_action(env, epsilon=0.1):
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        return None

    if random.random() < epsilon:
        return random.choice(valid_actions)

    best_action = None
    best_score = float("-inf")

    for action in valid_actions:
        sim_state, sim_reward, sim_done = env.simulate_step(action)
        with torch.no_grad():
            value = torch.tensor(sim_state, dtype=torch.float32)
            value = value.unsqueeze(0)
            from models.neural import model
            q_value = model(value).item()

        score = sim_reward if sim_done else sim_reward + 0.99 * q_value
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def run_training():
    epsilon = EPS_START
    env = TetrisEnv()
    best_reward = float("-inf")
    episode_rewards = []

    if os.path.exists(MODEL_PATH):
        loaded = load_model(MODEL_PATH)
        if loaded:
            print(f"Loaded existing model from {MODEL_PATH}")
        else:
            print(f"Incompatible checkpoint {MODEL_PATH} skipped; starting fresh model.")

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        game_over = False
        total_reward = 0
        steps = 0

        while not game_over:
            action = choose_action(env, epsilon)
            if action is None:
                game_over = True
                break

            next_state, reward, game_over = env.step(action)
            train_step(state, reward, next_state, game_over, batch_size=BATCH_SIZE)
            state = next_state
            total_reward += reward
            steps += 1

        episode_rewards.append(total_reward)
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        if episode % TARGET_UPDATE_EVERY == 0:
            update_target_model()

        if episode % 50 == 0:
            avg_reward = sum(episode_rewards[-50:]) / min(50, len(episode_rewards))
            print(f"Episode {episode:4d} | reward {total_reward:4.1f} | avg last 50 {avg_reward:4.1f} | eps {epsilon:.3f}")
            save_model(MODEL_PATH)

        if total_reward > best_reward:
            best_reward = total_reward
            save_model(MODEL_PATH)

    print("Training finished")
    save_model(MODEL_PATH)


if __name__ == "__main__":
    run_training()