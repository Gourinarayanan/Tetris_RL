import os
import random
import math
import array
import torch
import pygame
from env.pieces import rotate
from env.tetris_env import TetrisEnv
from models.neural import model, load_model
from renderer import Renderer, WIDTH, HEIGHT

MODEL_PATH = "dqn_model.pth"

pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.init()
pygame.mixer.set_num_channels(8)
font_title = pygame.font.SysFont(None, 64, bold=True)
font_button = pygame.font.SysFont(None, 32, bold=True)
font_text = pygame.font.SysFont(None, 24)

MENU = 0
PLAY = 1

env = TetrisEnv()
renderer = Renderer()

loaded = load_model(MODEL_PATH)
if loaded:
    print(f"Loaded trained model from {MODEL_PATH}")
else:
    print("No trained model found; using untrained network.")

state = env.reset()

clock = pygame.time.Clock()
done = False
game_over = False
printed_game_over = False
epsilon = 0.0

active_piece = None
active_x = 3
active_y = 0
pending_action = None
mode = MENU
btn_level1 = pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 - 10, 240, 50)
btn_level2 = pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 + 50, 240, 50)
btn_level3 = pygame.Rect(WIDTH // 2 - 120, HEIGHT // 2 + 110, 240, 50)

def make_sound(frequency=440, duration_ms=180, volume=0.15):
    sample_rate = 44100
    n_samples = int(sample_rate * duration_ms / 1000)
    wave = array.array('h')
    for i in range(n_samples):
        t = i / sample_rate
        value = volume * 32767 * math.sin(2 * math.pi * frequency * t)
        wave.append(int(value))
    return pygame.mixer.Sound(buffer=wave.tobytes())

menu_tone = make_sound(220, duration_ms=400, volume=0.08)
start_sound = make_sound(600, duration_ms=120, volume=0.21)
land_sound = make_sound(520, duration_ms=100, volume=0.22)
game_over_sound = make_sound(120, duration_ms=500, volume=0.18)
menu_channel = pygame.mixer.Channel(0)
menu_channel.play(menu_tone, loops=-1)

score = 0
lines = 0
level = 1

fall_time = 0
fall_delay = 350
current_fall_delay = fall_delay
dt = 0


def choose_fall_delay():
    # Some pieces fall faster than others, like a Tetris soft drop effect.
    if random.random() < 0.25:
        return random.randint(180, 260)
    return random.randint(300, 420)


def draw_menu():
    renderer.screen.fill((15, 15, 45))
    pygame.draw.rect(renderer.screen, (18, 99, 150), (0, 0, WIDTH, HEIGHT))

    title_text = font_title.render("TETRIS RL", True, (255, 255, 255))
    title_rect = title_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 120))
    renderer.screen.blit(title_text, title_rect)

    subtitle = font_text.render("Train the model with train.py, then press start.", True, (230, 230, 230))
    subtitle_rect = subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 70))
    renderer.screen.blit(subtitle, subtitle_rect)

    pygame.draw.rect(renderer.screen, (0, 204, 204), btn_level1, border_radius=12)
    l1_text = font_button.render("LEVEL 1 (EXPERT)", True, (10, 10, 30))
    renderer.screen.blit(l1_text, l1_text.get_rect(center=btn_level1.center))

    pygame.draw.rect(renderer.screen, (0, 204, 204), btn_level2, border_radius=12)
    l2_text = font_button.render("LEVEL 2 (MEDIUM)", True, (10, 10, 30))
    renderer.screen.blit(l2_text, l2_text.get_rect(center=btn_level2.center))

    pygame.draw.rect(renderer.screen, (0, 204, 204), btn_level3, border_radius=12)
    l3_text = font_button.render("LEVEL 3 (NOVICE)", True, (10, 10, 30))
    renderer.screen.blit(l3_text, l3_text.get_rect(center=btn_level3.center))

    info_text = font_text.render("Use the trained agent model. Close the window to quit.", True, (200, 200, 200))
    info_rect = info_text.get_rect(center=(WIDTH // 2, HEIGHT - 40))
    renderer.screen.blit(info_text, info_rect)

    pygame.display.flip()


def choose_action(env, epsilon=0.0):
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
            value = model(torch.tensor(sim_state, dtype=torch.float32)).item()

        score = sim_reward if sim_done else sim_reward + 0.99 * value
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def rotated_piece(shape, rotation):
    rotated = shape
    for _ in range(rotation):
        rotated = rotate(rotated)
    return rotated


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif mode == MENU:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if btn_level1.collidepoint(event.pos):
                    epsilon = 0.0
                    mode = PLAY
                    menu_channel.stop()
                    start_sound.play()
                elif btn_level2.collidepoint(event.pos):
                    epsilon = 0.2
                    mode = PLAY
                    menu_channel.stop()
                    start_sound.play()
                elif btn_level3.collidepoint(event.pos):
                    epsilon = 0.5
                    mode = PLAY
                    menu_channel.stop()
                    start_sound.play()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                epsilon = 0.0
                mode = PLAY
                menu_channel.stop()
                start_sound.play()

    if mode == MENU:
        draw_menu()
    else:
        if not game_over:
            if pending_action is None:
                pending_action = choose_action(env, epsilon)
                if pending_action is not None:
                    if len(pending_action) == 2: # old model
                        rotation, active_x = pending_action
                        active_piece = rotated_piece(env.current_shape[0], rotation)
                        active_y = 0
                        active_path = []
                    else:
                        rotation, final_x, final_y, path = pending_action
                        active_path = path
                        active_path_index = 0
                        if len(active_path) > 0:
                            cur_r, active_x, active_y = active_path[0]
                            active_piece = rotated_piece(env.current_shape[0], cur_r)
                        else:
                            active_piece = rotated_piece(env.current_shape[0], 0)
                            active_x = 3
                            active_y = 0
                    current_fall_delay = 120 # Optimized speed for watching rotations and tucks
                else:
                    game_over = True

            if pending_action is not None and not game_over:
                fall_time += dt
                if fall_time >= current_fall_delay:
                    fall_time = 0
                    
                    if len(pending_action) > 2 and active_path_index < len(active_path):
                        cur_r, active_x, active_y = active_path[active_path_index]
                        active_piece = rotated_piece(env.current_shape[0], cur_r)
                        active_path_index += 1
                    elif len(pending_action) == 2 and env.board.is_valid_position(active_piece, active_x, active_y + 1):
                        active_y += 1
                    else:
                        next_state, reward, game_over = env.step(pending_action)
                        if reward > 0:
                            pass
                        
                        score += reward
                        lines = getattr(env, '_total_lines', lines + 1 if reward > 0 else lines)
                        level = max(1, 1 + lines // 10)
                        
                        if reward >= 0:
                            land_sound.play()
                        state = next_state
                        pending_action = None
                        active_piece = None

        renderer.draw(env.board, active_piece, active_x, active_y, score=score, lines=lines, level=level, next_piece=env.next_shape[0], game_over=game_over)

        if game_over and not printed_game_over:
            print("Game Over")
            game_over_sound.play()
            printed_game_over = True

    dt = clock.tick(60)

pygame.quit()