import pygame
import numpy as np
from env import SwarmSearchEnv
from agent import DroneAgent

ENV_SIZE = 50
NUM_DRONES = 120
CELL_SIZE = 14
FPS = 15

# Colors
COLOR_BG = (15, 23, 42)          
COLOR_GRID = (30, 41, 59)        
COLOR_EXPLORED = (30, 58, 138)   
COLOR_TARGET = (239, 68, 68)     
COLOR_DRONE = (56, 189, 248)     
COLOR_BASE = (234, 179, 8)       

def run_visual_demo():
    try:
        DroneAgent.q_table = np.load("q_table.npy")
        print("Loaded memory! Launching Swarm via Runway System...")
    except FileNotFoundError:
        print("Error: 'q_table.npy' not found! Run 'python train.py' first.")
        return

    env = SwarmSearchEnv(num_drones=NUM_DRONES, env_size=ENV_SIZE, is_training=False)
    agents = [DroneAgent(ENV_SIZE) for _ in range(NUM_DRONES)]
    
    for agent in agents:
        agent.epsilon = 0.05 

    pygame.init()
    screen = pygame.display.set_mode((ENV_SIZE * CELL_SIZE, ENV_SIZE * CELL_SIZE))
    pygame.display.set_caption("DR.ONE Swarm: Staggered Runway Launch")
    clock = pygame.time.Clock()

    running = True
    steps = 0
    unlaunched_queue = [f"drone{i}" for i in range(NUM_DRONES)]

    while running and not env.found_victim:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- STAGGERED RUNWAY LAUNCH ---
        if unlaunched_queue:
            launched = env.launch_drone(unlaunched_queue[0])
            if launched:
                unlaunched_queue.pop(0)

        # Move active drones using Pure RL Policy
        states = [env.get_state(d) for d in env.active_drones]
        actions = [agents[int(d.replace('drone', ''))].choose_action(s) for d, s in zip(env.active_drones, states)]

        for i, d in enumerate(env.active_drones):
            env.move_drone(d, actions[i])

        steps += 1
        screen.fill(COLOR_BG)

        for x in range(ENV_SIZE):
            for y in range(ENV_SIZE):
                rect = (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if env.explored[x, y] == 1:
                    pygame.draw.rect(screen, COLOR_EXPLORED, rect)
                pygame.draw.rect(screen, COLOR_GRID, rect, 1)

        base_rect = (1 * CELL_SIZE, 1 * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLOR_BASE, base_rect, 2)

        tx, ty = env.target_position
        target_center = (ty * CELL_SIZE + CELL_SIZE // 2, tx * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, COLOR_TARGET, target_center, CELL_SIZE // 2)

        # Render active drones
        for d in env.active_drones:
            pos = env.drone_positions[d]
            if pos != (-1, -1):
                dx, dy = pos
                drone_center = (dy * CELL_SIZE + CELL_SIZE // 2, dx * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(screen, COLOR_DRONE, drone_center, CELL_SIZE // 3)

        pygame.display.flip()
        clock.tick(FPS)

    print(f"Target located successfully in {steps} steps!")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    run_visual_demo()
