import numpy as np
import matplotlib.pyplot as plt
from env import SwarmSearchEnv
from agent import DroneAgent

episodes = 300
max_steps = 200
num_drones = 120
environment_size = 50

reward_history = []
steps_history = []
coverage_history = []
collision_history = []

agents = [DroneAgent(environment_size) for _ in range(num_drones)]

print("Starting Pure RL Swarm Training (With Hover & 5 Actions)...")

for ep in range(episodes):
    env = SwarmSearchEnv(num_drones=num_drones, env_size=environment_size, is_training=True)
    states = [env.get_state(f"drone{i}") for i in range(num_drones)]

    total_reward = 0
    steps = 0

    while not env.found_victim and steps < max_steps:
        actions = [agents[i].choose_action(states[i]) for i in range(num_drones)]
        rewards = []

        for i in range(num_drones):
            reward = env.move_drone(f"drone{i}", actions[i])
            rewards.append(reward)

        next_states = [env.get_state(f"drone{i}") for i in range(num_drones)]
        total_reward += sum(rewards)

        for i in range(num_drones):
            agents[i].learn(states[i], actions[i], rewards[i], next_states[i])

        states = next_states
        steps += 1

    for i in range(num_drones):
        agents[i].epsilon = max(0.05, agents[i].epsilon * 0.985)

    coverage = (env.explored.sum() / (env.size * env.size)) * 100
    reward_history.append(total_reward)
    steps_history.append(steps)
    coverage_history.append(coverage)
    collision_history.append(env.collision_count)

    if (ep + 1) % 20 == 0:
        print(f"Episode {ep+1:3d} | Reward: {total_reward:7.0f} | Steps: {steps:3d} | Collisions: {env.collision_count:4d}")

print("Training Complete!")
np.save("q_table.npy", DroneAgent.q_table)
print("Saved memory to 'q_table.npy' successfully!")

# --- MATPLOTLIB GRAPHS ---
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(reward_history, color='green')
plt.title("Total Swarm Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(2, 2, 2)
plt.plot(steps_history, color='blue')
plt.title("Steps to Locate Victim")
plt.xlabel("Episode")
plt.ylabel("Steps")

plt.subplot(2, 2, 3)
plt.plot(coverage_history, color='purple')
plt.title("Map Coverage (%)")
plt.xlabel("Episode")
plt.ylabel("Coverage")

plt.subplot(2, 2, 4)
plt.plot(collision_history, color='red')
plt.title("Swarm Collisions")
plt.xlabel("Episode")
plt.ylabel("Collisions")

plt.tight_layout()
plt.show()
