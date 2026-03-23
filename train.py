from env import SwarmSearchEnv
from agent import DroneAgent
import matplotlib.pyplot as plt
import inspect

print("ENV FILE PATH:", inspect.getfile(SwarmSearchEnv))

episodes = 300
max_steps = 50
num_drones = 5

reward_history = []
steps_history = []
coverage_history = []
collision_history = []

# Create agents ONCE (learning persists)
env = SwarmSearchEnv(num_drones=num_drones)
agents = [DroneAgent(env.size) for _ in range(num_drones)]

for ep in range(episodes):

    env = SwarmSearchEnv(num_drones=num_drones)
    states = list(env.drone_positions.values())

    total_reward = 0
    steps = 0

    while not env.found_victim and steps < max_steps:

        actions = []

        # Action selection
        for i, agent in enumerate(agents):
            action = agent.choose_action(states[i])
            actions.append(action)

        rewards = []

        # Move drones
        for i in range(num_drones):
            drone_id = f"drone{i}"
            reward = env.move_drone(drone_id, actions[i])
            rewards.append(reward)

        next_states = list(env.drone_positions.values())
        global_reward = sum(rewards)

        # Learning update
        for i, agent in enumerate(agents):
            agent.learn(states[i], actions[i], global_reward, next_states[i])

        states = next_states
        total_reward += global_reward
        steps += 1

    # Final episode animation only AFTER learning step loop
    if ep == episodes - 1:
        print("Rendering Final Episode...")
        for _ in range(10):
            env.render()
        plt.close()  # Close animation figure properly

    # Metrics
    coverage = (env.explored.sum() / (env.size * env.size)) * 100

    reward_history.append(total_reward)
    steps_history.append(steps)
    coverage_history.append(coverage)
    collision_history.append(env.collision_count)

    print(f"Episode {ep+1} | Reward: {total_reward} | Steps: {steps} | Coverage: {coverage:.2f}%")

# -----------------------------
# ANALYTICS SECTION
# -----------------------------

plt.figure()
plt.plot(reward_history)
plt.title("Reward vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

plt.figure()
plt.plot(steps_history)
plt.title("Steps to Target vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.show()

plt.figure()
plt.plot(coverage_history)
plt.title("Coverage % vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Coverage %")
plt.show()

plt.figure()
plt.plot(collision_history)
plt.title("Collisions vs Episodes")
plt.xlabel("Episode")
plt.ylabel("Collision Count")
plt.show()

import numpy as np

window = 20
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')

plt.figure()
plt.plot(smoothed)
plt.title("Smoothed Reward vs Episodes")
plt.show()
