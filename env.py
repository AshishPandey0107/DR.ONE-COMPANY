import numpy as np
import matplotlib.pyplot as plt


class SwarmSearchEnv:
    def __init__(self, size=5, num_drones=3):
        self.size = size
        self.num_drones = num_drones

        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.explored = np.zeros((self.size, self.size), dtype=int)

        self.collision_count = 0
        self.found_victim = False

        # Generate unique starting positions
        self.drone_positions = {}
        taken_positions = set()

        for i in range(self.num_drones):
            while True:
                pos = (
                    np.random.randint(0, self.size),
                    np.random.randint(0, self.size),
                )
                if pos not in taken_positions:
                    taken_positions.add(pos)
                    self.drone_positions[f"drone{i}"] = pos
                    break

        # Mark starting positions as explored
        for pos in self.drone_positions.values():
            self.explored[pos[0], pos[1]] = 1

        # Generate victim position (not overlapping drones)
        while True:
            victim = (
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            )
            if victim not in self.drone_positions.values():
                self.victim_position = victim
                break

    # --------------------------------------------------

    def update_grid(self):
        self.grid[:] = 0

        # Victim
        vx, vy = self.victim_position
        self.grid[vx, vy] = 2

        # Drones
        for x, y in self.drone_positions.values():
            self.grid[x, y] = 1

    # --------------------------------------------------

    def move_drone(self, drone_id, action):

        x, y = self.drone_positions[drone_id]

        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0:      # Up
            x = max(0, x - 1)
        elif action == 1:    # Down
            x = min(self.size - 1, x + 1)
        elif action == 2:    # Left
            y = max(0, y - 1)
        elif action == 3:    # Right
            y = min(self.size - 1, y + 1)

        new_pos = (x, y)
        reward = -1  # small time penalty

        # Collision detection
        collision = False
        for other_id, other_pos in self.drone_positions.items():
            if other_id != drone_id and other_pos == new_pos:
                reward -= 5
                self.collision_count += 1
                collision = True
                break

        # Only update position if no collision
        if not collision:
            self.drone_positions[drone_id] = new_pos

            # Exploration reward
            if self.explored[new_pos[0], new_pos[1]] == 0:
                reward += 2
                self.explored[new_pos[0], new_pos[1]] = 1

            # Victim found
            if new_pos == self.victim_position:
                reward += 20
                self.found_victim = True

        return reward

    # --------------------------------------------------

    def render(self):
        self.update_grid()

        plt.imshow(self.grid, cmap="viridis", vmin=0, vmax=2)
        plt.title("DR.ONE – AI Drone Swarm Search & Rescue")
        plt.xticks(range(self.size))
        plt.yticks(range(self.size))
        plt.grid(True)
        plt.pause(0.3)
        plt.clf()
