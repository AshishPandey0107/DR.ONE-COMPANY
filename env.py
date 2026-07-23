import numpy as np
import random

class SwarmSearchEnv:
    def __init__(self, num_drones=120, env_size=50, is_training=True):
        self.size = env_size
        self.num_drones = num_drones
        self.found_victim = False
        self.collision_count = 0
        self.explored = np.zeros((env_size, env_size))
        self.is_training = is_training
        
        while True:
            self.target_position = (random.randint(0, env_size - 1), random.randint(0, env_size - 1))
            if self.target_position != (1, 1):
                break
        
        self.drone_positions = {}
        self.active_drones = []
        
        if is_training:
            # Random initial positions for training
            for i in range(num_drones):
                pos = (random.randint(0, env_size - 1), random.randint(0, env_size - 1))
                self.drone_positions[f"drone{i}"] = pos
                self.explored[pos[0], pos[1]] = 1
                self.active_drones.append(f"drone{i}")
        else:
            # Demo starts with 0 active drones; they deploy via staggered launch from (1,1)
            for i in range(num_drones):
                self.drone_positions[f"drone{i}"] = (-1, -1)  # -1, -1 means inactive / in hangar

        self.explored[1, 1] = 1

    def launch_drone(self, drone_id):
        """Staggered Launch: Releases a drone at Base Station if spot is clear."""
        base_pos = (1, 1)
        if base_pos not in self.drone_positions.values():
            self.drone_positions[drone_id] = base_pos
            self.active_drones.append(drone_id)
            return True
        return False

    def _is_obstacle(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return 1
        if self.explored[x, y] == 1:
            return 1
        if (x, y) in self.drone_positions.values():
            return 1
        return 0

    def get_state(self, drone_id):
        pos = self.drone_positions[drone_id]
        if pos == (-1, -1):
            return (0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # Dummy state for inactive drones
            
        x, y = pos
        tx, ty = self.target_position
        
        dx = int(np.sign(tx - x))
        dy = int(np.sign(ty - y))
        
        w_up = 1 if x == 0 else 0
        w_right = 1 if y == self.size - 1 else 0
        w_down = 1 if x == self.size - 1 else 0
        w_left = 1 if y == 0 else 0
        
        e_up = self._is_obstacle(x-1, y)
        e_right = self._is_obstacle(x, y+1)
        e_down = self._is_obstacle(x+1, y)
        e_left = self._is_obstacle(x, y-1)
        
        return (dx, dy, w_up, w_right, w_down, w_left, e_up, e_right, e_down, e_left)

    def move_drone(self, drone_id, action):
        if drone_id not in self.active_drones:
            return 0.0

        x, y = self.drone_positions[drone_id]
        nx, ny = x, y

        # Action 4: Hover / Stand Still
        if action == 4:
            return -0.2  # Small time penalty for standing still

        if action == 0: nx -= 1
        elif action == 1: ny += 1
        elif action == 2: nx += 1
        elif action == 3: ny -= 1

        # Out of bounds collision
        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            self.collision_count += 1
            return -10.0  

        # Teammate collision
        if (nx, ny) in self.drone_positions.values():
            self.collision_count += 1
            return -10.0 

        self.drone_positions[drone_id] = (nx, ny)
        
        reward = -0.1 
        if self.explored[nx, ny] == 0:
            reward += 2.0  
            self.explored[nx, ny] = 1
        else:
            reward -= 1.5  

        if (nx, ny) == self.target_position:
            self.found_victim = True
            reward += 100.0  

        return reward
