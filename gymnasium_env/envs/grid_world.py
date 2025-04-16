from enum import Enum
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
import pygame
import numpy as np
from functools import lru_cache

weights = np.array([1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10])
weights = np.array(weights)
weights = weights / weights.sum()
shapes = np.array([
        [[1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 1, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 0, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 0, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
        [[1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [1, 1, 1, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]],
        
    ], dtype=np.int8)

@lru_cache(maxsize=256)
def get_trimmed_shape(shape_id):
    shape = trim_shape(shapes[shape_id])
    return shape, shape.shape

def trim_shape(shape: np.ndarray) -> np.ndarray:
    rows = np.any(shape, axis=1)
    cols = np.any(shape, axis=0)
    return shape[np.ix_(rows, cols)]

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=8):
        self.size = size  # The size of the square grid
        self.window_size = 800  # The size of the PyGame window
        self.cell_size = 40
        
        self.combo_multiplier = 1.5

        self.max_shape_size = 5
        self.observation_space = spaces.Dict(
            {
                "grid": Box(low=0, high=1, shape=(8, 8), dtype=np.int8),
                "shapes": Box(low=0, high=1, shape=(3, self.max_shape_size, self.max_shape_size), dtype=np.int8),
                "used_shapes": Box(low=0, high=1, shape=(3,), dtype=np.int8)
            }
        )

        self.action_space = MultiDiscrete([3, 8, 8])

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"grid": self._grid, "shapes": np.array([shapes[i] for i in self._curr_shapes], dtype=np.int8), "used_shapes": self._used_shapes}

    def _get_info(self):
        return {
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reward = 0
        self._grid = np.zeros((self.size, self.size), dtype=np.int8)
        self.reset_curr_shapes()

        observation = self._get_obs()
        info = self._get_info()

        # if self.render_mode == "human":
        #     self._render_frame()

        return observation, info

    def step(self, action):
        # action = (shape_idx, pos_x, pos_y)
        shape_idx, pos_x, pos_y = action
        
        # check if the action is valid
        # check if shape is used
        if self._used_shapes[shape_idx]:
            valid = False
        # check position
        else:
            valid = self.is_valid_position(shape_idx, pos_x, pos_y)
        
        # if valid, update the board and provide reward equal to the shape size
        if valid:
            reward = np.sum(shapes[self._curr_shapes[shape_idx]])
            
            # update board
            self.update_board(shape_idx, pos_x, pos_y)
            self._used_shapes[shape_idx] = 1
            
            # check for clears and update reward
            reward += self.size * self.combo_multiplier * self.check_clears()
            
            # check if all shapes used
            if np.sum(self._used_shapes) == 3:
                self.reset_curr_shapes()
        
        # if invalid, do not change the board and provide a reward of -10
        else:
            reward = -5
            
        terminated = True
        for idx in range(len(self._curr_shapes)):
            if self._used_shapes[idx]:
                continue
            for y in range(self.size):
                for x in range(self.size):
                    if self.is_valid_position(idx, x, y):
                        terminated = False
                        break
                else:
                    continue
                break

        # if self.render_mode == "human":
        #     self._render_frame()
        observation = self._get_obs()
        info = self._get_info()
        self._reward += reward
        return observation, reward, terminated, False, info
    
    def is_valid_position(self, shape_idx, pos_x, pos_y):
        shape_id = self._curr_shapes[shape_idx]
        shape, (h, w) = get_trimmed_shape(shape_id)

        x, y = pos_x, pos_y

        if y + h > self.size or x + w > self.size:
            return False

        grid_region = self._grid[y:y+h, x:x+w]

        overlap = np.bitwise_and(grid_region, shape)
        valid = not overlap.any()
        return valid
    
    def update_board(self, shape_idx, pos_x, pos_y):
        shape_id = self._curr_shapes[shape_idx]
        shape, (h, w) = get_trimmed_shape(shape_id)
        self._grid[pos_y:pos_y+h, pos_x:pos_x+w] |= shape
                    
    def check_clears(self):
        num_clears = 0
        row_sums = np.sum(self._grid, axis=1)
        col_sums = np.sum(self._grid, axis=0)
        
        for i, sum in enumerate(row_sums):
            if sum == self.size:
                self._grid[i, :] = 0
                num_clears += 1
                
        for i, sum in enumerate(col_sums):
            if sum == self.size:
                self._grid[:, i] = 0
                num_clears += 1
        
        return num_clears
    
    def reset_curr_shapes(self):
        self._curr_shapes = np.random.choice(len(shapes), size=3, replace=True, p=weights)
        self._used_shapes = np.zeros(3, dtype=np.int8)

    def render(self):
        return self._render_frame()

    def _render_frame(self):

        cell_size = self.cell_size
        grid_size = self.size  # 8 for an 8x8 grid
        top_offset = cell_size
        grid_pixel_size = grid_size * cell_size

        # Calculate canvas size dynamically
        shape_display_height = cell_size * 6  # Enough space for shapes
        window_height = top_offset + grid_pixel_size + shape_display_height
        window_width = max(grid_pixel_size + 2 * cell_size, 400)

        # Horizontal center offset
        left_offset = (window_width - grid_pixel_size) // 2

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", int(cell_size * 0.75))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((window_width, window_height))
        canvas.fill((255, 255, 255))
        
        reward_text = f"{self._reward}"
        text_surface = self.font.render(reward_text, True, (0, 0, 0))
        canvas.blit(text_surface, (left_offset, cell_size * 0.25))

        # Draw grid squares
        for y in range(grid_size):
            for x in range(grid_size):
                rect = pygame.Rect(
                    left_offset + x * cell_size,
                    top_offset + y * cell_size,
                    cell_size,
                    cell_size,
                )
                color = (50, 50, 50) if self._grid[y, x] else (220, 220, 220)
                pygame.draw.rect(canvas, color, rect)
                pygame.draw.rect(canvas, (0, 0, 0), rect, width=2)

        # Draw unused shapes at the bottom
        small_cell_size = cell_size / 2
        displayable_shapes = sum(not used for used in self._used_shapes)
        shape_padding = 10
        shape_display_width = displayable_shapes * (small_cell_size * 5 + shape_padding) - shape_padding
        shape_left_offset = (window_width - shape_display_width) // 2
        draw_index = 0
        for i, shape_idx in enumerate(self._curr_shapes):
            if self._used_shapes[i]:  # skip used shapes
                continue
            shape = shapes[shape_idx]
            shape_arr = np.array(shape)
            h, w = shape_arr.shape
            shape_x = shape_left_offset + draw_index * (small_cell_size * 5 + shape_padding)
            shape_y = top_offset + grid_pixel_size + shape_padding

            for dy in range(h):
                for dx in range(w):
                    if shape_arr[dy, dx]:
                        rect = pygame.Rect(
                            shape_x + dx * small_cell_size,
                            shape_y + dy * small_cell_size,
                            small_cell_size,
                            small_cell_size,
                        )
                        pygame.draw.rect(canvas, (100, 150, 255), rect)
                        pygame.draw.rect(canvas, (0, 0, 0), rect, width=2)
                        
            draw_index += 1

        # Blit and display
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
