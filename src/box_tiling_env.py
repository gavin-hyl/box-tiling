import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Set, Tuple, Dict, Any, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Import from local objects module
from .objects import Object, Square, Rectangle, LShape, TShape

class BoxTilingEnv(gym.Env):
    """
    A 2D grid world environment for placing objects to maximize occupancy.
    
    State:
        A numpy array representation of the occupancy of each grid cell.
    
    Action:
        A numpy array representation of the cells that would be occupied by placing an object.
    
    Reward:
        0 if non-terminal, the occupancy percentage if terminal.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, x_max: int = 10, y_max: int = 10, available_objects: Optional[List[Object]] = None, 
                 render_mode: Optional[str] = None):
        """
        Initialize the environment.
        
        Args:
            x_max: The maximum x-coordinate of the grid.
            y_max: The maximum y-coordinate of the grid.
            available_objects: List of objects that can be placed in the grid. If None, defaults to basic shapes.
            render_mode: The render mode to use (either 'human' or 'rgb_array').
        """
        self.x_max = x_max
        self.y_max = y_max
        self.grid_size = (x_max + 1, y_max + 1)  # +1 because we include 0
        
        # Initialize the grid (0 = empty, 1 = occupied)
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Set up available objects with their rotations
        if available_objects is None:
            # Default objects
            self.base_objects = [
                Square(size=2),
                Rectangle(width=2, height=3),
                LShape(size=3),
                TShape(width=3, height=2)
            ]
        else:
            self.base_objects = available_objects
        
        # Generate all rotations of the available objects
        self.all_objects = []
        for obj in self.base_objects:
            self.all_objects.extend(obj.get_all_rotations())
        
        # Generate all possible actions (object, x, y)
        self.all_actions = []
        self.action_to_placement = {}
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=self.grid_size, dtype=np.int8)
        
        # Determine the maximum number of possible placements
        max_actions = len(self.all_objects) * self.x_max * self.y_max
        self.action_space = spaces.Discrete(max_actions + 1)  # +1 for "no placement" action
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        # For rendering
        self.fig = None
        self.gs = None  # GridSpec for layout
        self.main_ax = None  # Main grid
        self.objects_ax = None  # Available objects
    
    def get_available_objects_info(self) -> List[Dict]:
        """
        Get information about available objects for external use.
        
        Returns:
            List of dictionaries with information about each base object.
        """
        objects_info = []
        for i, obj in enumerate(self.base_objects):
            obj_info = {
                "id": i,
                "cells": obj.get_cells(),
                "rotations": len(obj.get_all_rotations())
            }
            objects_info.append(obj_info)
        return objects_info
    
    def _generate_all_actions(self):
        """Generate all possible object placements and map them to action indices."""
        action_idx = 0
        for obj_idx, obj in enumerate(self.all_objects):
            for x in range(self.x_max + 1):
                for y in range(self.y_max + 1):
                    # Get the cells that would be occupied by the object at position (x, y)
                    cells = obj.get_cells_at_position(x, y)
                    
                    # Check if the placement is valid (all cells within grid bounds)
                    if all(0 <= cx <= self.x_max and 0 <= cy <= self.y_max for cx, cy in cells):
                        self.all_actions.append((obj_idx, x, y))
                        self.action_to_placement[action_idx] = cells
                        action_idx += 1
        
        # Add "no placement" action
        self.no_placement_action = action_idx
        
    def get_action_representation(self, action: int) -> np.ndarray:
        """
        Convert an action index to its representation as a numpy array of will-be occupied cells.
        
        Args:
            action: The action index.
            
        Returns:
            A numpy array representation of the will-be occupied cells.
        """
        if action == self.no_placement_action:
            return np.zeros(self.grid_size, dtype=np.int8)
        
        action_grid = np.zeros(self.grid_size, dtype=np.int8)
        cells = self.action_to_placement[action]
        for x, y in cells:
            action_grid[x, y] = 1
        
        return action_grid
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options for resetting the environment.
            
        Returns:
            A tuple of (observation, info).
        """
        # Initialize the random number generator
        super().reset(seed=seed)
        
        # Reset the grid
        self.grid = np.zeros(self.grid_size, dtype=np.int8)
        
        # Generate all possible actions
        self._generate_all_actions()
        
        # Return the initial observation and info
        observation = self.grid.copy()
        info = {}
        
        return observation, info
    
    def _is_valid_placement(self, cells: Set[Tuple[int, int]]) -> bool:
        """
        Check if the given cells can be validly placed in the grid.
        
        Args:
            cells: The set of (x, y) coordinates to check.
            
        Returns:
            True if the placement is valid, False otherwise.
        """
        # Check if all cells are within the grid bounds
        if not all(0 <= x <= self.x_max and 0 <= y <= self.y_max for x, y in cells):
            return False
        
        # Check if any of the cells are already occupied
        for x, y in cells:
            if self.grid[x, y] == 1:
                return False
        
        return True
    
    def _place_object(self, cells: Set[Tuple[int, int]]):
        """
        Place an object in the grid at the given cells.
        
        Args:
            cells: The set of (x, y) coordinates to occupy.
        """
        for x, y in cells:
            self.grid[x, y] = 1
    
    def _calculate_occupancy(self) -> float:
        """
        Calculate the occupancy percentage of the grid.
        
        Returns:
            The percentage of occupied cells.
        """
        total_cells = (self.x_max + 1) * (self.y_max + 1)
        occupied_cells = np.sum(self.grid)
        return occupied_cells / total_cells
    
    def _check_terminal(self) -> bool:
        """
        Check if the environment is in a terminal state (no more valid placements).
        
        Returns:
            True if terminal, False otherwise.
        """
        # Check if there are any valid placements left
        for action_idx in range(len(self.all_actions)):
            cells = self.action_to_placement[action_idx]
            if self._is_valid_placement(cells):
                return False
        
        return True
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: The action to take.
            
        Returns:
            A tuple of (observation, reward, terminated, truncated, info).
        """
        if action == self.no_placement_action:
            # No placement action, immediate terminal state
            observation = self.grid.copy()
            reward = self._calculate_occupancy()
            terminated = True
            truncated = False
            info = {"occupancy": reward}
            return observation, reward, terminated, truncated, info
        
        # Get the cells to be occupied by the action
        cells = self.action_to_placement[action]
        
        # Check if the placement is valid
        if not self._is_valid_placement(cells):
            # Invalid placement, immediate terminal state with penalty
            observation = self.grid.copy()
            reward = 0.0  # Penalty for invalid placement
            terminated = True
            truncated = False
            info = {"occupancy": self._calculate_occupancy(), "invalid_placement": True}
            return observation, reward, terminated, truncated, info
        
        # Place the object in the grid
        self._place_object(cells)
        
        # Check if the environment is in a terminal state
        terminated = self._check_terminal()
        
        # Calculate the reward
        reward = self._calculate_occupancy() if terminated else 0.0
        
        # Return the observation, reward, and done flag
        observation = self.grid.copy()
        truncated = False
        info = {"occupancy": self._calculate_occupancy()}
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _draw_object(self, ax, obj, grid_size=3, title=None, highlight=False):
        """
        Draw a single object in a subplot.
        
        Args:
            ax: The matplotlib axis to draw on.
            obj: The object to draw.
            grid_size: Size of the grid to show the object in.
            title: Optional title for the subplot.
            highlight: Whether to highlight the object (e.g., for active objects).
        """
        # Create a blank grid for this object
        obj_grid = np.zeros((grid_size, grid_size), dtype=np.int8)
        
        # Get normalized cells (centered in the grid)
        cells = obj.get_cells()
        max_x = max(x for x, _ in cells)
        max_y = max(y for _, y in cells)
        offset_x = (grid_size - max_x - 1) // 2
        offset_y = (grid_size - max_y - 1) // 2
        
        # Fill the grid with the object's cells
        for x, y in cells:
            grid_x, grid_y = x + offset_x, y + offset_y
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                obj_grid[grid_x, grid_y] = 1
        
        # Create a colormap for the grid (white for empty, blue for occupied)
        cmap = ListedColormap(['white', '#0080ff' if not highlight else '#ff4000'])
        
        # Plot the grid
        ax.imshow(obj_grid.T, cmap=cmap, origin='lower', 
                 extent=[-0.5, grid_size-0.5, -0.5, grid_size-0.5])
        
        # Draw grid lines
        for x in range(grid_size):
            ax.axvline(x - 0.5, color='lightgray', linestyle='-', linewidth=1)
        for y in range(grid_size):
            ax.axhline(y - 0.5, color='lightgray', linestyle='-', linewidth=1)
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=8)
        
        # Set aspect ratio
        ax.set_aspect('equal')
    
    def render(self):
        """
        Render the environment using Matplotlib.
        
        If render_mode is 'human', displays the plot.
        If render_mode is 'rgb_array', returns an RGB array representation of the plot.
        """
        if self.render_mode is None:
            return

        # Create a colormap for the grid (white for empty, blue for occupied)
        cmap = ListedColormap(['white', '#0080ff'])  # Light blue for occupied cells
        
        # Create a new figure if one doesn't exist
        if self.fig is None:
            plt.ion()  # Turn on interactive mode for 'human' rendering
            # Create a figure with appropriate size
            self.fig = plt.figure(figsize=(10, 6))
            # Create a grid layout with main grid and objects panel
            grid_width = 3
            obj_width = 1
            self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            
            # Calculate width ratios based on number of objects
            width_ratios = [grid_width, obj_width]
            self.gs = self.fig.add_gridspec(1, 2, width_ratios=width_ratios)
            
            # Create the axes
            self.main_ax = self.fig.add_subplot(self.gs[0, 0])
            self.objects_ax = self.fig.add_subplot(self.gs[0, 1])
        
        # Clear the previous plots
        self.main_ax.clear()
        self.objects_ax.clear()
        
        # Plot the main grid
        self.main_ax.imshow(self.grid.T, cmap=cmap, origin='lower', 
                         extent=[-0.5, self.x_max + 0.5, -0.5, self.y_max + 0.5])
        
        # Draw grid lines
        for x in range(self.x_max + 2):
            self.main_ax.axvline(x - 0.5, color='lightgray', linestyle='-', linewidth=1)
        for y in range(self.y_max + 2):
            self.main_ax.axhline(y - 0.5, color='lightgray', linestyle='-', linewidth=1)
        
        # Set axis labels
        self.main_ax.set_xlabel('X')
        self.main_ax.set_ylabel('Y')
        
        # Set the title with occupancy information
        occupancy = self._calculate_occupancy()
        self.main_ax.set_title(f'Box Tiling Environment - Occupancy: {occupancy:.2%}')
        
        # Set ticks at grid cell centers
        self.main_ax.set_xticks(range(self.x_max + 1))
        self.main_ax.set_yticks(range(self.y_max + 1))
        
        # Set limits to ensure the grid is fully visible with borders
        self.main_ax.set_xlim(-0.5, self.x_max + 0.5)
        self.main_ax.set_ylim(-0.5, self.y_max + 0.5)
        
        # Ensure the aspect ratio is equal
        self.main_ax.set_aspect('equal')
        
        # Display available objects
        self.objects_ax.set_title("Available Objects")
        
        # Create a grid for the objects panel
        num_objects = len(self.base_objects)
        if num_objects > 0:
            # Calculate reasonable grid layout
            rows = min(6, num_objects)
            
            # Create mini-grids for each object
            grid_size = 3  # Size of each object grid
            obj_grid = np.zeros((rows * grid_size, grid_size), dtype=np.int8)
            
            # Fill in the object grids
            for i, obj in enumerate(self.base_objects):
                if i < rows:  # Only display up to max rows
                    cells = obj.get_cells()
                    # Center the object in its grid
                    max_x = max((x for x, _ in cells), default=0)
                    max_y = max((y for _, y in cells), default=0)
                    offset_x = (grid_size - max_x - 1) // 2
                    offset_y = (grid_size - max_y - 1) // 2
                    
                    # Place the object in the overall grid
                    row_offset = i * grid_size
                    for x, y in cells:
                        grid_x, grid_y = x + offset_x, y + offset_y
                        if (0 <= grid_x < grid_size and 
                            0 <= grid_y < grid_size and 
                            0 <= row_offset + grid_x < rows * grid_size):
                            obj_grid[row_offset + grid_x, grid_y] = 1
            
            # Display the objects grid
            self.objects_ax.imshow(obj_grid.T, cmap=cmap, origin='lower',
                                 extent=[-0.5, rows * grid_size - 0.5, -0.5, grid_size - 0.5])
            
            # Draw grid lines
            for x in range(rows * grid_size + 1):
                if x % grid_size == 0:  # Thicker lines between objects
                    self.objects_ax.axvline(x - 0.5, color='black', linestyle='-', linewidth=1)
                else:
                    self.objects_ax.axvline(x - 0.5, color='lightgray', linestyle='-', linewidth=0.5)
                    
            for y in range(grid_size + 1):
                self.objects_ax.axhline(y - 0.5, color='lightgray', linestyle='-', linewidth=0.5)
            
            # Add object numbers
            for i in range(min(rows, num_objects)):
                self.objects_ax.text(i * grid_size + grid_size // 2, -0.25, f"#{i}",
                                  horizontalalignment='center', verticalalignment='top')
            
            # Remove ticks
            self.objects_ax.set_xticks([])
            self.objects_ax.set_yticks([])
        else:
            # No objects to display
            self.objects_ax.text(0.5, 0.5, "No objects available",
                               horizontalalignment='center', verticalalignment='center')
            self.objects_ax.set_xticks([])
            self.objects_ax.set_yticks([])
        
        # Update the display
        self.fig.tight_layout()
        plt.draw()
        
        # Display the plot
        if self.render_mode == 'human':
            plt.pause(1 / self.metadata['render_fps'])  # Pause to respect render_fps
        
        # For rgb_array mode, convert the plot to an RGB array
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            image_from_plot = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image_from_plot
    
    def close(self):
        """Close the environment and release resources."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.main_ax = None
            self.objects_ax = None
        plt.close('all')  # Close any other open figures 