# Box Tiling Environment

A 2D grid world environment for placing and tiling objects, implemented using Gymnasium (formerly Gym).

## Description

This environment simulates a 2D grid world with a rectangular boundary where different objects can be placed. The objects are defined by the set of grid cells they occupy and can be rotated. The goal is to maximize the occupancy of the grid.

## Features

- 2D grid world with configurable dimensions (from (0,0) to (x_max, y_max))
- Objects defined by their occupied grid cells (e.g., a 2x2 square is {(0,0), (0,1), (1,0), (1,1)})
- Support for object rotation
- State representation as a numpy array of currently occupied cells in the world
- Action representation as a numpy array of cells that would be occupied by placing an object
- Reward is 0 if non-terminal and the occupancy percentage if terminal
- Visualization using Matplotlib

## Installation

The project uses traditional relative imports rather than being a formal Python package. To use it:

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

```python
import gymnasium as gym
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath("src"))

# Import from src module
from src import BoxTilingEnv

# Register the environment with Gymnasium
gym.register(
    id='BoxTiling-v0',
    entry_point='src:BoxTilingEnv',
)

# Create the environment
env = gym.make('BoxTiling-v0', x_max=10, y_max=10)

# Reset the environment
obs, info = env.reset()

# Take an action
action = env.action_space.sample()  # Random action
obs, reward, terminated, truncated, info = env.step(action)
```

## Running the Example

The repository includes an example script that demonstrates the environment with both a random agent and manual object placement:

```bash
python example.py
```

Then choose option 1 for a random agent demo or option 2 for manual placement demo.

## Project Structure

- `src/`: Main source directory
  - `__init__.py`: Makes the module importable
  - `box_tiling_env.py`: Main environment class
  - `objects.py`: Object classes and rotation logic
- `example.py`: Example script demonstrating the environment
- `requirements.txt`: Dependencies list

## Requirements

- Python 3.7+
- gymnasium>=0.28.1
- numpy>=1.22.0
- matplotlib>=3.5.0 (for rendering)