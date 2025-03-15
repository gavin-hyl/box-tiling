import gymnasium as gym
import time
import random
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.abspath("src"))

# Import from src module
from src import Object

gym.register(
    id='CustomBoxTiling-v0',
    entry_point='src:BoxTilingEnv',
)

def create_custom_object(name, cells):
    """Create a custom object with the given cells."""
    obj = Object(cells)
    obj.name = name
    return obj

def custom_objects_demo():
    """
    Demonstrate the environment with custom objects.
    """
    # Create custom objects
    custom_objects = [
        create_custom_object("3x3 Square", {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)}),
        create_custom_object("U-Shape", {(0, 0), (1, 0), (2, 0), (0, 1), (2, 1)}),
        create_custom_object("Z-Shape", {(0, 0), (1, 0), (1, 1), (2, 1)}),
        create_custom_object("T-Shape", {(0, 0), (1, 0), (2, 0), (1, 1)}),
        create_custom_object("L-Shape", {(0, 0), (0, 1), (0, 2), (1, 0)}),
        create_custom_object("2x2 Square", {(0, 0), (0, 1), (1, 0), (1, 1)}),
    ]
    
    # Create the environment with custom objects
    env = gym.make('CustomBoxTiling-v0', 
                  x_max=8, 
                  y_max=8, 
                  available_objects=custom_objects,
                  render_mode="human")
    
    # Reset the environment
    observation, info = env.reset()
    
    # Print available objects info
    unwrapped_env = env.unwrapped
    objects_info = unwrapped_env.get_available_objects_info()
    
    print("\nAvailable Custom Objects:")
    for i, obj_info in enumerate(objects_info):
        obj_name = getattr(custom_objects[i], 'name', f"Object {i}")
        print(f"{obj_name}: {obj_info['cells']} ({obj_info['rotations']} rotations)")
    
    # Random agent demo with the custom objects
    terminated = False
    truncated = False
    total_reward = 0
    steps = 0
    
    print("\nStarting placement...")
    time.sleep(2)
    
    while not (terminated or truncated):
        # Sample a random action (ensure it's valid)
        valid_actions = [action for action in unwrapped_env.action_to_placement.keys() 
                        if unwrapped_env._is_valid_placement(unwrapped_env.action_to_placement[action])]
        
        if not valid_actions:  # No valid actions left
            action = unwrapped_env.no_placement_action
        else:
            action = random.choice(valid_actions)
        
        # Take a step
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Print state and reward
        print(f"\nStep {steps + 1}:")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        print(f"Occupancy: {info['occupancy']:.2%}")
        
        total_reward += reward
        steps += 1
        
        # Wait to better visualize the steps
        time.sleep(1.0)
    
    print("\nEpisode complete!")
    print(f"Total steps: {steps}")
    print(f"Total reward: {total_reward}")
    print(f"Final occupancy: {info['occupancy']:.2%}")
    
    # Keep the visualization open for a while to observe the final state
    time.sleep(5.0)
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    print("Running custom objects demo...")
    custom_objects_demo() 