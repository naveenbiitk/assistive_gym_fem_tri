import multipyparallel as ipp
import pybullet as p
import ipywidgets as widgets
from IPython.display import display
import os

# Create a client to use for parallel execution
client = ipp.Client()

# Load the parallel view
view = client.load_balanced_view()

# Define a function that creates and evaluates a PyBullet environment
def evaluate_environment():
    # Create a PyBullet client in non-rendering mode
    p.connect(p.DIRECT)
    
    # Load a plane to act as the ground
    p.loadURDF("plane.urdf")
    
    # Create a box
    box_id = p.loadURDF("box.urdf", [0, 0, 1])
    
    # Run the simulation for 100 steps
    for i in range(100):
        p.stepSimulation()
        
    # Calculate the position of the box
    pos, orn = p.getBasePositionAndOrientation(box_id)
    
    # Disconnect the PyBullet client
    p.disconnect()
    
    # Return the position of the box
    return pos

# Use map to execute the function in parallel
positions = view.map(evaluate_environment, range(8))

# Wait for the positions to be calculated
positions.wait()

# Print the positions of the boxes in each environment
print(positions.result())

