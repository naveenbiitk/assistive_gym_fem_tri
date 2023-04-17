import numpy as np
from scipy.optimize import minimize
import tensorboardX
import tensorboard.plugins.scalarexamples/cane_handover/scipy_plot_2.py
# Define the loss function
def loss(x):
    return x**2 + 2*x + 1

writer = tensorboardX.SummaryWriter()
# Define the callback function
def callback(x):
    # Save the minimization value
    minimization_values.append(loss(x))
    # Save the index of the iteration
    iteration_indices.append(len(minimization_values) - 1)
    writer.add_scalar('minimization_value', loss(x),len(minimization_values) - 1)

# Initialize lists to store the minimization values and iteration indices
minimization_values = []
iteration_indices = []

# Use the minimize function to find the minimum of the loss function
res = minimize(loss, x0=100, callback=callback)

# Convert the lists to numpy arrays
minimization_values = np.array(minimization_values)
iteration_indices = np.array(iteration_indices)

# Print the results
print(res)
print(minimization_values)
print(iteration_indices)
writer.close()
