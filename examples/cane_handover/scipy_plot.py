import tensorboardX
import tensorboard.plugins.scalar
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from scipy.optimize import minimize
fig = plt.figure()
# Define the loss function
def loss(x):
    return x**2 + 2*x + 1

# Create a TensorBoard scalar writer
writer = tensorboardX.SummaryWriter()

# Use the minimize function to find the minimum of the loss function
res = minimize(loss, x0=10, callback=lambda x: writer.add_scalar('minimization_value', x) ) 

# Extract the minimization values from the result
minimization_values = res.fun
print(minimization_values)
# Add the minimization values to TensorBoard
writer.add_figure('matplotlib', fig)
# Close the TensorBoard writer
writer.close()
