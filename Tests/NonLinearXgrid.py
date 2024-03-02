import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def create_focused_non_linear_grid(minH0, maxH0, center=70, num_samples=10):
    """
    Create a non-linear grid with a focus around a specific center value.

    Parameters:
    - minH0: Minimum value of the grid.
    - maxH0: Maximum value of the grid.
    - center: The value around which the grid will be focused.
    - num_samples: Number of points in the grid.

    Returns:
    - A numpy array containing the non-linear grid.
    """
    # Generate a linear grid
    linear_grid = np.linspace(0.025, 0.99, num_samples)

    # Calculate the distance of each point in the linear grid from the center
    non_linear_grid = norm.ppf(linear_grid, loc = 70, scale = (maxH0-70)/3)
    # non_linear_grid = normal.ppf(linear_grid)

    return non_linear_grid

# Define the range
minH0 = 50
maxH0 = 100

# Create the non-linear grid
non_linear_grid = create_focused_non_linear_grid(minH0, maxH0, center=70, num_samples=100)

# Visualization
plt.figure(figsize=(10, 2))
plt.plot(non_linear_grid, np.zeros_like(non_linear_grid), 'x', label='Non-linear Grid')
plt.ylim(-0.1, 0.1)
plt.yticks([])
plt.legend()
plt.title("Non-linear Grid Centered at 70")
plt.show()

# Output the first 10 values for inspection
# non_linear_grid[:10]