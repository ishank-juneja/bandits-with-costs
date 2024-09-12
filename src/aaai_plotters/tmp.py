import matplotlib.pyplot as plt
import numpy as np

# Sample data
x_markers = np.arange(12)  # x-markers from 0 to 11
y_values = np.random.rand(12, 50)  # Random y-values for each x-marker, 50 per marker

# Create the plot
plt.figure(figsize=(10, 6))
for i, x in enumerate(x_markers):
    plt.scatter([x] * len(y_values[i]), y_values[i], alpha=0.5)  # Plot all y-values for each x

# Customizing the plot
plt.title('Scatter Plot of Multiple Y-values for Each X-marker')
plt.xlabel('X-marker index')
plt.ylabel('Y-values')
plt.grid(True)

# Show the plot
plt.show()
