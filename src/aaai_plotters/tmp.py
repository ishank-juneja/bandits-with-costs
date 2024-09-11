import matplotlib.pyplot as plt
import numpy as np

# Generate x values equi-spaced between 0.01 and 0.15
x_values = np.linspace(0.01, 0.15, 10)

# Generate y values (can be any function of x, here using y = x for simplicity)
y_values = x_values

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(x_values, y_values, marker='o', linestyle='-', color='b')

# Adding markers with numerical values as annotations
for i, value in enumerate(x_values):
    plt.text(value, value, f'{value:.2f}', fontsize=12, ha='right')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Equi-spaced Markers with Numerical Values')

# Show the plot
plt.grid(True)
plt.show()
