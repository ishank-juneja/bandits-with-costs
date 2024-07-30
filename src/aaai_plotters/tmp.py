import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import argparse
import pathlib
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

# Creating sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = x ** 2
y4 = np.log(x + 1)

# Creating the figure and subplots
fig, axs = plt.subplots(2, 2)  # 2x2 grid of axes
fig.tight_layout(pad=3.0)       # Adds padding between plots

# Plotting on the first subplot
axs[0, 0].plot(x, y1, 'tab:blue')
axs[0, 0].set_title('Sine Wave')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('sin(x)')

# Plotting on the second subplot
axs[0, 1].plot(x, y2, 'tab:orange')
axs[0, 1].set_title('Cosine Wave')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('cos(x)')

# Plotting on the third subplot
axs[1, 0].plot(x, y3, 'tab:green')
axs[1, 0].set_title('Quadratic')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('x^2')

# Plotting on the fourth subplot
axs[1, 1].plot(x, y4, 'tab:red')
axs[1, 1].set_title('Logarithmic')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('log(x+1)')

# Display the plots
plt.show()
