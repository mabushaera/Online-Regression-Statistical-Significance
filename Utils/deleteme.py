import matplotlib.pyplot as plt
import numpy as np

# Set the ggplot style
plt.style.use('ggplot')

# Define the equations of two lines: y = m*x + b
def line1(x):
    return 0.9 * x + 2

def line2(x):
    return 0.3 * x + 4

# Generate x values
x_values = np.linspace(-10, 10, 400)

# Calculate corresponding y values for each line
y_line1 = line1(x_values)
y_line2 = line2(x_values)

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_line1, label='Incremental Model', color='green')
plt.plot(x_values, y_line2, label='Base Model', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Two Averages of the Base Model and the Incremental Model')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()
