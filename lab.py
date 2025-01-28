import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

daysOfExercise = np.array([140,155,159,179,192,200,212])
weight = np.array([60,62,67,70,71,72,75])

slope, intercept, r_value, p_value, std_err = linregress(daysOfExercise, weight)

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R_squared: {r_value**2}")

reg_line = slope * daysOfExercise + intercept

plt.scatter(daysOfExercise, weight, color = 'green', label = 'Data')
plt.plot(daysOfExercise, reg_line, color = 'red', label = 'Regression Line')

plt.xlabel('Days of Exercise')
plt.ylabel('Weight Loss')
plt.title('Weight Progress')
plt.legend()

plt.show()