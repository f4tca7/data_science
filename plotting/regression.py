# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
auto = pd.read_csv('../datasets/auto-mpg.csv')

# ------------ linear regression ------------ #

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)
plt.show()

# ------------ residual plot ------------ #

# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')
plt.show()

# ------------ higher-order regression ------------ #

# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')
# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='order 1')
# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='green', label='order 2', order=2)
# Add a legend and display the plot
plt.legend(loc='upper right')
plt.show()

# ------------ grouping regression by hue ------------ #

# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='hp', data=auto, palette='Set1', hue='origin')
plt.show()

# ------------ grouping regression by row or col ------------ #

# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x='weight', y='hp', data=auto, palette='Set1', row='origin')
plt.show()


