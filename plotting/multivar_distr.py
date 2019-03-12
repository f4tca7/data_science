# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
auto_all = pd.read_csv('../datasets/auto-mpg.csv')
auto = auto_all[['mpg', 'hp', 'origin']]

# ------------ plot joint distributions ------------ #
# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x='hp', y='mpg', data=auto)
plt.show()
sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')
plt.show()

# ------------ plot pairwise distributions ------------ #

# Plot the pairwise joint distributions from the DataFrame 
sns.pairplot(auto)
plt.show()
# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto, kind='reg', hue='origin')
plt.show()

# ------------ plot correlation heatmaps ------------ #

cov_data = auto_all.loc[:, ['mpg', 'hp', 'weight', 'accel', 'displ']]
cov_matrix = cov_data.corr()
# Print the covariance matrix
print(cov_matrix)
# Visualize the covariance matrix using a heatmap
sns.heatmap(cov_matrix)
plt.show()
