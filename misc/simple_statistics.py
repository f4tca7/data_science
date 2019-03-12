# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


# Load data
auto = pd.read_csv('../datasets/auto-mpg.csv')
df_all_states = pd.read_csv('../datasets/2008_all_states.csv')
versicolor_petal_length = [4.7, 4.5, 4.9, 4. , 4.6, 4.5, 4.7, 3.3, 4.6, 3.9, 3.5, 4.2, 4. , 4.7, 3.6, 4.4, 4.5, 4.1, 4.5, 3.9, 4.8, 4. , 4.9, 4.7, 4.3, 4.4, 4.8, 5. , 4.5, 3.5, 3.8, 3.7, 3.9, 5.1, 4.5, 4.5, 4.7, 4.4, 4.1, 4. , 4.4, 4.6, 4. , 3.3, 4.2, 4.2, 4.2, 4.3, 3. , 4.1]

# ------------ ecdf ------------ #

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y

# ------------ Pearson correlation coefficient ------------ #

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)

    # Return entry [0,1]
    return corr_mat[0,1]    

# ------------ var, stddev ------------ #

# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)
diff_sq = differences ** 2
variance_explicit = np.mean(diff_sq)
variance_np = np.var(versicolor_petal_length)
print(variance_explicit)
print(variance_np)

variance = np.var(versicolor_petal_length)
print(np.sqrt(variance))
print(np.std(versicolor_petal_length))

# ------------ percentiles, ecdf ------------ #

mpg = auto['mpg']
percentiles = np.array([2.5, 25, 50, 75, 97.5])
ptiles = np.percentile(mpg, percentiles)
x_vers, y_vers = ecdf(mpg)
plt.plot(x_vers, y_vers, '.')

plt.xlabel('mpg')
plt.ylabel('ECDF')
plt.plot(ptiles, percentiles/100, marker='D', color='red',
         linestyle='none')

plt.show()

# ------------ boxplot ------------ #

sns.boxplot(x='east_west', y='dem_share', data=df_all_states)
plt.xlabel('region')
plt.ylabel('percent of vote for Obama')
plt.show()

# ------------ scatter ------------ #

plt.plot(df_all_states.total_votes/1000, df_all_states.dem_share, marker='.', linestyle='none')
plt.xlabel('total votes (thousands)')
plt.ylabel('percent of vote for Obama')
plt.show()