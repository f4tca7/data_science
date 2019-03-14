import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

df = pd.read_csv('../datasets/gm_2008_region.csv')
df = df.drop('Region', axis=1)
# Create arrays for features and target variable
y = df['life'].values
X = df.drop('life', axis=1).values


# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)
df = df.drop('life', axis=1)
df_columns = df.columns
print(df_columns)
# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()


