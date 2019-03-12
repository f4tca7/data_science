import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('../datasets/auto-mpg.csv')
hp = df['hp']
mpg = df['mpg']

#####################################
# ------------ 2D hist ------------ #
#####################################

plt.hist2d(hp, mpg, bins=(20,20), range=((40, 235), (8, 48)))
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()

#########################################
# ------------ 2D hex hist ------------ #
#########################################

plt.hexbin(hp, mpg, gridsize=(15,12), extent=(40, 235, 8, 48))
plt.colorbar()
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()