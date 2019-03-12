# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
stocks = pd.read_csv('../datasets/stocks.csv', index_col='Date', parse_dates=True)
aapl = stocks['AAPL']
ibm = stocks[['IBM']]
csco = stocks[['CSCO']]
msft = stocks[['MSFT']]

# ------------ simple lineplot ------------ #

plt.plot(aapl, color='blue', label='AAPL')
plt.plot(ibm, color='green', label='IBM')
plt.plot(csco, color='red', label='CSCO')
plt.plot(msft, color='magenta', label='MSFT')
# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')
# Specify the orientation of the xticks
plt.xticks(rotation=60)
plt.show()

# ------------ multiple time series slices ------------ #

# Plot the series in the top subplot in blue
plt.subplot(2,1,1)
plt.xticks(rotation=45)
plt.title('AAPL: 2001 to 2011')
plt.plot(aapl, color='blue')
view = aapl['2007':'2008']
plt.subplot(2,1,2)
plt.xticks(rotation=45)
plt.title('AAPL: 2007 to 2008')
plt.plot(view, color='black')
plt.tight_layout()
plt.show()

# ------------ multiple time series slices ------------ #

# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']
plt.subplot(2,1,1)
plt.plot(view, color='red')
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.xticks(rotation=45)
view = aapl['2008-01']
plt.subplot(2,1,2)
plt.plot(view, color='green')
plt.title('AAPL: Jan. 2008')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------ inset view ------------ #

# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')
plt.axes ((0.25, 0.5, 0.35, 0.35))
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()

# ------------ moving averages ------------ #

mean_30 = aapl.rolling(window=30).mean()
mean_75 = aapl.rolling(window=75).mean()
mean_125 = aapl.rolling(window=125).mean()
mean_250 = aapl.rolling(window=250).mean()

# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2,2,1)
plt.plot(mean_30, color='green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 75-day moving average in the top right subplot in red
plt.subplot(2,2,2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

# Plot the 125-day moving average in the bottom left subplot in magenta
plt.subplot(2, 2, 3)
plt.plot(mean_125, 'magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2, 2, 4)
plt.plot(mean_250, 'cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')

# Display the plot
plt.show()

# ------------ moving std dev ------------ #


std_30 = aapl.rolling(30).std()
std_75 = aapl.rolling(75).std()
std_125 = aapl.rolling(125).std()
std_250 = aapl.rolling(250).std()

plt.plot(std_30, color='red', label='30d')
plt.plot(std_75, color='cyan', label='75d')
plt.plot(std_125, color='green', label='125d')
plt.plot(std_250, color='magenta', label='250d')
plt.legend(loc='upper left')
plt.title('Moving standard deviations')
plt.show()
