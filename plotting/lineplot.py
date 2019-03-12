import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('../datasets/percent-bachelors-degrees-women-usa.csv')
physical_sciences = df['Physical Sciences']
computer_science = df['Computer Science']
year = df['Year']
education = df['Education']
health = df['Health Professions']

# Simple Line Plot
plt.plot(year, physical_sciences, color='blue')
plt.plot(year, computer_science, color='red')
plt.show()

# 1x2 subplot
plt.subplot(1, 2, 1)
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
plt.subplot(1, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')
plt.tight_layout()
plt.show()

# 2x2 subplot
plt.subplot(2, 2, 1)
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')
plt.tight_layout()
plt.show()