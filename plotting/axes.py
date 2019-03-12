import matplotlib.pyplot as plt
import pandas as pd

# Load data
df = pd.read_csv('../datasets/percent-bachelors-degrees-women-usa.csv')
physical_sciences = df['Physical Sciences']
computer_science = df['Computer Science']
year = df['Year']
education = df['Education']
health = df['Health Professions']

plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Limit axes
# plt.xlim([1990, 2010])
# plt.ylim([0, 50])
# Same as plt.axis((1990,2010,0, 50))

# Add a legend at the lower center
plt.legend(loc='lower center')

# Add a black arrow annotation at cs_max peak
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', arrowprops=dict(facecolor='black'), xy=(yr_max, cs_max), xytext = (yr_max+5, cs_max+5))

# Add a title and labels
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Set the style to 'ggplot'
plt.style.use('ggplot')
# 2x2 layout
plt.subplot(2, 2, 1) 
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()