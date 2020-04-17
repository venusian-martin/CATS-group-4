# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd



file = open('data')
input = file.readlines()

raw_data = {}
raw_data['-1'] = []
raw_data['0'] = []
raw_data['1'] = []
raw_data['2'] = []

for line in input:
    line = line.rstrip('\n')
    line = line.split('\t')

    raw_data['-1'].append(float(line[2]))
    raw_data['0'].append(float(line[3]))
    raw_data['1'].append(float(line[4]))
    raw_data['2'].append(float(line[5]))
    #print(raw_data)

# Data
r = range(len(input))
df = pd.DataFrame(raw_data)

# From raw value to percentage
totals = [i + j + k + l for i, j, k, l in zip(df['-1'], df['0'], df['1'], df['2'])]

greenBars = [i / j * 100 for i, j in zip(df['-1'], totals)]
orangeBars = [i / j * 100 for i, j in zip(df['0'], totals)]
blueBars = [i / j * 100 for i, j in zip(df['1'], totals)]
redBars = [i / j * 100 for i, j in zip(df['2'], totals)]

# plot
barWidth = 0.85
names = range(1,24)
# Create green Bars
plt.bar(r, greenBars, color='lightgreen', edgecolor='white',  width=barWidth, label='-1')
# Create orange Bars
plt.bar(r, orangeBars, bottom =greenBars, color='orange', edgecolor='white',width=barWidth,label='0')
# Create blue Bars
plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], color='lightblue',edgecolor='white',
        width=barWidth,label='1')
# red Bars
plt.bar(r, redBars, bottom=[i + j + k for i, j ,k in zip(greenBars, orangeBars, blueBars)], color='violet',edgecolor='white',
        width=barWidth,label='2')

# Custom x axis
plt.xticks(r, names)
plt.xlabel("chromosome")
plt.title('All chromosome')


# Show graphic
plt.legend(loc='upper right')
plt.show()