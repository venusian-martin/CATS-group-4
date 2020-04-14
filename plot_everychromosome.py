# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd


file = open('Train_call.txt')
input = file.readlines()


for j in range(1,24):
    raw_data = {}
    #raw_data['region_start'] = []
    #raw_data['region_end'] = []
    raw_data['-1'] = []
    raw_data['0'] = []
    raw_data['1'] = []
    raw_data['2'] = []
    length_every_chromosome = 0
    for line in input:
        count_min1 = 0
        count_0 = 0
        count_1 = 0
        count_2 = 0
        line = line.split('\t')
        #raw_data['region_start'] = line [1]
        #raw_data['region_end'] = [2]
        if line[0] == str(j) :
            length_every_chromosome +=1
            for i in range(4,len(line)):
                if line[i] == '-1':
                    count_min1 += 1
                elif line[i] == '0':
                    count_0 += 1
                elif line[i] == '1':
                    count_1 += 1
                elif line[i] == '2':
                    count_2 += 1

            raw_data['-1'].append(float(count_min1))
            raw_data['0'].append(float(count_0))
            raw_data['1'].append(float(count_1))
            raw_data['2'].append(float(count_2))

            #print('chromosome', j, count_min1, count_0, count_1, count_2)
    print(j,raw_data)
    print(length_every_chromosome)
    # Data
    r = range(length_every_chromosome)
    df = pd.DataFrame(raw_data)
    # From raw value to percentage
    totals = [i + j + k + l for i, j, k, l in zip(df['-1'], df['0'], df['1'], df['2'])]

    greenBars = [i / j * 100 for i, j in zip(df['-1'], totals)]
    orangeBars = [i / j * 100 for i, j in zip(df['0'], totals)]
    blueBars = [i / j * 100 for i, j in zip(df['1'], totals)]
    redBars = [i / j * 100 for i, j in zip(df['2'], totals)]
    # plot
    barWidth = 0.85

    # Create green Bars
    plt.bar(r, greenBars, color='lightgreen', edgecolor='white', width=barWidth, label='-1')
    # Create orange Bars
    plt.bar(r, orangeBars, bottom=greenBars, color='orange', edgecolor='white', width=barWidth, label='0')
    # Create blue Bars
    plt.bar(r, blueBars, bottom=[i + j for i, j in zip(greenBars, orangeBars)], color='lightblue', edgecolor='white',
            width=barWidth, label='1')
    # red Bars
    plt.bar(r, redBars, bottom=[i + j + k for i, j, k in zip(greenBars, orangeBars, blueBars)], color='violet',
            edgecolor='white',
            width=barWidth, label='2')

    # Custom x axis
    #plt.xticks(r)
    plt.xlabel("chromosome region")
    plt.title(j)

    # Show graphic
    plt.legend(loc='upper right')
    plt.show()
    plt.savefig('/Users/pingyi/Desktop/CAT/plot')
