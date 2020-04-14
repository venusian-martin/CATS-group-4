# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd


file = open('Train_call.txt')
input = file.readlines()
#print(input[0])


for j in range(1,24):
    count_min1=0
    count_0 = 0
    count_1 = 0
    count_2 = 0

    for line in input:
        line = line.split('\t')
        if line[0] == str(j) :
            for i in range(4,len(line)):
                if line[i] == '-1':
                    count_min1 += 1
                elif line[i] == '0':
                    count_0 += 1
                elif line[i] == '1':
                    count_1 += 1
                elif line[i] == '2':
                    count_2 += 1
    all=count_min1 + count_2 + count_0 + count_1
    percentagea_min1 = float(count_min1)/float(all)
    percentagea_1 = float(count_1)/float(all)
    percentagea_0 = float(count_0)/float(all)
    percentagea_2 = float(count_2)/float(all)

    print('chromosome','\t',j,'\t',count_min1,'\t',count_0,'\t',count_1,'\t',count_2)

