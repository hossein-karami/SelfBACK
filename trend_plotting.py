# -*- coding: utf-8 -*-
import copy as cp
import numpy as np
import scipy as sp
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import matplotlib as mpl

from attributes import *

cmap = mpl.cm.autumn


    

dataset_array = np.load('./dataset_file.npy')

## ---- deleting columns of dataset which aren't used in plotting
temp_array = cp.deepcopy(dataset_array)
plotting_attributes = outcomes_generators
for attr in dataset_array[0]:
    if attr not in plotting_attributes:
        deleting_index = list(temp_array[0]).index(attr)
        temp_array = sp.delete(temp_array, deleting_index, 1)

dataset_array = temp_array



## ---- deleting rows with missing data
i=0
for row in dataset_array:
    
    if not any(elem is None for elem in row) and i==0:
       
        temp = row
        i+=1
    elif not any(elem is None for elem in row) and i!=0:
        temp = np.vstack((temp, row))
    
dataset_array = temp



## ---- sorting the remainder based on first week respond value
sorting_attr = outcomes_generators[0]
sorting_attr_index = list(dataset_array[0]).index(sorting_attr)

dataset_sorted = sorted(dataset_array[1:], key=lambda x: x[sorting_attr_index])



## ---- sorting data based on mean value of the pain intensity for each patient to the data set.
mean_column = np.mean(dataset_sorted, axis=1)
mean_column_rounded = np.round(mean_column.astype(np.double))
dataset = np.column_stack((mean_column_rounded, dataset_sorted))

sorting_attr_index = 0
dataset_sorted = sorted(dataset, key=lambda x: x[sorting_attr_index])
dataset_sorted = np.asarray(dataset_sorted)




## ---- plotting the sorted data (one plot for each value of sorting attribute)
start_point_value = dataset_sorted[0,sorting_attr_index]

plotting_data = dataset_sorted[0]

colors =['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
for row in dataset_sorted[1:]:
    if row[sorting_attr_index] == start_point_value:
        plotting_data = np.vstack((plotting_data, row))
    elif row[sorting_attr_index] != start_point_value:
        fig = plt.figure(1, figsize=(14,6))
        
        for i, y in enumerate(plotting_data):
            plt.plot(np.arange(0,56,4), y[1:], color=colors[i%8])  #cmap(i / float(8))
            
        plt.title("Trend, when the average of pain intensity during survey is "+ str(start_point_value) +'. Number of instances: '+ str(i))
        plt.xlabel("Week number")
        plt.ylabel("Pain level")
        fig.savefig('Trend_plots/plot for average pain level-' + str(start_point_value) + '.png', bbox_inches='tight', dpi=150) #
        plt.close()
       
        start_point_value = row[sorting_attr_index]
        plotting_data = row
        print(plotting_data)

fig = plt.figure(1, figsize=(14,6))
for i, y in enumerate([plotting_data]):
    print(y)
    plt.plot(np.arange(0,56,4), y[1:], color=colors[i%8])
        
plt.title("Trend, when the average of pain intensity during survey is "+ str(start_point_value) +'. Number of instances: '+ str(i+1))
plt.xlabel("Week number")
plt.ylabel("Pain level")

fig.savefig('Trend_plots/plot for average pain level-' + str(start_point_value) + '.png', bbox_inches='tight', dpi=150) #
plt.close()