# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp



def filtering_for_main_problem(data_array):

## filtering the main prblem for "Rygg" if it is necessary:   

    if 'hovedproblem' in data_array[0]:
        filtered_data = data_array[0]
          
        hovedproblem_index = list(filtered_data).index('hovedproblem')
        for row in data_array:
            if row[hovedproblem_index] == 3:
                filtered_data = np.vstack([filtered_data,row])
                
        filtered_data = sp.delete(filtered_data, hovedproblem_index, 1)
    else:
        filtered_data = data_array[:]

## adding BMI to the dataset:
    if 'BMI' not in data_array[0]:
        
        height_index = list(filtered_data[0]).index('hoyde')
        weight_index = list(filtered_data[0]).index('vekt')
        BMI_column = []
        for row in filtered_data[1:]:
            if row[height_index] != None and row[weight_index] != None:
                
                BMI = row[weight_index] / ((row[height_index]/100)**2)
                BMI_column.append(BMI)
            else:
                BMI_column.append(None)
                
        BMI_column.insert(0, 'BMI') 
        filtered_data = np.column_stack((filtered_data, BMI_column))
    
    return filtered_data
    
 
    



## Reducing dataset to attributes which are used in different analyses:

def grouping_by_attributes(filtered_data, attribute_set_for_analysis):
    grouped_data = filtered_data[:,0]
    
     
    for attribute in attribute_set_for_analysis:
       
        attribute_index = list(filtered_data[0]).index(attribute)
        if attribute_index < len(filtered_data[0]):
            grouped_data = np.column_stack([grouped_data, filtered_data[:,attribute_index]])
            
    temp = grouped_data[0]
    for row in grouped_data[1:]:
        ID_index = list(filtered_data[:,0]).index(row[0])
        num_of_data = sum(x is not None for x in filtered_data[ID_index])
        
        if num_of_data > (len(attribute_set_for_analysis)/2):
            temp = np.vstack((temp, row))
            
    grouped_data = temp  
    return grouped_data   







## Normalizing dataset to values between 0 and 1:

def normalizing_data(grouped_data, nominal_attributes):
    normalized_data = grouped_data[:,0]
    attribute_titles = grouped_data[0]
    temp1 = list(grouped_data)[:]
       
    for index in range(1,len(attribute_titles)):
        column = np.asarray(temp1)[:,index]
        
        if attribute_titles[index] not in nominal_attributes:
            temp =[]
            for item in column[1:]:
                if item != None:
                    temp.append(item)
            
            min_value = min(temp) 
            max_value = max(temp)
            delta = max_value - min_value
            
            for i in range(1,len(column)):
                if column[i] != None:
                    column[i] = (column[i]-min_value) / delta
                    if column[i] == 0:
                        column[i] = 0.000001
            
        normalized_data = np.column_stack((normalized_data, column))
   
    return normalized_data

  