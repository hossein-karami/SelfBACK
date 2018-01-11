# -*- coding: utf-8 -*-
from random import randint
from collections import Counter

import numpy as np



## ____________________________ Making complete data set by removing incomplete distances ________________________

def del_incomplete_data (grouped_data):
    complete_data = grouped_data[0]
    
    for row in grouped_data[1:]:
        
        condition1 = np.equal(row, None)
        if True not in condition1: 
            complete_data = np.vstack ((complete_data, row))
    
    return complete_data



## ____________________________ Completing by mean (or most repeated) values of attributes ________________________

def completing_by_mean_value (grouped_data):
    
    attribute_list = grouped_data[0, 1:]
    mean_value_vector = [0]
    
    for attribute in attribute_list:
        
        
        attribute_index = list(grouped_data[0]).index(attribute)
        
        lst = list(grouped_data[1: , attribute_index])
        num_of_values = sum(x is not None for x in lst)
        sum_of_column = sum(filter(None, lst))
        
        mean_of_column = sum_of_column/num_of_values
        List = [x for x in lst if x is not None]
        
        closest_to_the_mean = min(List, key=lambda x:abs(x-mean_of_column))
        
        mean_value_vector.append(closest_to_the_mean)
    
    temp = grouped_data[0]
    for row in grouped_data[1:]:
        num_of_values = sum(x is not None for x in row)
        if num_of_values > (len(attribute_list)/2.0 +1):
            temp = np.vstack((temp, row))
    
    grouped_data = temp
    attribute_list = grouped_data[0, 1:]
    attribute_titles = grouped_data[0]
    complete_by_mean = grouped_data[1:,0]    
    i = 1
    for attribute in attribute_list: 
        attribute_index = list(grouped_data[0]).index(attribute)
        lst = list(grouped_data[1: , attribute_index])
        complete_column = [mean_value_vector[i] if item is None else item for item in lst]
        complete_by_mean = np.column_stack((complete_by_mean, complete_column))
        i += 1
        
    complete_by_mean = np.vstack((attribute_titles, complete_by_mean))
    
    return complete_by_mean
    




## ____________________________ Completing by use of KNN ________________________

def completing_by_KNN(normalized_data, reduced_data, nominal_attributes):
#    grouped_data = np.asarray(grouped_data)
    complete_data = reduced_data[0]
    
    for row in reduced_data[1:]:
        row = list(row)
        if None in row:
            instance_ID = row[0]
            neighbor_IDs_and_dists = ID_and_dist_of_neighbors(normalized_data, nominal_attributes, instance_ID)
            neighbor_IDs = neighbor_IDs_and_dists[:,0]
            
            for i in range(len(row)):
                if row[i] == None: 
                    
                    None_index_in_reduced = i
                    None_index_in_normalized = list(normalized_data[0]).index(reduced_data[0, None_index_in_reduced])
                    
                    if (reduced_data[0,None_index_in_reduced] in nominal_attributes):
                        row[None_index_in_reduced] = max_vote_for_nominal_attribute(normalized_data, neighbor_IDs, None_index_in_normalized)
                    else:
                        row[None_index_in_reduced] = weighted_mean_by_neighbors(normalized_data, neighbor_IDs_and_dists, None_index_in_normalized)
        
        complete_data = np.vstack((complete_data, row))
        
    return complete_data


def completing_by_KNN_error_checking_version(normalized_data, grouped_data, nominal_attributes, rand_row_index, rand_column_index):
    grouped_data = np.asarray(grouped_data)
    complete_data = grouped_data[0]
    
    for row in normalized_data[1:]:
        row = list(row)
        
        instance_ID = grouped_data[rand_row_index, 0]
        neighbor_IDs_and_dists = ID_and_dist_of_neighbors(normalized_data, nominal_attributes, instance_ID)
        neighbor_IDs = neighbor_IDs_and_dists[:,0]
            
        
        None_index_in_reduced = rand_column_index
        None_index_in_normalized = list(normalized_data[0]).index(grouped_data[0, None_index_in_reduced])
        
        if (grouped_data[0,None_index_in_reduced] in nominal_attributes):
            row[None_index_in_reduced] = max_vote_for_nominal_attribute(normalized_data, neighbor_IDs, None_index_in_normalized)
        else:
            row[None_index_in_reduced] = weighted_mean_by_neighbors(normalized_data, neighbor_IDs_and_dists, None_index_in_normalized)
        
        complete_data = np.vstack((complete_data, row))
        
    return complete_data




def ID_and_dist_of_neighbors(normalized_data, nominal_attributes, instance_ID):
    temp = [0, 0]
    instance_ID_index = list(normalized_data[:,0]).index(instance_ID)
    instance = normalized_data[instance_ID_index]
    
    for row in normalized_data[1:]:       
        if row[0] != instance_ID:
            distance = KNN_Manhattan_distance(normalized_data, nominal_attributes, row, instance)
            temp_row = [int(row[0]), distance]
            temp = np.vstack((temp, temp_row))

    temp = temp[1:]
    max_dist = max(temp[:,1])
    min_dist = min(temp[:,1])
    
    delta_dist = max_dist - min_dist
    for i in range(len(temp[:,0])):
        temp[i,1] = (temp[i,1]-min_dist)/delta_dist
    
    neighbor_IDs_and_distances = np.asarray(sorted(temp, key=lambda x: x[1]))

    return neighbor_IDs_and_distances
        


def KNN_Manhattan_distance(normalized_data, nominal_attributes, row, instance):
    dist = 0
    for i in range(1,len(row)):
        if (normalized_data[0,i] in nominal_attributes):
            if row[i] == None  or  instance[i] == None  or  row[i] != instance[i]:
                dist = dist + 1
        else:
            if row[i] == None  or  instance[i] == None:
                dist = dist+ 1
            else:
                dist = dist + abs(row[i] - instance[i])
                
    return dist
    



def weighted_mean_by_neighbors(normalized_data, neighbor_IDs_and_dists, None_attribute_index):
    
    weighted_neighbors_value_sum = 0
    sum_of_weights = 0
    List = []
    n_neighbors = 4
    counter = 0
    
    while counter < 3: #sum_of_weights == 0:
        n_neighbors = n_neighbors+1
        k_nearest_neighbors = neighbor_IDs_and_dists[0:n_neighbors]  ## starts with 5 neighbors
        
        for row in k_nearest_neighbors:
            ID_index = list(normalized_data[:, 0]).index((row[0]))
            if normalized_data[ID_index, None_attribute_index] != None:
                counter = counter +1
                weighted_neighbors_value_sum = weighted_neighbors_value_sum + normalized_data[ID_index, None_attribute_index] * (1 - row[1])
                sum_of_weights = sum_of_weights + (1 - row[1])
                List.append(normalized_data[ID_index, None_attribute_index])
    
    weighted_mean_value = weighted_neighbors_value_sum / sum_of_weights
    closest_to_the_mean = min(List, key=lambda x:abs(x-weighted_mean_value))
        
    return closest_to_the_mean




def max_vote_for_nominal_attribute (normalized_data, neighbor_IDs, None_attribute_index):
    neighbor_column = []
    for ID in neighbor_IDs:
        ID_index = list(normalized_data[:, 0]).index(ID)
        if normalized_data[ID_index, None_attribute_index] != None:
            neighbor_column.append(normalized_data[ID_index, None_attribute_index])
        if len(neighbor_column) >= 5:
            break
      
    number_of_0s = sum(x is 0 for x in neighbor_column)
    number_of_1s = sum(x is 1 for x in neighbor_column)
    number_of_2s = sum(x is 2 for x in neighbor_column)
    number_of_3s = sum(x is 3 for x in neighbor_column)
    number_of_4s = sum(x is 4 for x in neighbor_column)
    
    if number_of_0s > number_of_1s and number_of_0s > number_of_2s and number_of_0s > number_of_3s and number_of_0s > number_of_4s:
        max_vote = 0
    elif number_of_1s >= number_of_0s and number_of_1s > number_of_2s and number_of_1s > number_of_3s and number_of_1s > number_of_4s:
        max_vote = 1
    elif number_of_2s >= number_of_0s and number_of_2s >= number_of_1s and number_of_2s > number_of_3s and number_of_2s > number_of_4s:
        max_vote = 2
    elif number_of_3s >= number_of_0s and number_of_3s >= number_of_1s and number_of_3s >= number_of_2s and number_of_3s > number_of_4s:
        max_vote = 3
    elif number_of_4s >= number_of_0s and number_of_4s >= number_of_1s and number_of_4s >= number_of_2s and number_of_4s >= number_of_3s:
        max_vote = 3
    
    return max_vote


