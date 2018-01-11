
import numpy as np
import copy as cp



def del_incomplete_data (incomplete_data):
    
    complete_data = np.asarray([row for row in incomplete_data if True not in np.equal(row, None)])
    
    return complete_data



def group_data_by_attributes(data_set, grouping_attributes):
    attributes = data_set[0]
    useless_attributes_indices = [list(attributes).index(attr) for attr in attributes[1:] if attr not in grouping_attributes]
    grouped_data = np.delete(data_set, useless_attributes_indices, axis=1)
    return grouped_data   





def normalize_data(data_set, categoricals=[]):
    normalized_data = data_set[:,0]
    attributes = data_set[0]
    temp1 = list(data_set)[:]
       
    for index in range(1,len(attributes)):
        column = np.asarray(temp1)[:,index]
        
        if attributes[index] not in categoricals:
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




def filter_for_attribute(data_set, attributes=[], filtering_attribute='', filtering_value=0):
    
    filtered_data = cp.deepcopy(data_set)
    if not attributes:
        attributes = data_set[0]
    
    if filtering_attribute in attributes:
        filtered_data = data_set[0]
        filtering_attr_index = list(attributes).index(filtering_attribute)
        for row in data_set:
            if row[filtering_attr_index] == filtering_value:
                filtered_data = np.vstack([filtered_data,row])
                
        filtered_data = sp.delete(filtered_data, filtering_attr_index, 1)
     
    return filtered_data




def mean_values (data_set, attributes, ordinals, categoricals):
    attributes_mean = []
    for attr in attributes:
        attr_index = list(attributes).index(attr)
        if attr in ordinals:
            attr_mean = np.mean(data_set[:,attr_index])
            attributes_mean.append(attr_mean)
        elif attr in categoricals:
            attr_most_repeated = max(set(data_set[:,attr_index]), key =list(data_set[:,attr_index]).count)
            attributes_mean.append(attr_most_repeated)
    
    return attributes_mean


def mixed_difference(instance1, instance2, attributes, categoricals):
    difference = []
    for attr in attributes:
        if attr in categoricals:
            attr_index = list(attributes).index(attr)
            diff = 0 if instance1[attr_index]==instance2[attr_index] else 1
            difference.append(diff)
        else:
            attr_index = list(attributes).index(attr)
            diff = instance1[attr_index] - instance2[attr_index]
            difference.append(diff)
    return difference