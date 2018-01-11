# -*- coding: utf-8 -*-
import copy as cp
import numpy as np
import scipy as sp

from openpyxl import load_workbook

from imputation_methods import *
from data_preparation import normalizing_data

    
xlFile_addr = '.\datasets\SupportPrim_pilotdata_utenLabels.xlsx'    
wb = load_workbook(xlFile_addr)
sheet = wb.get_sheet_by_name('Sheet1')

attribute_filename = 'attributes.py'



def make_dataset_files(sheet, numerical_attribute_set, categorical_attribute_set):
    
    data = list(sheet.values)
    data_array = np.asarray(data)
    
    ID_index = list(data_array[0]).index('id')
    if ID_index != 0:
        ID_column = data_array[:, ID_index]
        temp_array_2 = np.delete(data_array, ID_index, 1)
        data_array = np.column_stack((ID_column, temp_array_2))
    
    
    
    temp_array = cp.deepcopy(data_array)
    attribute_set = numerical_attribute_set + categorical_attribute_set    
    for attribute in data_array[0]:
        if attribute == None or attribute not in attribute_set and attribute != 'id':
            none_index = list(temp_array[0]).index(attribute)
            temp_array = sp.delete(temp_array, none_index, 1)

    normalized_data = normalizing_data(temp_array, categorical_attribute_set)
## imputing missing data in the data set
    completed_data = completing_by_KNN(normalized_data, normalized_data, categorical_attribute_set)

##  ----- Removing repeated IDs ------    
    row_index = 0
    filter_attr_index = list(completed_data[0]).index('id')
    ID_index = list(completed_data[0]).index('id')
    ID = 0
    
    for row in completed_data:
        if row[filter_attr_index]== None:
            completed_data = sp.delete(completed_data, row_index,0)
            row_index -=1
        
        row_index += 1
    
    row_index =0
    for row in completed_data:
        if row[ID_index] !=ID:
            ID = row[ID_index]
        else:
            completed_data = sp.delete(completed_data, row_index,0)
            row_index -=1
        
        row_index += 1
   
    attributes_file = open(attribute_filename, 'w')
    attributes_file.write('# -*- coding: utf-8 -*- \n\n\n')
    attributes_file.write('numerical_attribute_set = [')
    for attribute in numerical_attribute_set:
        attributes_file.write("'" + str(attribute) + "'" + ', ')
    attributes_file.write(']\n\n\n')
    
    attributes_file.write('categorical_attribute_set = [')
    for attribute in categorical_attribute_set:
        attributes_file.write("'" + str(attribute) + "'" + ', ')
    attributes_file.write(']\n\n\n')
      
    attributes_file.close()
    np.save('./dataset_file', completed_data)
    
    
    
def make_attribute_sets(sheet, attribute_filename)  :
    
    numerical_attribute_set =[]
    categorical_attribute_set = []
    numerical_attributes_color = ''
    categorical_attributes_color = ''
    
    for item in sheet[1]:
        
        if item.value == 'fdselsr':
            numerical_attributes_color = item.fill.start_color.index
        elif item.value == 'kjonn':
            categorical_attributes_color = item.fill.start_color.index
        
        
        if item.fill.start_color.index == numerical_attributes_color:
            numerical_attribute_set.append(item.value) 
        elif item.fill.start_color.index == categorical_attributes_color:
            categorical_attribute_set.append(item.value)
#        
    return numerical_attribute_set, categorical_attribute_set






numerical_attribute_set, categorical_attribute_set = make_attribute_sets(sheet, attribute_filename)
make_dataset_files(sheet, numerical_attribute_set, categorical_attribute_set)


