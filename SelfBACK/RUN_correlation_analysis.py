# -*- coding: utf-8 -*-
import sys
sys.path.append('D:\SelfBACK\OO-SelfBACK\fysioprim-data-analysis\program_body')

import numpy as np
from correlation_assessment_class import *
from attributes import *

dataset_input = input('enter the dataset which you want ot analyze it (f1: fysioprim 01/2017,  f8: fysioprim 08/2017,  d:Dphacto): ')
if dataset_input == 'f1':
    file = open ('./config_files/fysioprim_012017_configfile.py')
elif dataset_input == 'f8':
    file = open ('./config_files/fysioprim_082017_configfile.py')
elif dataset_input == 'd':
    file = open ('./config_files/dphacto_configfile.py')


file_lines = file.readlines()
for line in file_lines:
    exec(line)
        

data_set = np.load(numpy_dataset_destination_address_and_filename+'.npy')
attributes = ordinal_attribute_set + categorical_attribute_set + outcomes_measures

correlation_analyzer = correlation_analysis(data_set=np.asarray(data_set), attributes=attributes, outcome_measures=outcomes_measures,  ordinals=ordinal_attribute_set, categoricals=categorical_attribute_set)

if if_filter: correlation_analyzer.filter_data(filtering_attribute=filtering_attribute, filtering_value=filtering_value)
if if_gorup: correlation_analyzer.group_data()
if if_normalize: correlation_analyzer.normalize_data()

correlation_analyzer.run()
