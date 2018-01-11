# -*- coding: utf-8 -*-


##  dataset making setting
numpy_dataset_destination_address_and_filename = '../datasets/dphacto_dataset_file'
excel_dataset_address = '../datasets/selfback-dphacto.xlsx'
excel_sheet_name = 'dphacto'

ID_attribute = 'ID'
if_add_BMI = False
height_attribute = 'hoyde'
weight_attribute = 'vekt'

first_colored_ordinal_attribute = 'OA1_Age'
first_colored_categorical_attribute = 'OA1_Gender'
first_colored_outcome_attribute = 'OA_sms_resp_Week_00_Q2'
manipulated_outcomes = False


##  correlation analysis setting
if_filter = False
if_gorup = False
if_normalize = True
filtering_attribute = 'hovedproblem'
filtering_value = 3


##  clustering settings
##  information from correlation analysis can be used to find 'attributes_for_clustering' below
attributes_for_clustering = ['OA1_T15_7', 'OA1_T15_8', 'OA1_T16_5', 'OA2_B13_1', 'OA2_B13_2', 'OA2_B13_3', 'OA2_B13_4', 'OA1_B6_4', 'OA1_B7_4']

imputation_method = 'REMOVE'           # use  'KNN' to use KNN method for imputation,  or 'MEANS' to use mean values for imputation, or 'REMOVE' to only remove incomplete instances
clustering_method = 'KPrototypes'        # use  'Kmeans' for  just ordinals,   'KModes' for just categoricals,   and 'KPrototypes' for mixed attributes
if_representatives = True          # True: introduce representative IDs in an Excel file,    False: not to do that
if_boxplots = True                 # True: draw boxplots and save them in the results folder,    False: not to do that
n_test_clusters = 10                  # The range of number of clusters to be tested for finding a good valuse of K.
n_run_to_find_representatives = 6   #the number of runs to find representative ID of each cluster.


