# -*- coding: utf-8 -*-


##  dataset making setting
numpy_dataset_destination_address_and_filename = '../datasets/fysioprim012017_dataset_file'
excel_dataset_address = '../datasets/fysioprim_01_2017_utenlabels.xlsx'
excel_sheet_name = 'Sheet1'

ID_attribute = 'id'
if_add_BMI = False
height_attribute = 'hoyde'
weight_attribute = 'vekt'

first_colored_ordinal_attribute = 'fdselsr'
first_colored_categorical_attribute = 'kjonn'
first_colored_outcome_attribute = 'psfs1_3'
manipulated_outcomes = False


##  correlation analysis setting
if_filter = True
if_gorup = True
if_normalize = True
filtering_attribute = 'hovedproblem'
filtering_value = 3


##  clustering settings
##  information from correlation analysis can be used to find 'attributes_for_clustering' below
attributes_for_clustering = ['arbeidsevne', 'keelestartback_risk_1',  'smertevarighet', 'orebro_total_1', 'aktivitetsnivaa', 'pseq_1', 'antallsmerteregioner_1']
imputation_method = 'KNN'           # use  'KNN' to use KNN method for imputation,  or 'MEANS' to use mean values for imputation, or 'REMOVE' to only remove incomplete instances
clustering_method = 'KMeans'        # use  'Kmeans' for  just ordinals,   'KModes' for just categoricals,   and 'KPrototypes' for mixed attributes
if_representatives = True           # True: introduce representative IDs in an Excel file,    False: not to do that
if_boxplots = True                 # True: draw boxplots and save them in the results folder,    False: not to do that
n_test_clusters = 15                  # The range of number of clusters to be tested for finding a good valuse of K.
n_run_to_find_representatives = 6   #the number of runs to find representative ID of each cluster.
