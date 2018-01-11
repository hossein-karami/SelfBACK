import numpy as np
import copy as cp
from kmodes import kmodes
from kmodes.kprototypes import KPrototypes
from scipy import spatial
from math import log, exp, cos
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

from attributes import *
from imputation_methods import *


data_set = np.asarray([[0.1,0.3,0.3,0.31], [0.11,0.19,0.31,0.6], [0.09,0.2,0.31,0.3], [0.4,0.46,0.5,0.01], [0.41,0.5,0.51,0.2]])
attributes= ['a', 'b', 'c', 'd']
numerics= ['a', 'b', 'c', 'd']
nominals = [ ]
n_base_attr=1
n_clusters=2
def entropy_calculation (data_set, attribute_list, numerics, nominals):
    n_numerics = len(numerics) 
    n_nominals = len(nominals)
    
    inst_index = 0
    numeric_distances = []
    nominal_similarities =[]
    n_similarities = 0
    
    for inst1 in data_set:
                
        for i in range(1, len(data_set[inst_index:, 0])):
            
            inst2 = data_set[inst_index+i]
            numeric_values_1 = [] if n_numerics != 0 else [None]
            numeric_values_2 = [] if n_numerics != 0 else [None]
            nominal_values_1 = []
            nominal_values_2 = []
            
            for attr in attribute_list:
                if attr in numerics:
                    numeric_index = list(attribute_list).index(attr)
                    numeric_values_1.append(inst1[numeric_index])
                    numeric_values_2.append(inst2[numeric_index])
                elif attr in nominals:
                    nominal_index = list(attribute_list).index(attr)
                    nominal_values_1.append(inst1[nominal_index])
                    nominal_values_2.append(inst2[nominal_index])
                    
            euclidean_dist = spatial.distance.euclidean(numeric_values_1, numeric_values_2) if numeric_values_1 != [None] else 0
            numeric_distances.append(euclidean_dist)
            
            nominal_similarity = float(sum([1 for j in range(len(nominal_values_1)) if nominal_values_1[j] == nominal_values_2[j]]) / n_nominals) if n_nominals else 0
            
            nominal_similarities.append(nominal_similarity)
            n_similarities +=1 
            
        inst_index += 1
        
    max_numeric_distance = max(numeric_distances) if max(numeric_distances) != 0 else 1
    temp_list_1 = [dist*np.pi/max_numeric_distance for dist in numeric_distances]
    
    temp_list_2 = list(map(cos, temp_list_1))
    
    numeric_similarities = [(0.5 + 0.5*element) for element in temp_list_2]
    
    if None in nominal_similarities: nominal_similarities = np.zeros(len(nominal_similarities))
    if None in numeric_similarities: numeric_similarities = np.zeros(len(numeric_similarities))
    
    similarities = [(n_nominals * nominal_similarities[i] + n_numerics* numeric_similarities[i]) / (n_nominals + n_numerics) for i in range(n_similarities)]
    
    entropy = 0
    for element in similarities:
        if element!=0 and element !=1:  entropy = entropy - ((element * log(element)) + ((1-element) * log(1-element)))
    
    return entropy




def rank_features(data_set, attributes, numerics, nominals):
    entropies = []
    entropy_indices = np.arange(len(attributes))
    for attr in attributes:
        numerics_for_entropy = numerics
        nominals_for_entropy = nominals
        print(attr)
        temp_data = cp.deepcopy(data_set)
        attr_index = list(attributes).index(attr)
        dataset_for_ranking = np.delete(temp_data, attr_index, 1)
        if attr in numerics: 
            index = list(numerics).index(attr)
            numerics_for_entropy = np.delete(numerics, index)
        elif attr in nominals:
            index = list(nominals).index(attr)
            nominals_for_entropy = np.delete(nominals, index)
            
        attributes_for_ranking = np.delete(attributes, attr_index)
        attr_entropy = entropy_calculation (dataset_for_ranking, attributes_for_ranking, numerics_for_entropy, nominals_for_entropy)
        
        entropies.append(attr_entropy)
    
    
    indexed_entropies = np.column_stack((entropy_indices, entropies, attributes))
    entropies_sorted = np.asarray(sorted(indexed_entropies, key=lambda x: x[1], reverse=True))
    ranked_indices = entropies_sorted[:,0]
    ranked_attributes = entropies_sorted[:,2]
    for item in entropies_sorted[:,1]:
        print('entropy', item)
    return ranked_attributes



ranked_attributes = rank_features(data_set, attributes, numerics, nominals)
for item in ranked_attributes: print(item)



def do_clustering(dataset, attributes, ranked_attributes, numerics, nominals, n_base_attr):
    elvaluation_metrics = []
    singulars=[]
    attribute_set = ranked_attributes[:n_base_attr]
    data_for_clustering = []
    nominals_indices =[]
    for attribute in attribute_set:
        if attribute in nominals:
            nominal_index = list(attribute_set).index(attribute)
            nominals_indices.append(nominal_index)
    
    for attribute in attribute_set:
        attr_index = list(attributes).index(attribute)
        data_for_clustering = dataset[:, attr_index] if data_for_clustering ==[] else np.column_stack((data_for_clustering, dataset[:, attr_index]))
    
    for attribute in ranked_attributes[n_base_attr:]:
        attribute_set = np.append(attribute_set , attribute)
        attr_index = list(attributes).index(attribute)
        data_for_clustering = np.column_stack((data_for_clustering, dataset[:, attr_index]))
        
        if attribute in nominals:
            nominal_index = list(attribute_set).index(attribute)
            nominals_indices.append(nominal_index)
        
        
        if any(attr in nominals for attr in attribute_set) and not all(attr in nominals for attr in attribute_set):
            kpro = KPrototypes(n_clusters=n_clusters, init='Cao', n_init=50, verbose=1)
            clusters = kpro.fit_predict(data_for_clustering, categorical=nominals_indices)
            cluster_column = list(map(int, kpro.labels_+1))
            clustered_dataset = np.column_stack((dataset, cluster_column))
            clustered_dataset_sorted = np.asarray(sorted(clustered_dataset, key=lambda x: x[-1]))
            clustered_reduced = np.column_stack((data_for_clustering, cluster_column))
            clustered_reduced_sorted = np.asarray(sorted(clustered_reduced, key=lambda x: x[-1]))
            #Print the cluster centroids
            #print(km.cluster_centroids_)
        elif all(attr in nominals for attr in attribute_set):
            km = kmodes.KModes(n_clusters=n_clusters, init='Huang', n_init=50, verbose=1)
            clusters = km.fit_predict(data_for_clustering, categorical=nominals_indices)
            cluster_column = list(map(int, km.labels_+1))
            clustered_dataset = np.column_stack((dataset, cluster_column))
            clustered_dataset_sorted = np.asarray(sorted(clustered_dataset, key=lambda x: x[-1]))
            clustered_reduced = np.column_stack((data_for_clustering, cluster_column))
            clustered_reduced_sorted = np.asarray(sorted(clustered_reduced, key=lambda x: x[-1]))
        else:
            kmeans = KMeans(n_clusters = n_clusters, init='k-means++', n_init=50, max_iter = 1000)
            clusters = kmeans.fit(data_for_clustering)
            cluster_column = list(map(int, kmeans.labels_+1))
            clustered_dataset = np.column_stack((dataset, cluster_column))
            clustered_dataset_sorted = np.asarray(sorted(clustered_dataset, key=lambda x: x[-1]))
            clustered_reduced = np.column_stack((data_for_clustering, cluster_column))
            clustered_reduced_sorted = np.asarray(sorted(clustered_reduced, key=lambda x: x[-1]))
            
        silhouette_avg = silhouette_score(data_for_clustering, cluster_column)
        print('\nHaving', n_clusters, 'clusters, the average silhouette score is: ', silhouette_avg)
        
        evaluation_metric, singular_counter = scatter_criteria(clustered_reduced_sorted, attribute_set, numerics, nominals)
        elvaluation_metrics.append(evaluation_metric)
        singulars.append(singular_counter)
    print('\n\n\n', elvaluation_metrics)
    
    return clustered_dataset_sorted
    
        

     


def mean_values (dataset, attributes, numerics, nominals):
    attributes_mean = []
    for attr in attributes:
        attr_index = list(attributes).index(attr)
        if attr in numerics:
            attr_mean = np.mean(dataset[:,attr_index])
            attributes_mean.append(attr_mean)
        elif attr in nominals:
            attr_most_repeated = max(set(dataset[:,attr_index]), key =list(dataset[:,attr_index]).count)
            attributes_mean.append(attr_most_repeated)
    
    return attributes_mean


def mixed_difference(instance1, instance2, attributes, nominals):
    difference = []
    for attr in attributes:
        if attr in nominals:
            attr_index = list(attributes).index(attr)
            diff = 0 if instance1[attr_index]==instance2[attr_index] else 1
            difference.append(diff)
        else:
            attr_index = list(attributes).index(attr)
            diff = instance1[attr_index] - instance2[attr_index]
            difference.append(diff)
    return difference


def scatter_criteria(clustered_dataset, attributes, numerics, nominals):
    between_scatter =[]
    within_scatter=[]
    W_matrix_maker =[]
    divided_clusters =np.split(clustered_dataset, np.where(np.diff(clustered_dataset[:,-1]))[0]+1)
    n_clusters = len(divided_clusters)
    n_instances = len(clustered_dataset[:,0])
    avg_n_members = n_instances / n_clusters
    overall_means = mean_values(clustered_dataset, attributes, numerics, nominals)
    counter1 = 0
    singular_counter = 0
    
    for cluster in divided_clusters:
        n_cluster_member = len(cluster[:,0])
        cluster_means = mean_values(cluster, attributes, numerics, nominals)
        means_diff = list(map(abs, mixed_difference(cluster_means, overall_means, attributes, nominals))) #np.subtract(cluster_means, overall_means)
        B_matrix_maker = np.outer(means_diff, means_diff)
        B_matrix_maker = np.multiply(B_matrix_maker, 1/n_cluster_member)
        between_scatter = np.add(between_scatter, B_matrix_maker) if counter1 != 0 else B_matrix_maker
        counter2 = 0
        for instance in cluster:
            instance_diff = list(map(abs, mixed_difference(instance[:-1], cluster_means, attributes, nominals)))  #np.subtract(instance[:-1], cluster_means)  #
            W_matrix_maker_inner = np.outer(instance_diff, instance_diff)
            W_matrix_maker = np.add(W_matrix_maker, W_matrix_maker_inner) if counter2 != 0 else W_matrix_maker_inner
            counter2 += 1
            
        W_matrix_maker = np.multiply((1/n_cluster_member), W_matrix_maker)
        
        within_scatter = np.add(within_scatter, W_matrix_maker) if counter1 != 0 else W_matrix_maker
        
        counter1 += 1
    
    between_scatter = np.multiply(between_scatter, avg_n_members)
    within_scatter = np.multiply(within_scatter, avg_n_members)
    
    trace_W = np.matrix.trace(within_scatter)
    trace_B = np.matrix.trace(between_scatter)
    print('TRACES',trace_W, trace_B)
    
#    if np.where(~within_scatter.any(axis=1))[0]: singular_counter =1 
#    zero_row = np.where(~within_scatter.any(axis=1))[0]
#    while zero_row:
#        within_scatter[zero_row[0], zero_row[0]] = 1
#        zero_row = np.where(~within_scatter.any(axis=1))[0]
#    if np.where(~within_scatter.any(axis=1))[0]: print(np.where(~within_scatter.any(axis=1))[0])
    
    Winv_B = np.matmul( np.linalg.inv(within_scatter) , between_scatter)
    evaluation_metric = np.matrix.trace( Winv_B)
    
    print('evaluation metric:', evaluation_metric)
   
    return evaluation_metric, singular_counter

#ranked_attributes = ['arbeidsevne_1', 'keelestartback_risk_1', 'utdanning', 'BMI',
#                    'mosjonvarighet_1', 'tropaabehandling', 'smerteintensitet_1',
#                    'psfs1_1', 'orebro_total_1', 'odi_total_1', 'aktivitetsnivaa_1', 
#                    'hscl_mean_1', 'pseq_1', 'mosjonhyppighet_1', 'smertevarighet', 
#                    'sovn_1', 'tampa_1item', 'mosjonintensitet_1', 'fdselsr']


ranked_attributes = ['hscl_mean_1', 'mosjonvarighet_1', 'keelestartback_risk_1', 'mosjonintensitet_1', 
                     'aktivitetsnivaa_1', 'tropaabehandling', 'sovn_1', 'mosjonhyppighet_1', 'utdanning', 
                     'smerteintensitet_1', 'psfs1_1', 'pseq_1', 'tampa_1item', 'smertevarighet', 'arbeidsevne_1',
                     'odi_total_1', 'BMI', 'orebro_total_1', 'fdselsr' ]

aa = do_clustering(data_set, attributes, ranked_attributes, numerics, nominals, n_base_attr)  

print(aa)

'arbeidsevne_1', 'tampa_1item', 'keelestartback_risk_1', 'sovn_1', 'fdselsr', 'pseq_1', 'hscl_mean_1', 'orebro_total_1',
 'tropaabehandling', 'BMI', 'mosjonhyppighet_1', 'mosjonvarighet_1', 'aktivitetsnivaa_1', 'mosjonintensitet_1',
 'utdanning', 'smerteintensitet_1', 'rykerdu', 'psfs1_1', 'bluewhite_collar_1', 'kjonn', 'analgesica_janei_1', 'smertevarighet', 'odi_total_1'