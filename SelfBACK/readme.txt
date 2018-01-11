Analysis Instruction:

1- Put color_coded excel file of datasets in a folder. inside each file you should determine intended attributes for different analyses by three different colors as backgrounds; a color per ordinal/categorical/outcomes attributes. Leave the other attributes' background white. 

2- Set the parameters of config files in the "config_files" folder. There is enough information about the variables inside the files.

3- Run the program "RUN_making_dataset_file.py". During runtime you should determine the target datasest according to what the program asks. The resulting dataset can be found in the distination which is set in the config file.

4- If you want to do the correlation analysis, run the code "RUN_correlation_analysis.py". The result is placed in the "results" folder as an .xlsx file and could be used in determining the attribute set for clustering. 

5- Before running the clustring code, you should set "attributes_for_clustering" (using attributes determined by correlation analysis, recommendations, or other techniques) in the related config file and save it.

6- to do the clustering on target dataset, first make sure that you have "KModes" package installed in your pyton. Check "https://pypi.python.org/pypi/kmodes/" for more information. Afte installing KModes, run the code "RUN_SelfBACK_clustering.py". Check the comments in the console from the code and answer the questions.

7- The results of cluster analysis can be found in the 'results' folder, in form of plots and excel files. 
