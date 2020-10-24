import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Imarticus\Python_program")
os.getcwd()

Cancer_data = pd.read_csv("cancerdata.csv")
Cancer_data.isnull().sum()
Cancer_data.drop(['id'],axis = 1, inplace=True)

import numpy as np
Cancer_data['diagnosis'] = np.where(Cancer_data['diagnosis'] == 'M', 1, 0)

from sklearn.preprocessing import StandardScaler

Cancer_Data1 = StandardScaler().fit_transform(Cancer_data)
Cancer_Standradized = pd.DataFrame(Cancer_Data1,columns=Cancer_data.columns)

#### Number of Cluster

from sklearn.cluster import KMeans

WSS = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k, random_state=123).fit(Cancer_Standradized)
    WSS.append(kmeans.inertia_)  # inertia has the overall WSS
    
## Plot 
    
import seaborn as sns
sns.lineplot(x=range(1,15), y= WSS)

## Modeling

Kmeans_model = KMeans(n_clusters=3,random_state=123).fit(Cancer_Standradized)

## Clustering output Binding

Kmeans_model.labels_
Cancer_Standradized2 = pd.concat([Cancer_Standradized,pd.Series(Kmeans_model.labels_)],\
                                  axis =1).rename(columns={0:"Cluster"}).copy()

Cancer_Data1 = pd.concat([Cancer_data,pd.Series(Kmeans_model.labels_)],axis =1).rename(columns={0:"cluster"})

## Cluster Size

Cancer_Data1['cluster'].value_counts()

## Cluster profiling
Cluster_Profiling_Df =Cancer_Data1.groupby(['cluster']).mean()

#Bi-Variate plots

Cancer_Data1['cluster'] = np.select([Cancer_Data1['cluster'] ==0 , Cancer_Data1['cluster'] ==1, \
                                    Cancer_Data1['cluster'] == 2],['A','B','C'])
    
Cancer_Data1_copy = Cancer_Data1.copy()
Cancer_Data1_copy.sort_values(['cluster'], inplace = True)

sns.scatterplot(x = 'diagnosis', y = 'perimeter_mean', hue = 'cluster', data = Cancer_Data1_copy)
sns.scatterplot(x = 'area_worst', y = 'concavity_worst', hue = 'cluster', data = Cancer_Data1_copy)
sns.scatterplot(x = 'concavity_mean', y = 'perimeter_mean', hue = 'cluster', data = Cancer_Data1_copy)

#3 variable plot
sns.scatterplot(x = 'area_worst', y = 'concavity_worst', hue = 'cluster',size = 'diagnosis', data = Cancer_Data1_copy)

## Cluster Validation using Silhouette value

from sklearn.metrics import silhouette_samples,silhouette_score 

Sample_Silhouette_Values = silhouette_samples(Cancer_Standradized,Kmeans_model.labels_)
Cancer_Standradized2['Silhouette_Value'] = Sample_Silhouette_Values
Cancer_Standradized2.groupby(['Cluster'])['Silhouette_Value'].mean()

silhouette_score(Cancer_Standradized,Kmeans_model.labels_)

### Overall Vizualization of K-means(PCA application)

from sklearn.decomposition import PCA
Kmeans_Standardized_Data = Cancer_Standradized2.copy()

Kmeans_Standardized_Data2 = Kmeans_Standardized_Data.drop(['Cluster'],axis=1).copy()

#PCA model creation
PCA_model = PCA(n_components=2).fit(Kmeans_Standardized_Data2)

#PCA model variation
PCA_model.explained_variance_ratio_

#transform standardized data using PCA mpdel to transformed variables

Kmeans_Transformed_Data=pd.DataFrame(PCA_model.transform(Kmeans_Standardized_Data2))
Kmeans_Transformed_Data.columns = ['PC1','PC2']
Kmeans_Transformed_Data_With_Clusters = pd.concat([Kmeans_Transformed_Data,\
                                                   Kmeans_Standardized_Data['Cluster']],axis =1)
    
sns.scatterplot(x= 'PC1', y='PC2', hue = 'Cluster', data = Kmeans_Transformed_Data_With_Clusters, palette =['red','green','blue'])
