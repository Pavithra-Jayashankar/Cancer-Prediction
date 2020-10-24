import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import os
os.chdir("D:\Imarticus\Python_program")
os.getcwd()

Cancer_data = pd.read_csv("cancerdata.csv")
Cancer_data.isnull().sum()

import numpy as np

Cancer_data["diagnosis"] = np.where(Cancer_data["diagnosis"] == "M" , 0, 1)
Cancer_data.drop(['id'],axis = 1, inplace = True)

#### Sampling

from sklearn.model_selection import train_test_split
Trainset , Testset = train_test_split(Cancer_data,train_size = 0.8, random_state = 123)

Train_x = Trainset.drop(['diagnosis'], axis = 1).copy()
Train_y = Trainset['diagnosis'].copy()
Test_x = Testset.drop(['diagnosis'], axis = 1).copy()
Test_y = Testset['diagnosis'].copy()


#### Standardization

from sklearn.preprocessing import StandardScaler

Train_Scaling = StandardScaler().fit(Train_x)
#Train_Scaling.mean_
Train_x_STD = Train_Scaling.transform(Train_x)
Test_x_STD = Train_Scaling.transform(Test_x)

Train_x_STD = pd.DataFrame(Train_x_STD,columns = Train_x.columns)
Test_x_STD = pd.DataFrame(Test_x_STD,columns = Test_x.columns)

#### Model Building

from sklearn.neighbors import KNeighborsClassifier
M1 = KNeighborsClassifier(n_neighbors= 5).fit(Train_x_STD,Train_y)


#### Model Prediction

Test_pred = M1.predict(Test_x_STD)


### Probability prediction

Test_prob = M1.predict_proba(Test_x_STD)
Test_prob_Df = pd.DataFrame(Test_prob)
Test_prob_Df['Class'] = Test_pred

#### Model evaluation 

from sklearn.metrics import confusion_matrix, precision_score,recall_score,f1_score
Confusion_Mat = confusion_matrix(Test_y,Test_pred)

sum(np.diagonal(Confusion_Mat))/Test_x.shape[0]*100   #0.93
precision_score(Test_y,Test_pred)   #0.91
f1_score(Test_y,Test_pred)   #0.95
recall_score(Test_y,Test_pred)  #1
Confusion_Mat[0][1]/sum(Confusion_Mat[0])  #0.17

#### Grid Search 

from sklearn.model_selection import GridSearchCV

param = dict(n_neighbors = [1,3,5,7,9] , p=[1,2])

grid = GridSearchCV(M1, param, cv = 5, scoring='accuracy')
grid.fit(Train_x_STD,Train_y)
Grid_Search_Df = pd.DataFrame.from_dict(grid.cv_results_)

grid.best_estimator_

Final_model = KNeighborsClassifier(n_neighbors=5,p=1).fit(Train_x_STD,Train_y)
Final_pred = Final_model.predict(Test_x_STD)

confusion_matrix1 = confusion_matrix(Test_y,Final_pred)
sum(np.diagonal(confusion_matrix1))/Test_x.shape[0]*100 #0.95
precision_score(Test_y,Final_pred)  #0.93
f1_score(Test_y,Final_pred)  #0.966
recall_score(Test_y,Final_pred) #1
