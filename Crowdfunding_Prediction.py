#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:47:08 2021

@author: zorak
"""
#################### Developing the model #######################
import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
standardizer = StandardScaler()
# Import data
kickstarter_df = pandas.read_excel("/Users/zorak/Desktop/Fall_courses/INSY662/sessions/csv/Kickstarter.xlsx")
# Pre-Processing
#only include observations where the variable “state” takes the value “successful” or “failure” 
kickstarter = kickstarter_df[(kickstarter_df['state'] == 'successful') | (kickstarter_df['state'] == 'failed')]
#reassign values to variables before converting to dummies
kickstarter['disable_communication'] = np.where(kickstarter['disable_communication'] == True, 'No_communication', 'With_communication')

# For analysis - we need to change these variables as dummy variables
dummy_state=pandas.get_dummies(kickstarter.state)
dummy_disable_communication=pandas.get_dummies(kickstarter.disable_communication)
# genearte final data for analysis including dummy variables
kickstarter = kickstarter.join(dummy_state)
kickstarter = kickstarter.join(dummy_disable_communication)

#assign new column as goal*static_usd_rate
kickstarter['goal_usd'] = kickstarter['static_usd_rate']*kickstarter['goal']


#new dataframe with used variables
kickstarter_used = kickstarter[["successful", 'goal_usd',  'With_communication',
                              'name_len', 'blurb_len', 'launch_to_deadline_days', 'create_to_launch_days']]
kickstarter_used = kickstarter_used.dropna()


######classification model
y = kickstarter_used['successful']
X = kickstarter_used.drop(columns=['successful'])

from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import cross_val_score


### random forest ###
randomforest = RandomForestClassifier(random_state=5, oob_score=True)
model1 = randomforest.fit(X, y)

#to find the important features
coe = list()
for i in range(len(model1.feature_importances_)):
    a = model1.feature_importances_[i]
    coe.append(a)  
li = []
for i in range(len(coe)):
    if coe[i] > 0:
        li.append(i)

#to get the three most important predictors
import heapq
col = heapq.nlargest(3, range(len(coe)), key=coe.__getitem__)
X.columns[col]
rf_X = X[X.columns[col]]
#standardize
rf_X_std = standardizer.fit_transform(rf_X)

###run Gradient Boosting
X_train, X_test, y_train, y_test = train_test_split(rf_X , y, test_size = 0.3, random_state = 8)
from sklearn.ensemble import GradientBoostingClassifier
gbt = GradientBoostingClassifier()
model1b = gbt.fit(X_train, y_train)
y_test_pred = model1b.predict(X_test)
metrics.accuracy_score(y_test, y_test_pred)


### Run K-NN
#X_train, X_test, y_train, y_test = train_test_split(rf_X_std, y, test_size = 0.3, random_state = 5)
#knn = KNeighborsClassifier(n_neighbors=5) 
#model1b = knn.fit(X_train,y_train)
#y_test_pred = model1b.predict(X_test)
#metrics.accuracy_score(y_test, y_test_pred)
#from sklearn.metrics import mean_squared_error 
#mse = mean_squared_error(y_test, y_test_pred)
#print(mse)

###run random forest
#X_train, X_test, y_train, y_test = train_test_split(rf_X , y, test_size = 0.3, random_state = 5)
#model1b = randomforest.fit(X_train,y_train)
#y_test_pred = model1b.predict(X_test)
#metrics.accuracy_score(y_test, y_test_pred)

###run ANN
#X_train, X_test, y_train, y_test = train_test_split(rf_X_std, y, test_size = 0.3, random_state = 5)
#from sklearn.neural_network import MLPClassifier
#mlp = MLPClassifier(hidden_layer_sizes=(11),max_iter=1000,random_state=5)
#model1b = mlp.fit(X_train,y_train)
#y_test_pred = model1b.predict(X_test)




#####clustering model#############
### k-mean ###
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as pyplot
# Optimal K
#using the predictors selected by random forest and standardized
from sklearn.metrics import silhouette_score
df2 = X[X.columns[col]]
#standardize
df2_std = standardizer.fit_transform(df2)

#check the relationship between number of clusters and within-cluster variance
withinss = []
for i in range (2,8):
    kmeans = KMeans(n_clusters=i, random_state = 8)
    model = kmeans.fit(df2_std)
    withinss.append(model.inertia_)
pyplot.plot([2,3,4,5,6,7],withinss)

#check the best silhouette_score
for i in range (2,5):    
    kmeans = KMeans(n_clusters=i, random_state = 8)
    model2 = kmeans.fit(df2_std)
    labels = model2.labels_
    print(i,':',np.average(silhouette_score(df2_std,labels)))


##choosing n_clusters=3 which created the relatively great silhouette_score
kmeans = KMeans(n_clusters=3, random_state = 8)
model2 = kmeans.fit(df2_std)
labels = model2.labels_
silhouette = silhouette_samples(df2_std,labels)
silhouette_score(df2_std,labels)

##Fitting KMeans for 3 Clusters and find centers of clusters
Km = KMeans(init = 'k-means++', n_clusters =3, n_init = 100, random_state = 8).fit(df2_std)
labels = pandas.DataFrame(Km.labels_)
clustered_data = df2.assign(Cluster = labels)
clustered_data.groupby(['Cluster']).mean()

############################## Grading ##############################
# Import test data
#kickstarter_df_test = pandas.read_excel("/Users/zorak/Desktop/Fall_courses/INSY662/sessions/csv/Kickstarter-Grading-Sample.xlsx")
kickstarter_df_test = pandas.read_excel("Kickstarter-Grading.xlsx")
kickstarter_test = kickstarter_df_test[(kickstarter_df_test['state'] == 'successful') | (kickstarter_df_test['state'] == 'failed')]
kickstarter_test['disable_communication'] = np.where(kickstarter_test['disable_communication'] == True, 'No_communication', 'With_communication')
dummy_state=pandas.get_dummies(kickstarter_test.state)
dummy_disable_communication=pandas.get_dummies(kickstarter_test.disable_communication)

#assign new column as goal*static_usd_rate
kickstarter_test['goal_usd'] = kickstarter_test['static_usd_rate']*kickstarter_test['goal']

# genearte final data for analysis including dummy variables
kickstarter_test = kickstarter_test.join(dummy_state)
kickstarter_test = kickstarter_test.join(dummy_disable_communication)


##################classification model#############
#new dataframe with used variables
kickstarter_used_test = kickstarter_test[["successful", 'goal_usd', 'With_communication',
                              'name_len', 'blurb_len', 'launch_to_deadline_days', 'create_to_launch_days']]
kickstarter_used_test = kickstarter_used_test.dropna()

y_grading = kickstarter_used_test['successful']
a = kickstarter_used_test.drop(columns=['successful'])[X.columns[col]]
#standardize
X_grading = standardizer.fit_transform(a)
# Using the model to predict the results based on the test dataset
y_grading_pred = model1b.predict(a)
m = metrics.accuracy_score(y_grading, y_grading_pred)
print('the accuracy on grading data set is', m)

##################clustering model#############
kmeans = KMeans(n_clusters=3, random_state = 8)
model2 = kmeans.fit(X_grading)
labels = model2.labels_
silhouette = silhouette_samples(X_grading,labels)
silhouette_score(X_grading,labels)

Km = KMeans(init = 'k-means++', n_clusters =3, n_init = 100, random_state = 8).fit(X_grading)
labels = pandas.DataFrame(Km.labels_)
clustered_data = a.assign(Cluster = labels)
grouped = clustered_data.groupby(['Cluster']).mean()
print(grouped)

