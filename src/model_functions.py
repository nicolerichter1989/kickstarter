## add libraries

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_squared_error

## functions

def classification_tree(X, y):

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # make predictions
    predictions = model.predict(X_test)
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)   
    RSME = math.sqrt(MSE)

    return print('MAE' + MAE, 'MSE' +  MSE, 'RSME' +  RSME)
#
#
#
def classification_KNN(X, y):

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = KNeighborsRegressor(n_neighbors=4)
    model.fit(X_train, y_train)

    # make predictions
    predictions = model.predict(X_test)
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)   
    RSME = math.sqrt(MSE)

    return print('MAE' + MAE, 'MSE' +  MSE, 'RSME' +  RSME)
#
#
#
#
def classification_LogisticRegression(X, y):

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
 
    # make predictions
    predictions = model.predict(X_test)
    MAE = mean_absolute_error(y_test, predictions)
    MSE = mean_squared_error(y_test, predictions)   
    RSME = math.sqrt(MSE)

    return print('MAE' + MAE, 'MSE' +  MSE, 'RSME' +  RSME)
#
#
#
def confusion_matrix(y_test, predictions):

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    cf_matrix = confusion_matrix(y_test, predictions)
    group_names = ['True A', 'False A','False B', 'True B']

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    return
#
#
#





