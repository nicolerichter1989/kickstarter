## add libraries

import pandas as pd
import numpy as np
import math
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

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

    return cf_matrix
#
#
#
def encode_categoricals(X):

    '''write description'''

    # split into numerical and categorical variables
    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)

    # encode categorical variables 
    X_cat = pd.get_dummies(X_cat) 
 
    # create new df from encoded cat_df and num_df (define X again)
    X = pd.concat([X_cat, X_num], axis = 1)

    return X
#
#
#
def standardizer(X):

    '''write description'''

    # split into numerical and categorical variables
    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)

    # standardize
    transformer = StandardScaler().fit(X_num)
    x_standardized = transformer.transform(X_num)
 
    # create new df from encoded cat_df and num_df (define X again)
    X = pd.concat([X_cat, x_standardized], axis = 1)

    return X
#
#
#
def min_max_imizer(X):

    '''write description'''

    # split into numerical and categorical variables
    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)

    # standardize
    transformer = MinMaxScaler().fit(X_num)
    x_min_max = transformer.transform(X_num)
 
    # create new df from encoded cat_df and num_df (define X again)
    X = pd.concat([X_cat, x_min_max], axis = 1)

    return X
#
#
#
def normalizer(X):

    '''write description'''

    # split into numerical and categorical variables
    X_num = X.select_dtypes(include = np.number)
    X_cat = X.select_dtypes(include = object)

    # standardize
    transformer = Normalizer().fit(X_num)
    x_normalized = transformer.transform(X_num)
 
    # create new df from encoded cat_df and num_df (define X again)
    X = pd.concat([X_cat, x_normalized], axis = 1)

    return X
#
#
#
