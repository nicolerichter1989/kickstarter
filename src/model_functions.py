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
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

## functions

def classification_tree(model_name, X, y):

    '''write description'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # calculate score and predictions
    score = "{:.2%}".format(round(model.score(X_test, y_test),4))
    predictions = predictions = model.predict(X_test)

    # create confusion_matrix
    confusion = metrics.confusion_matrix(y_test, predictions)
    true_positives = confusion[1,1]
    true_negatives = confusion[0,0]
    false_positives = confusion[0,1]
    false_negatives = confusion[1,0]

    accuracy = "{:.2%}".format(round(((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)),4))
    sensitivity = "{:.2%}".format(round((true_positives/(true_positives+false_negatives)),4))
    sepcificity = "{:.2%}".format(round(((+true_negatives)/(true_negatives+false_positives)),4))

    return print('model: ', f'{model_name}', 'accuracy: ', accuracy, ' - ', 'sensitivity: ', sensitivity, ' - ', 'specificity: ', sepcificity)
#
#
#
def classification_KNN(model_name, X, y):

    '''write description'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = KNeighborsRegressor(n_neighbors=4)
    model.fit(X_train, y_train)

    # calculate score and predictions
    score = "{:.2%}".format(round(model.score(X_test, y_test),4))
    predictions = predictions = model.predict(X_test)

    # create confusion_matrix
    confusion = metrics.confusion_matrix(y_test, predictions)
    true_positives = confusion[1,1]
    true_negatives = confusion[0,0]
    false_positives = confusion[0,1]
    false_negatives = confusion[1,0]

    accuracy = "{:.2%}".format(round(((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)),4))
    sensitivity = "{:.2%}".format(round((true_positives/(true_positives+false_negatives)),4))
    sepcificity = "{:.2%}".format(round(((+true_negatives)/(true_negatives+false_positives)),4))

    return print('model: ', f'{model_name}', 'accuracy: ', accuracy, ' - ', 'sensitivity: ', sensitivity, ' - ', 'specificity: ', sepcificity)
#
#
#
def classification_LogisticRegression(model_name, X, y):

    '''write description'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # calculate score and predictions
    score = "{:.2%}".format(round(model.score(X_test, y_test),4))
    predictions = predictions = model.predict(X_test)

    # create confusion_matrix
    confusion = metrics.confusion_matrix(y_test, predictions)
    true_positives = confusion[1,1]
    true_negatives = confusion[0,0]
    false_positives = confusion[0,1]
    false_negatives = confusion[1,0]

    accuracy = "{:.2%}".format(round(((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)),4))
    sensitivity = "{:.2%}".format(round((true_positives/(true_positives+false_negatives)),4))
    sepcificity = "{:.2%}".format(round(((+true_negatives)/(true_negatives+false_positives)),4))

    return print('model: ', f'{model_name}', 'accuracy: ', accuracy, ' - ', 'sensitivity: ', sensitivity, ' - ', 'specificity: ', sepcificity)
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

    # change np.array into df
    x_array = pd.DataFrame(x_standardized, columns = X_num.columns)

    # create new df from encoded cat_df and num_df (define X again)
    Xresult = pd.concat([X_cat, x_array], axis = 1)

    return Xresult
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
 
    # change np.array into df
    x_array = pd.DataFrame(x_min_max, columns = X_num.columns)

    # create new df from encoded cat_df and num_df (define X again)
    Xresult = pd.concat([X_cat, x_array], axis = 1)

    return Xresult
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
 
    # change np.array into df
    x_array = pd.DataFrame(x_normalized, columns = X_num.columns)

    # create new df from encoded cat_df and num_df (define X again)
    Xresult = pd.concat([X_cat, x_array], axis = 1)

    return Xresult
#
#
#