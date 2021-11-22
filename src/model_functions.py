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
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from scipy.stats.mstats import winsorize

## functions

def classification_tree(model_name, X, y):

    '''this function takes a model name, X and y and returns different prediction scores for the classification tree model for those inputs'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

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
def classification_KNN(model_name, X, y, nnn):

    '''this function takes a model name, X and y and returns different prediction scores for the classification KNN model for those inputs'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = KNeighborsClassifier(n_neighbors=nnn)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # create confusion_matrix
    confusion = confusion_matrix(y_test, predictions)
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

    '''this function takes a model name, X and y and returns different prediction scores for the Logistic Regression model for those inputs'''

    # split in train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # pick and fit model  
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # create confusion_matrix
    confusion = metrics.confusion_matrix(y_test, predictions)
    true_positives = confusion[1,1]
    true_negatives = confusion[0,0]
    false_positives = confusion[0,1]
    false_negatives = confusion[1,0]

    accuracy = "{:.2%}".format(round(((true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)),4))
    sensitivity = "{:.2%}".format(round((true_positives/(true_positives+false_negatives)),4))
    sepcificity = "{:.2%}".format(round(((true_negatives)/(true_negatives+false_positives)),4))

    return print('model: ', f'{model_name}', 'accuracy: ', accuracy, ' - ', 'sensitivity: ', sensitivity, ' - ', 'specificity: ', sepcificity)
#
#
#
def encode_categoricals(X):

    '''this function takes an X variable and encodes all categoricals for X'''

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

    '''this function takes an X variable and standardizes with standardscaler'''

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

    if len(X_cat.columns) == 0:
        X_out = x_array
    else:
        X_out = Xresult

    return X_out
#
#
#
def min_max_imizer(X):

    '''this function takes an X variable and standardizes with min_max_scaler'''

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

    if len(X_cat.columns) == 0:
        X_out = x_array
    else:
        X_out = Xresult

    return X_out
#
#
#
def normalizer(X):

    '''this function takes an X variable and standardizes with normalizer'''

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

    if len(X_cat.columns) == 0:
        X_out = x_array
    else:
        X_out = Xresult

    return X_out
#
#
#
def fences(df, column):
    
    '''this function takes an X variable and standardizes with normalizer'''
  
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3-q1
    
    outer_fence = 3*iqr
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence
    
    return outer_fence_le, outer_fence_ue
#
#
#
def closest_fence(df, position):
   
    '''this function takes an X variable and standardizes with normalizer'''

    quantiles = [0.900,0.925,0.950,0.975,0.990,0.999]
    distance = []

    for i in df:

        a = abs((df[i].quantile(0.900)) - (fences(df, i)[position]))
        b = abs((df[i].quantile(0.925)) - (fences(df, i)[position]))
        c = abs((df[i].quantile(0.950)) - (fences(df, i)[position]))
        d = abs((df[i].quantile(0.975)) - (fences(df, i)[position]))
        e = abs((df[i].quantile(0.990)) - (fences(df, i)[position]))
        f = abs((df[i].quantile(0.999)) - (fences(df, i)[position]))
        distance.append([a,b,c,d,e,f])

    new_quantile = []

    for i in distance:
        index = i.index(min(i))
        new_quantile.append(quantiles[index])
    
    return new_quantile
#
#
#
def wins(column, df, target):
  
    '''this function takes an X variable and standardizes with normalizer'''

    wins = []

    for iteration in enumerate(df):
        right_limit = ("{:.2}".format(round((1-(target[iteration])),3)))

    df[f'{column}' + '_wins'] = winsorize((df[f'{column}']), limits=(0, right_limit))

    return df
#
#
#