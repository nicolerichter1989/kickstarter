## add libraries

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

## functions

def classification_tree():

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    return
#
#
#
def classification_KNN():

    '''write description'''

    #what needs to be the input?
    #what needs to be the return?

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = KNeighborsRegressor(n_neighbors=4)
    model.fit(X_train, y_train)

    return
#
#
#
def classification_LogisticRegression():

    '''write description'''

    return
#
#
#




def x_y_split(target, df):

    '''write description'''

    y = df[target]
    X = df.drop(target, axis=1)

    return df
#
#
#
def train_test_split(model):

    '''write description'''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classification = LogisticRegression(random_state=42, max_iter=1000)
    classification.fit(X_train, y_train)

    return x
#
#
#
def confusion_matrix():

    '''write description'''

    group_names = ['True failed', 'False failed', 'False success', 'True success']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    return sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
#
#
#



### write functions to be able to easier run all different model I want to try


