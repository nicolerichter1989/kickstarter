## add libraries

import os
import pandas as pd
import datetime
import time
import ast

## functions

def create_dataframe():

    '''this function takes all files in a data folder from the current directory and creates a dataframe from them'''

    files = os.listdir('Data')

    df = pd.DataFrame()

    path = os.getcwd() + "\\Data" + "\\"

    for i in files:
        data = pd.read_csv(path + i)
        df = df.append(data)
    
    return df
#
#
#
def drop_and_compare_duplicates(df):

    '''this function drops all duplicated rows and compares before and after'''

    before_dropping_duplicates = df.shape

    df = df.drop_duplicates()

    after_dropping_duplicates = df.shape

    return df
    
#print(before_dropping_duplicates[0] - after_dropping_duplicates[0], 'rows have been dropped!')
#
#
#
def drop_and_compare_duplicate_id(df):

    '''this function drops all duplicated projects and compares before and after'''

    duplicate_project_id_before = df.duplicated(subset='id', keep='first').sum()
    before_dropping_duplicates = df.shape

    df = (df.sort_values(by=['id', 'state_changed_at'], ascending=True).drop_duplicates(subset='id', keep= 'first').reset_index(drop=True))

    duplicate_project_id_after = df.duplicated(subset='id', keep='first').sum()
    after_dropping_duplicates = df.shape

    return df  
    
#print('duplicate IDs present in dataframe (before)', duplicate_project_id_before), print('duplicate IDs present in dataframe (after)', duplicate_project_id_after), print(before_dropping_duplicates[0] - after_dropping_duplicates[0], 'rows have been dropped!')
#
#
#
def get_data_from_timestamp(column, df):
    
    '''this function takes a timestamp and creates 3 new columns for date,time,weekday out of it'''  
    
    date = []
    time = []
    weekday = []
    
    for i in df[f'{column}']:
        date.append(datetime.datetime.fromtimestamp(int(i)).strftime('%Y-%m-%d'))
        time.append(datetime.datetime.fromtimestamp(int(i)).strftime('%H:%M:%S'))
        weekday.append(datetime.datetime.fromtimestamp(int(i)).strftime('%A'))
    
    df[f'{column}' + '_date'] = date
    df[f'{column}' + '_time'] = time
    df[f'{column}' + '_weekday'] = weekday
    
    return df
#
#
#
def get_category_data(column, df):

    '''this function takes a timestamp and creates 3 new columns for date,time,weekday out of it'''  

    slug = []
    parent_name = []

    for i in df[f'{column}']:

        dictionary = ast.literal_eval(i)
        
        slug.append(dictionary.get('slug'))
        parent_name.append(dictionary.get('parent_name'))

    df[f'{column}' + '_slug'] = slug
    df[f'{column}' + '_parent_name'] = parent_name

    return df
#
#
#