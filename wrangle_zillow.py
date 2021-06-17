import pandas as pd
import numpy as np
import os
# use get_db_url function to connect to the codeup db
from env import get_db_url

import matplotlib.pyplot as plt

def get_zillow_data(cached=False):
    '''
    This function returns the zillow database as a pandas dataframe. 
    If the data is cached or the file exists in the directory, the function will read the data into a df and return it. 
    Otherwise, the function will read the database into a dataframe, cache it as a csv file
    and return the dataframe.
    '''
    # If the cached parameter is false, or the csv file is not on disk, read from the database into a dataframe
    if cached == False or os.path.isfile('zillow_df.csv') == False:
        sql_query = '''
        SELECT *
        FROM properties_2017 prop
        JOIN propertylandusetype landuse ON prop.propertylandusetypeid = landuse.propertylandusetypeid
       JOIN architecturalstyletype USING (architecturalstyletypeid)
        	LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
        	LEFT JOIN storytype st ON prop.storytypeid = st.storytypeid
        	LEFT JOIN typeconstructiontype ct ON prop.typeconstructiontypeid = ct.typeconstructiontypeid
        	JOIN 
        		(SELECT parcelid, max(logerror) AS logerror, max(transactiondate) AS transactiondate
                FROM predictions_2017
                GROUP BY parcelid) AS pred_2017
                USING(parcelid)
        	WHERE transactiondate LIKE '2017-%%-%%'
        	AND parcelid IN(
                            SELECT DISTINCT parcelid)
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL; 
        '''
        zillow_df = pd.read_sql(sql_query, get_db_url('zillow'))
        #also cache the data we read from the db, to a file on disk
        zillow_df.to_csv('zillow_df.csv')
    else:
        # either the cached parameter was true, or a file exists on disk. Read that into a df instead of going to the database
        zillow_df = pd.read_csv('zillow_df.csv', index_col=0)
    # return our dataframe regardless of its origin
    return zillow_df


def summarize(df):
    '''
    This function will take in a single argument (a pandas dataframe) and 
    output to console various statistices on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('----------------------')
    print('Dataframe head')
    print(df.head(3))
    print('----------------------')
    print('Dataframe Info ')
    print(df.info())
    print('----------------------')
    print('Dataframe Description')
    print(df.describe())
    print('----------------------')
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('----------------------')
    print('Dataframe value counts ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            # define bins for continuous columns and don't sort them
            print(df[col].value_counts(bins=10, sort=False))
    print('----------------------')
    print('nulls in df by column')
    print(nulls_by_col(df))
    print('----------------------')
    print('null in df by row')
    print(nulls_by_row(df))
    print('----------------------')


def nulls_by_col(df):
    '''
    This function  takes in a dataframe of observations and attributes(or columns) and returns a dataframe where each row is an atttribute name, the first column is the 
    number of rows with missing values for that attribute, and the second column is percent of total rows that have missing values for that attribute.
    '''
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 
                                 'percent_rows_missing': prcnt_miss})
    return cols_missing
    

def nulls_by_row(df):
    '''
    This function takes in a dataframe and returns a dataframe with 3 columns: the number of columns missing, percent of columns missing, 
    and number of rows with n columns missing.
    '''
    num_missing = df.isnull().sum(axis = 1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 
                                 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'index': 'num_rows'}).reset_index().set_index('num_cols_missing')
    return rows_missing


# Data cleanign functions 

def remove_columns(df, cols_to_remove):
    '''
    This function takes in a pandas dataframe and a list of columns to remove. It drops those columns from the original df and returns the df.
    '''
    df = df.drop(columns=cols_to_remove)
    return df
                 
                 
def handle_missing_values(df, prop_required_column=0.5 , prop_required_row=0.75):
    '''
    This function takes in a pandas datafeame, default proportion of required columns (set to 50%) and proprtion of required rows (set to 75%).
    It drops any rows or columns that contain null values more than the threshold specified from the original dataframe and returns that dataframe.
    '''
    threshold = int(round(prop_required_column * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    return df

# combined in one function
def data_prep(df, cols_to_remove=[], prop_required_column=0.5, prop_required_row=0.75):
    '''
    This function calls the remove_columns and handle_missing_values to drop columns that need to be removed. It also drops rows and columns that have more 
    missing values than the specified threshold.
    '''
    df = remove_columns(df, cols_to_remove)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    return df

