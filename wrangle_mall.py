import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
# use get_db_url function to connect to the codeup db
from env import get_db_url

from sklearn.model_selection import train_test_split


# Acquire Data 

def get_mallcustomer_data():
    df = pd.read_sql('SELECT * FROM customers;', get_db_url('mall_customers'))
    return df.set_index('customer_id')


# Prepare Data

def split_data(df, seed=123):
    '''
    This function takes in a pandas dataframe and a random seed. It splits the original
    data into train, test and split dataframes and returns them.
    Test dataset is 20% of the original dataset
    Train is 56% (0.7 * 0.8 = .56) of the original dataset
    Validate is 24% (0.3 * 0.7 = 0.24) of the original dataset
    '''
    train, test = train_test_split(df, train_size=0.8, random_state=seed)
    train, validate = train_test_split(train, train_size=0.7, random_state=seed)
    return train, validate, test

# One hot encoding 
def one_hot_enocde(df):

    dummy_df=pd.get_dummies(df['gender'], dummy_na=False, 
                                drop_first=True)

    # rename columns that have been one hot encoded
    dummy_df = dummy_df.rename(columns={'Male': 'is_male'})  

    # join dummy df to original df
    df = pd.concat([df, dummy_df], axis=1)

    # drop encoded column
    df = df.drop(['gender'], axis=1)
    
    return df

# Scaling
def scale_data(X_train, X_validate, X_test):
    '''
    This function takes in the features for train, validate and test splits. It creates a MinMax Scaler and fits that to the train set.
    It then transforms the validate and test splits and returns the scaled features for the train, validate and test splits.
    '''
    # create scaler
    scaler = MinMaxScaler()
    # Note that we only call .fit with the training data,
    # but we use .transform to apply the scaling to all the data splits.
    scaler.fit(X_train)

    # convert scaled variables to a dataframe 
    X_train_scaled = pd.DataFrame(scaler.transform(X_train),index=X_train.index,
                                    columns=X_train.columns)
    X_validate_scaled = pd.DataFrame(scaler.transform(X_validate), index=X_validate.index,
                                    columns=X_validate.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index,
                                    columns=X_test.columns)

    return X_train_scaled, X_validate_scaled, X_test_scaled
