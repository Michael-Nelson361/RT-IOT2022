import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import os

# define a random state
ranstate = 42

def check_file_exists(filename):
    """
    Function takes a filename and checks if the file exists. If the file it will load it, otherwise it will return 0.
    """
    if os.path.exists(filename):
        print('Reading from file...')
        df = pd.read_csv(filename,index_col=0)
        
        return df
    else:
        return 0 

def acquire_iot2022():
    """
    Function to download and import the RT-IoT2022 dataset. Takes no parameters. Assumes the ucimlrepo has already been installed.
    
    Parameters:
    -----------
    - None
    
    Return:
    -------
    - Pandas DataFrame containing the RT IoT2022 information dataset
    - Dictionary object containing all RT IoT2022 data, including metadata.
    """
    # Define the filename
    fname = 'RT_IoT2022.csv'
    
    df = check_file_exists(filename=fname)
    
    # download the data if 0 was returned
    if type(df)==type(0):
        try:
            from ucimlrepo import fetch_ucirepo 

            # Fetch dataset 
            rt_iot2022 = fetch_ucirepo(id=942)

            # Concatenate the two datasets
            df = pd.concat([rt_iot2022.data.features,rt_iot2022.data.targets],axis=1)
            
            # Create a csv copy
            df.to_csv(fname)

        except:
            print("Import failed due to 1 or more of the following reasons:\n\t - User is missing the UC Irvine Python package.\n\t - Dataset is no longer available at the queried location.\n\t - Some of the libraries in use have changed.")
            
    # Return the dataframe
    return df.copy()

        
def df_info(df,include=False,samples=1):
	"""
	Function takes a dataframe and returns potentially relevant information about it (including a sample)
	include=bool, default to False. To add the results from a describe method, pass True to the argument.
	samples=int, default to 1. Shows 1 sample by default, but can be modified to include more samples if desired.
	"""
		# create the df_inf dataframe
	df_inf = pd.DataFrame(index=df.columns,
			data = {
				'nunique':df.nunique()
				,'dtypes':df.dtypes
				,'isnull':df.isnull().sum()
			})
		# append samples based on input
	if samples >= 1:
		df_inf = df_inf.merge(df.sample(samples).iloc[0:samples].T,how='left',left_index=True,right_index=True)
		# append describe results if option selected
	if include == True:
		return df_inf.merge(df.describe(include='all').T,how='left',left_index=True,right_index=True)
	elif include == False:
		return df_inf
	else:
		print('Value passed to "include" argument is invalid.')

def data_split(df,ranstate):
    """
    Function to split the given DataFrame. Returns three DataFrames containing the split data.
    
    Parameters:
    -----------
    - df: DataFrame
        The set of data to be split up
    - ranstate: Integer
        An integer value to define the random state
    
    Return:
    -------
    - train: DataFrame
    - validate: DataFrame
    - test: DataFrame
    
    """
    
    # Use a copy of the dataset instead of the data itself
    copy = df.copy()
    
    # Split into the train and val_test sets
    train, val_test = train_test_split(
        copy,
        train_size=0.7,
        test_size=0.3,
        random_state=ranstate,
        stratify=copy.Attack_type
    )
    
    # Split the val_test into validate and test sets
    validate, test = train_test_split(
        val_test,
        train_size=0.7,
        test_size=0.3,
        random_state=ranstate,
        stratify=val_test.Attack_type
    )
    
    return train, validate, test

def clean_data(df):
    """
    Take the RT_IoT2022 dataframe and clean and prepare it for usage.
    
    Parameters:
    -----------
    - df: DataFrame
        The DataFrame (train, validate, test, or full) to be cleaned
    
    Return:
    -------
    - copy_df: DataFrame
        The DataFrame cleaned and prepared for usage
    
    """
    
    # Create a copy to work with
    copy_df = df.copy()
    
    # remap using lambda and apply
    copy_df.service = copy_df.service.apply(lambda x: 'none' if x == '-' else x)
    
    # Fix the spelling error on ARP poisoning
    copy_df.Attack_type.apply(lambda x:'ARP_poisoning' if x=='ARP_poisioning' else x)
    
    # Initiate a collector list
    single_val_cols = []

    # Find any columns that have only a single value in them
    for col in copy_df.columns:
        # print(col)

        if len(copy_df[col].value_counts()) == 1:
            # print(col)
            single_val_cols.append(col)
    
    # Drop any single value columns
    copy_df = copy_df.drop(single_val_cols,axis=1,errors='ignore')
    
    # Add a column denoting attack or normal pattern
    normal_pattern = ['MQTT_Publish','Thing_Speak','Wipro_bulb','Amazon-Alexa']
    copy_df['traffic_type'] = np.where(copy_df.Attack_type.isin(normal_pattern),'Normal','Attack')
    
    return copy_df

def wrangle_iot2022():
    """
    Summary function that acquires, cleans, and returns the RT_IoT2022 data
    """
    ranstate = 42
    
    # Acquire
    df = acquire_iot2022()
    
    # Clean
    df = clean_data(df)
    
    # Split
    train,validate,test = data_split(df,ranstate=ranstate)
    
    return [train,validate,test]