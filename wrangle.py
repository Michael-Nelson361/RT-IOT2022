import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# define a random state
ranstate = 42

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
    try:
        from ucimlrepo import fetch_ucirepo 
        
        # Fetch dataset 
        rt_iot2022 = fetch_ucirepo(id=942)

        # Concatenate the two datasets
        df = pd.concat([rt_iot2022.data.features,rt_iot2022.data.targets],axis=1)

        return df.copy(),rt_iot2022
    
    except:
        print("Import failed due to 1 or more of the following reasons:\n\t - User is missing the UC Irvine Python package.\n\t - Dataset is no longer available at the queried location.\n\t - Some of the libraries in use have changed.")

        
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

