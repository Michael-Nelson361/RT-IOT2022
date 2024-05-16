# Basic imports
import pandas as pd
import numpy as np
import time

# SciKitLearn imports
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

# SciKitLearn modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import matplotlib for text arrangement
import matplotlib.pyplot as plt

def encode_df(dframe,target):
	'''
	Takes a processed dataframe and encodes the object columns for usage in modeling.
		Takes a dataframe and a target variable (assuming the target variable is an object). Target variable keeps the thing the model is being trained on from splitting and altering it.
		!!! MAKE ME MORE DYNAMIC !!!
	- Add functionality to check if passed a list or dataframe
	- If dataframe, then run standard loop
	- If list then check if each item is a dataframe (checking for train/validate/test)
	- If list and each item is dataframe, then try loop on each dataframe
	- Otherwise return an error
	'''
	df = dframe.copy()
	# Get the object columns from the dataframe
	obj_col = [col for col in df.columns if df[col].dtype == 'O']
		# remove target variable
	obj_col.remove(target)
		# Begin encoding the object columns
	for col in obj_col:
		# Grab current column dummies
		dummies = pd.get_dummies(df[col])
				# concatenate the names in a descriptive manner
		dummies.columns = [col+'_is_'+column for column in dummies.columns]
		# add these new columns to the dataframe
		for column in dummies.columns:
			df[column] = dummies[column].astype(float)
				# Drop the old columns from the dataframe
		df = df.drop(columns=col)
	return df 

def fill_missing_columns(df1, df2, df3):
    # Step 1: Collect all unique columns from all three dataframes
    all_columns = set(df1.columns) | set(df2.columns) | set(df3.columns)
    
    # Step 2: Find missing columns in each dataframe and add them
    for df in (df1, df2, df3):
        missing_columns = all_columns - set(df.columns)
        for col in missing_columns:
            df[col] = 0  # Add the missing column filled with zeros
    
    return df1, df2, df3

def pca_transform(X,y,ranstate=42):
    """
    Return 
    
    Parameters:
    -----------
    - X: DataFrame
        Data to be transformed
    - y: Series
        Target variable
    - ranstate: Int (Optional)
        Default = 42
    
    Returns:
    --------
    
    """
    
    # build the selector
    pca = PCA(random_state=ranstate)
    
    # export as dataframe
    df = pd.DataFrame(pca.fit_transform(X,y),columns=pca.get_feature_names_out())
    
    return df
    
def kbest(X,y):
    """
    Returns a dataframe containing the top 50 KBest features.
    
    Parameters:
    -----------
    - X: DataFrame
        The data to be trained on
    - y: Series
        The target variable
    
    Returns:
    --------
    - kbest: DataFrame
        The transformed dataset
    
    """
    
    # Build the model
    kbest = fselect.SelectKBest(k=50)
    
    # fit and transform the model
    df_kbest = pd.DataFrame(data=kbest.fit_transform(X,y),columns=kbest.get_feature_names_out())
    
    return df_kbest

