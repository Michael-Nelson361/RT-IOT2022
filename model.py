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

# Warning handling
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning

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
    
def model_testing(X_train,y_train,warn_suppress=True):
    """
    Runs through the 5 predetermined algorithms and runs basic testing.
    
    Paramaters:
    -----------
    - X_set: list
        List containing 
    - y_set: list
    - warn_suppress: Boolean, default = True
        Whether to display the warnings being given or not
    
    Returns:
    --------
    - DataFrame containing results of the grid search.
    
    """
    # Calculate elapsed times
    import time
    start_time = time.time()
    
    
    # handle warnings
    if warn_suppress:
        print('WARNINGS SUPPRESSED.')
        
        # suppress divide by zero and other runtime warnings
        warnings.filterwarnings('ignore',category=RuntimeWarning)
        
        # suppress convergence warnings
        warnings.filterwarnings('ignore',category=ConvergenceWarning)
    
    # Build the pipeline
    print("Assembling pipeline...")
    pipeline = Pipeline([
        ('scalar',RobustScaler()),
        ('kbest',SelectKBest()),
        ('clf',LogisticRegression())
    ])
    
    # Build the search space
    print("Assembling search parameters...")
    search_space = [
        {'kbest__k':[25,50,'all']},
        {'clf':[LogisticRegression(random_state=42)],
             'clf__C':[1.0,0.1]
        },
        {'clf':[DecisionTreeClassifier(random_state=42)],
             'clf__max_depth':[3,6,9]
        },
        {'clf':[RandomForestClassifier(random_state=42)],
             'clf__max_depth':[3,6,9]
        },
        {'clf':[SVC(random_state=42)],
             'clf__C':[1.0,0.1]
        },
        {'clf':[MLPClassifier(random_state=42)],
             'clf__hidden_layer_sizes':[(50,),(100,),(150,)]
        }
    ]
    
    # Build and fit the GridSearchCV
    print("Building the grid search....")
    gscv = GridSearchCV(pipeline,search_space,scoring='accuracy',cv=5,verbose=1)

    print("Fitting models...")
    gscv.fit(X_train,y_train)
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print(f'Models fit. Elapsed time: {elapsed_time} seconds.')
    
    return pd.DataFrame(gscv.cv_results_)

# def print_report(results,X_set,y_set,n_show=3,suppress_warnings=True):
#     """
#     Shows the top 3 performing algorithms and their classification reports
    
#     Parameters:
#     -----------
#     - results: DataFrame
#         A DataFrame containing the cv_results_ of a grid search
#     - X_set: List
#         A list containing the 3 DataFrames of X_train, X_validate, and X_test. Does not use the test set.
#     - y_set: List
#         A list containing the 3 Series of y_train, y_validate, and y_test. Does not use the test set.
#     - n_show: Integer, default=3
#         How many of the top algorithms to show
#     - suppress_warnings: Boolean, default=True
#         Whether to suppress warnings or show them.
    
#     Returns:
#     --------
#     - None
#     """
    
#     # Suppress warnings
#     if suppress_warnings:
#         warnings.filterwarnings('ignore')
#         warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
#     # Sort by test score
#     sorted_results_df = results.sort_values('mean_test_score', ascending=False)

#     # Calculate the top 30% cutoff
#     cutoff_index = int(len(sorted_results_df) * 0.3)
#     top_30_df = sorted_results_df.head(cutoff_index)
    
#     # Extract algorithm names assuming param_clf is directly usable or adjust accordingly
#     top_30_df['algorithm_name'] = top_30_df['param_clf'].copy().apply(lambda x: type(x).__name__)
#     top_algorithms = top_30_df['algorithm_name'].value_counts().nlargest(n_show).index.tolist()
#     print(f"Top {n_show} Algorithm(s):", top_algorithms)
    
#     classification_reports = []
#     for algo in top_algorithms:
#         # Filter the DataFrame for this specific algorithm
#         algo_df = top_30_df[top_30_df['algorithm_name'] == algo]

#         # Get the best set of parameters for this algorithm
#         best_params = algo_df.iloc[0]['params']

#         # Set up the estimator with the best parameters
#         estimator = gscv.estimator.set_params(**best_params)

#         # Fit the estimator on the full dataset (or a training subset if specified)
#         estimator.fit(X_set[0], y_set[0])

#         # Optionally, evaluate on a test set and generate classification reports
#         y_pred = estimator.predict(X_set[1])
#         report = classification_report(y_set[1], y_pred)
#         classification_reports.append(report)

#     fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#     for i, report in enumerate(classification_reports):
#         axs[i].text(0.01, 1, report, {'fontsize': 10}, fontfamily='monospace', verticalalignment='top', horizontalalignment='left')
#         axs[i].axis('off')
#         axs[i].set_title(f"Report for {top_salgorithms[i]}")

#     plt.tight_layout()
#     plt.show()
    
#     return None

def print_report(results, X_set, y_set, n_show=3, suppress_warnings=True):
    """
    Shows the top performing algorithms and their classification reports
    
    Parameters:
    -----------
    - results: DataFrame
        A DataFrame containing the cv_results_ of a grid search
    - X_set: List
        A list containing the DataFrames of X_train, X_validate, and X_test. Does not use the test set.
    - y_set: List
        A list containing the Series of y_train, y_validate, and y_test. Does not use the test set.
    - n_show: Integer, default=3
        How many of the top algorithms to show
    - suppress_warnings: Boolean, default=True
        Whether to suppress warnings or show them.
    
    Returns:
    --------
    - None
    """
    
    # Suppress warnings
    if suppress_warnings:
        warnings.filterwarnings('ignore')
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    
    # Sort by test score
    sorted_results_df = results.sort_values('mean_test_score', ascending=False)

    # Calculate the top 30% cutoff
    cutoff_index = int(len(sorted_results_df) * 0.4)
    top_30_df = sorted_results_df.head(cutoff_index)
    
    # Extract algorithm names assuming param_clf is directly usable or adjust accordingly
    top_30_df['algorithm_name'] = top_30_df['param_clf'].apply(lambda x: type(x).__name__)
    top_algorithms = top_30_df['algorithm_name'].value_counts().nlargest(n_show).index.tolist()
    print(f"Top {len(top_algorithms)} Algorithm(s):", top_algorithms)
    
    classification_reports = []
    for algo in top_algorithms:
        # Filter the DataFrame for this specific algorithm
        algo_df = top_30_df[top_30_df['algorithm_name'] == algo]

        # Get the index of the best row for this algorithm
        best_row_index = algo_df.index[0]
        
        # Extract the estimator
        best_estimator = results.loc[best_row_index, 'param_clf']
        
        # Fit the estimator on the full dataset (or a training subset if specified)
        best_estimator.fit(X_set[0], y_set[0])

        # Optionally, evaluate on a validation set and generate classification reports
        y_pred = best_estimator.predict(X_set[1])
        report = classification_report(y_set[1], y_pred)
        classification_reports.append(report)

    # Plot the classification reports side by side
    fig, axs = plt.subplots(1, len(top_algorithms), figsize=(15, 5))
    for i, report in enumerate(classification_reports):
        axs[i].text(0.01, 1, report, {'fontsize': 10}, fontfamily='monospace', verticalalignment='top', horizontalalignment='left')
        axs[i].axis('off')
        axs[i].set_title(f"Report for {top_algorithms[i]}")

    plt.tight_layout()
    plt.show()
    
    return None

def final_models(X_train,y_train,suppress_warnings=True):
    """
    Return results of the top performing models using a grid search and pre-defined parameters.
    
    Parameters:
    -----------
    - X_train: DataFrame
        The dataset to be trained on
    - y_train: Series
        The target variable
    - suppress_warnings: Boolean, default=True
    
    Returns:
    --------
    DataFrame containing the results of the grid search.
    """
    
    start_time = time.time()
    
    # handle warnings
    if suppress_warnings:
        print('WARNINGS SUPRESSED.')
        
        # suppress divide by zero and other runtime warnings
        warnings.filterwarnings('ignore',category=RuntimeWarning)
        
        # suppress convergence warnings
        warnings.filterwarnings('ignore',category=ConvergenceWarning)
    
    print('Building search space...')
    search_space = [
        {
            'clf':[MLPClassifier(random_state=42)],
            'clf__hidden_layer_sizes':[(75,),(100,),(125,)],
            'clf__alpha':np.logspace(-4,2,20),
            'clf__learning_rate':['constant','invscaling','adaptive']
        },
        {
            'clf':[RandomForestClassifier(random_state=42)],
            'clf__n_estimators':[75,100,125],
            'clf__max_depth':[3,6,9,12,15],
            'clf__class_weight':['balanced','balanced_subsample']
        },
        {
            'kbest__k':[40,50,60,'all']
        }
    ]
    
    print('Building pipeline...')
    finalist_pipeline = Pipeline(
        [
            ('scalar',RobustScaler()),
            ('kbest',SelectKBest()),
            ('clf',MLPClassifier())
        ]
    )
    
    print('Building grid search...')
    gscv = GridSearchCV(finalist_pipeline,search_space,cv=5,scoring='accuracy',verbose=1)

    print('Fitting models...')
    gscv.fit(X_train,y_train)
    
    print(f'Models fitted. Elapsed time:{time.time() - start_time}')
    
    return pd.DataFrame(gscv.cv_results_)