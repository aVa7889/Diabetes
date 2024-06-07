# pipeline_utilities

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
#from sklearn.svm import SVR
#from sklearn.svm import SVC 
#from sklearn.linear_model import LogisticRegression

#from sklearn.ensemble import RandomForestClassifier
#from sklearn import tree
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

#import statsmodels.api as sm
import glob

def load_data():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('data/*.csv')

    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df])
        first_column_value_counts = combined_df.iloc[:,0].value_counts()
        first_column_shape = combined_df.shape
        print("Loading the DataFrame...")
        display(combined_df)
        display('value counts'.title(),first_column_value_counts)
        display('shape'.title(),first_column_shape)
    return combined_df


def X_y_set():
    data = load_data()
    X = data.copy().dropna().drop(columns=data.columns[0])
    print("Display X below:")
    display(X)
    y = data[data.columns[0]].values.reshape(-1,1)
    print("y")
    display(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("X_train")
    display(X_train)
    print("X_test")
    display(X_test)
    print("y_train")
    display(y_train)
    print("y_test")
    display(y_test)
    return train_test_split(X, y)

def My_X_StandardScaler(X_train, X_test):
    X_scaler = StandardScaler().fit(X_train)
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    print("X_train_scaled")
    display(X_train_scaled)
    print("X_test_scaled")
    display(X_test_scaled)
    return X_train_scaled, X_test_scaled

def My_MinMaxScaler(X_train, X_test):
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    print("MinMax X_train:")
    display(X_train_scaled)

    X_test_scaled = scaler.transform(X_test)
    print("MinMax X_test:")
    display(X_test_scaled)
    return X_train_scaled, X_test_scaled

def r2_adj(x, y, lr):
    r2 = lr.score(x,y)
    n_cols = x.shape[1]
    return 1 - (1 - r2) * (len(y) - 1) / (len(y) - n_cols - 1)

def My_LinearRegression(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    predicted_y = lr.predict(X_test)

    mse = mean_squared_error(y_test, predicted_y)
    r2 = r2_score(y_test, predicted_y)
    rmse = np.sqrt(mse)
    score = lr.score(X_test, predicted_y)
    adj_score = r2_adj(X_test, predicted_y, lr)
    cross_validation_scores = cross_val_score(LinearRegression(), X_train, y_train, scoring = "r2")
    
    print(f"mse: {mse}")
    print(f"R2: {r2}")
    print(f"root mean squared error:  {rmse}")
    print(f"score:  {score}")
    print(f"Adjusted R2:  {adj_score}")
    print(f"All scores: {cross_validation_scores}")
    print(f"Mean score: {cross_validation_scores.mean()}")
    print(f"Standard Deviation: {cross_validation_scores.std()}")
    return mse, r2, rmse, score. adj_score, cross_validation_scores

    


    