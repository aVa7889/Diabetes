# pipeline_utilities
import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LinearRegression
#import statsmodels.api as sm
import glob



def load_data():
    # Get a list of all CSV files in a directory
    csv_files = glob.glob('data/*.csv')

    
    # dfs = [pd.read_csv(file) for csv_file in csv_files]
    combined_df = pd.DataFrame()

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        #df.concat(combined_df.append([csv_file, df])
        combined_df = pd.concat([combined_df, df])
        combined_df.head()
    return combined_df


# def preprocess_data_df(df_[i]):
#     display(df_[i].head())
#     X = df_[i].copy().dropna().drop(columns=df_[i][0])
#     y = df[i][0].values.resahpe(-1,1)
#     # Use train_test_split to separate the data
#     return train_test_split(X, y)








