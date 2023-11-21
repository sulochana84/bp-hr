
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def bp_regression(sys_pressure, dys_pressure ):
    """ function return the bp values"""
    df = pd.read_csv("Patient_data.csv")
    #add the column name to patient data
    df1 = df.rename(columns={"Unnamed: 0": "Patient"})
    # rearrange the rows of the data frame in correct order
    df1['sort'] = df1['Patient'].str.extract('(\d+)', expand=False).astype(int)
    df1 = df1.sort_values('sort')
    df1 = df1.drop('sort', axis=1)
    df1['sys_pressure'] = sys_pressure
    df1['dys_pressure'] = dys_pressure
    target_cols = ['sys_pressure', 'dys_pressure']
    Data_cols = ["PP", "FF", "PF", "FP", "pp_ff", "pp_fp", "fp_ff", "fp_pf", "ppg_height", "af_height", "ad_height",
                 "systolic_area", "diastolic_area", "ratio_area", "total_area", "delta_time", "reflection_index"]
    # independent variables
    X = df1[Data_cols]
    # Dependent variables
    Y = df1[target_cols]
    #check the null values
    df.isnull().values.any()
    # split the data set for testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=15)
    # linear regression model
    lr_model = LinearRegression()
    # Fit the model X and y
    lr_model.fit(X_train, y_train)
    #save the model
    pickle.dump(lr_model, open('lr_model.pkl', 'wb'))
    #prediction
    pickled_model = pickle.load(open('lr_model.pkl', 'rb'))
    return pickled_model

