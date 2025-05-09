from typing import Tuple

import pandas as pd
import numpy as np
from numpy import ndarray
from sklearn.datasets import load_boston
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


from src.constants import DATA, TEST_SIZE

PARKINSONS_DIR = DATA / "parkinsons"
DIABETES_DIR = DATA / "diabetes"
TRANSFUSION_DIR = DATA / "transfusion"

PARKINSONS_DATA = PARKINSONS_DIR / "parkinsons.data"
DIABETES_DATA = DIABETES_DIR / "diabetes.csv"
TRANSFUSION_DATA = TRANSFUSION_DIR / "transfusion.data"


def parkinson_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    df = pd.read_csv(PARKINSONS_DATA)
    
    # print(df.describe(include='all') )
    # print(df.dtypes)
    
    # Checking Outliers by the boxplots # MDVP contains a lot outliers
    # temp_df = df.drop(columns="name")
    # colList = list(temp_df.columns.values)
    # # print(colList)
    # df.boxplot(colList)
    # plt.show()
    df = df.drop(columns="name")
    X = df.drop(columns="status").values
    y = df["status"].values

    # X = X_temp.iloc[:, :-1].values
    # y = y_temp.iloc[:, -1].values


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test


def liver_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    df = pd.read_csv(TRANSFUSION_DATA)
    # print(df.describe(include='all') )
    # print(df.dtypes)
    
    # Checking Outliers by the boxplots # Monetary contains a lot outliers
    # colList = list(df.columns.values)
    # print(colList)
    # df.boxplot(colList)
    # plt.show()

    X = df.drop(columns="donated").values
    y = df["donated"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return X_train, X_test, y_train.ravel(), y_test.ravel()


def diabetes_dataset() -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    df = pd.read_csv(DIABETES_DATA)
    
    # print(df.describe(include='all') )
    # print(df.dtypes)
    
    # Checking Outliers by the boxplots # Insuline columns contains a lot of outliers
    # colList = list(df.columns.values)
    # print(colList)
    # df.boxplot(colList)
    # plt.show()


    X = df.drop(columns="Outcome").values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)
    return X_train, X_test, y_train, y_test

    # # Doing standarscaler
    # scaler = StandardScaler().fit(X_train)
    # X_train_scaled = scaler.transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    # return X_train_scaled, X_test_scaled, y_train.ravel(), y_test.ravel()



if __name__ == "__main__":
    parkinson_dataset()