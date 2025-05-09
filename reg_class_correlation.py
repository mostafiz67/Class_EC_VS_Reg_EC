"""
Author: Md Mostafizur Rahman
File: Calculate Regression Error Consistency using different methods and benchmark datasets
Run Command: python3 reg_class_correlation.py --dataset=Parkinsons

Classification EC: https://github.com/stfxecutables/error-consistency
Some code is inherited from https://stackoverflow.com/questions/71430032/how-to-compare-two-numpy-arrays-with-multiple-condition
"""
from itertools import combinations
from typing import Any, Dict, List, Tuple
from warnings import filterwarnings

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score
from error_consistency.consistency import error_consistencies
import argparse

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from src import data_preprocess
from src.constants import DATASET_NAMES, EC_METHODS, OUT
# from src.classification_EC import calculate_ECs

from src.constants import ECMethod
ECMethod = ["ratio", "ratio-diff", "ratio-signed", "ratio-diff-signed", 
            "intersection_union_sample", "intersection_union_all", "intersection_union_distance"]


def regression_ec(residuals: List[ndarray], method: ECMethod) -> List[ndarray]:
    filterwarnings("ignore", "invalid value encountered in true_divide", category=RuntimeWarning)
    consistencies = []
    for pair in combinations(residuals, 2):
        r1, r2 = pair
        r = np.vstack(pair)
        sign = np.sign(np.array(r1) * np.array(r2))
        if method == "ratio-signed":
            consistency = np.multiply(sign, np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0))
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio":
            consistency = np.min(np.abs(r), axis=0) / np.max(np.abs(r), axis=0)
            consistency[np.isnan(consistency)] = 1
        elif method == "ratio-diff-signed":
            consistency = np.multiply(sign, (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2)))
            consistency[np.isnan(consistency)] = 0
        elif method == "ratio-diff":
            consistency = (np.abs(np.abs(r1) - np.abs(r2))) / (np.abs(r1) + np.abs(r2))
            consistency[np.isnan(consistency)] = 0
        elif method =="intersection_union_sample":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(numerator, denominator)
            consistency[np.isnan(consistency)] = 1
        elif method =="intersection_union_all":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choice_numerator = [np.minimum(r1, r2), np.minimum(np.abs(r1), np.abs(r2))]
            choice_denominator = [np.maximum(r1, r2), np.maximum(np.abs(r1), np.abs(r2))]
            numerator = np.select(conditions, choice_numerator, np.zeros(len(r1)))
            denominator = np.select(conditions, choice_denominator, np.abs(np.add(np.abs(r1), np.abs(r2))))
            consistency = np.divide(np.sum(numerator), np.sum(denominator)) # all sum and then divide
            consistency = np.nan_to_num(consistency, copy=True, nan=1.0)
        elif method =="intersection_union_distance":
            conditions = [((r1>=0)&(r2>=0)), ((r1<=0)&(r2<=0))]
            choiceValue = [np.abs(np.subtract(np.abs(r1), np.abs(r2))), np.abs(np.subtract(np.abs(r1), np.abs(r2)))]
            consistency = np.select(conditions, choiceValue, np.add(np.abs(r1), np.abs(r2)))
        else:
            raise ValueError("Invalid method")
        consistencies.append(consistency)
    filterwarnings("default", "invalid value encountered in true_divide", category=RuntimeWarning)
    # print(np.shape(consistencies))
    return consistencies


def calculate_ECs(
    dataset: Tuple[ndarray, ndarray, ndarray, ndarray],
    reg_name: Any,
    k: int,
) -> DataFrame:

    X_train, X_test, y_train, y_test = dataset
    fold_reg_residuals, fold_dfs, fold_class_preds = [], [], []
    kf = KFold(n_splits=k, shuffle=True)
    for train_index, _ in kf.split(X_train):
        if reg_name in ["Linear"]:
            reg_preds = LinearRegression().fit(X_train[train_index], y_train[train_index]).predict(X_test)
            class_preds = LogisticRegression().fit(X_train[train_index], y_train[train_index]).predict(X_test)
        elif reg_name in ["SVM"]:
            reg_preds = SVR().fit(X_train[train_index], y_train[train_index]).predict(X_test)
            class_preds = SVC().fit(X_train[train_index], y_train[train_index]).predict(X_test)            
        elif reg_name in ["Knn"]:
            reg_preds = KNeighborsRegressor().fit(X_train[train_index], y_train[train_index]).predict(X_test)
            class_preds = KNeighborsClassifier().fit(X_train[train_index], y_train[train_index]).predict(X_test)
        else:
            reg_preds = RandomForestRegressor().fit(X_train[train_index], y_train[train_index]).predict(X_test)
            class_preds = RandomForestClassifier().fit(X_train[train_index], y_train[train_index]).predict(X_test)
        reg_resid = reg_preds - y_test
        fold_reg_residuals.append(reg_resid)
        fold_class_preds.append(class_preds)
        fold_df = pd.DataFrame()
        fold_df["MSqE"] = [mean_squared_error(y_test, reg_preds)]
        fold_df["MAE"] = [mean_absolute_error(y_test, reg_preds)]
        fold_df["MAPE"] = [np.mean(np.abs(reg_preds - y_test) / (y_test - 1e-5))]
        fold_df["R2"] = [r2_score(y_test, reg_preds)]
        fold_df["Accuracy"] = [accuracy_score(y_test, class_preds)]
        fold_dfs.append(fold_df)
    fold_gofs = pd.concat(fold_dfs, axis=0, ignore_index=True)
    return fold_gofs, fold_class_preds, y_test, fold_reg_residuals


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameter for passing dataset information")
    parser.add_argument("--dataset", choices=DATASET_NAMES, required=True, help="Enter the name of the dataset")
    args = parser.parse_args()

    if args.dataset == "Parkinsons":
        load_data = data_preprocess.parkinson_dataset
    elif args.dataset == "Liver":
        load_data = data_preprocess.liver_dataset
    elif args.dataset == "Diabetes":
        load_data = data_preprocess.diabetes_dataset
    else:
        raise RuntimeError("Unreachable!")
    dataset = load_data()

    REGRESSORS = {
        "Linear": LinearRegression(),
        "RF": RandomForestRegressor(),
        "Knn": KNeighborsRegressor(),
        "SVM": SVR(),
    }
    K = 5
    N_REPS = 150
    dfs = []
    for reg_name, regressor in REGRESSORS.items():
        rep_gofs = []
        for rep in range(N_REPS):
            fold_gofs, fold_class_preds, y_test, fold_reg_residuals = calculate_ECs(
                    dataset=dataset,
                    reg_name=reg_name,
                    k=K,
                )
            fold_gofs = fold_gofs.mean().to_frame().T
            fold_gofs[f"model"] = reg_name
            clssification_EC = error_consistencies(fold_class_preds, y_test, empty_unions = 1)
            fold_gofs[f"EC_Class"] = np.mean(clssification_EC[0])
            
            for method in ECMethod:
                fold_gofs[f"EC_{method}"] = np.mean(regression_ec(list(fold_reg_residuals), method))
            rep_gofs.append(fold_gofs)
        rep_df = pd.concat(rep_gofs, axis=0, ignore_index=True)
        dfs.append(rep_df)

    df = pd.concat(dfs, axis=0, ignore_index=True)
    filename = f"{args.dataset}_reg_error.csv"
    outfile = OUT / filename
    df.to_csv(outfile)
    print(f"Saved results for {args.dataset} error to {outfile}")
