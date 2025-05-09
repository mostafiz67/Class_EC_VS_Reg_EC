"""
Some code is inherited from https://stackoverflow.com/questions/68397783/pythonic-way-to-generate-seaborn-heatmap-subplots
"""

# python3 -m  src.corr_plots
from pathlib import Path
import pandas as pd
from pandas.core.frame import DataFrame
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from typing import List
import math

from src.constants import OUT, PLOT_CORR_REG_CLASS_OUTPUT_PATH

ROOT = Path(__file__).resolve().parent
CSVS = [
    OUT / "Diabetes_reg_error.csv",
    OUT / "Liver_reg_error.csv",
    OUT / "Parkinsons_reg_error.csv"
    
]

ECMethods = ["EC_Class", "EC_ratio", "EC_ratio-diff", "EC_ratio-signed", "EC_ratio-diff-signed",
            "EC_intersection_union_sample", "EC_intersection_union_all", "EC_intersection_union_distance"]
REGRESSORS = ["Linera", "SVM", "RF", "Knn"]

def correlation_analysis_GoF_vs_EC() -> DataFrame:
        corrs = []
        for csv in CSVS:
            error_kind = csv.stem[: csv.stem.find("_")]
            main_df = pd.read_csv(csv).drop(columns=[ "MSqE", "MAE"])
            final_df = pd.melt(main_df, id_vars=['Unnamed: 0', "model","MAPE", "R2", "Accuracy"], value_vars=['EC_Class', 'EC_ratio', 'EC_ratio-diff', "EC_ratio-signed", "EC_ratio-diff-signed",
                                                                                                "EC_intersection_union_sample", "EC_intersection_union_all", "EC_intersection_union_distance"])

            final_df = final_df.drop(columns=["Unnamed: 0"])
            final_df.rename({"model": "Model", "variable": "EC_Method", "value": "EC"}, axis=1, inplace=True)
            for i in range(len(ECMethods)):

                df = final_df.loc[final_df['EC_Method'] == ECMethods[i]]
                # Method one using FacetGrid
                g = sns.FacetGrid(df, col="Model", col_wrap=2, height=4)
                g.map_dataframe(lambda data, color:sns.heatmap(data.corr(), annot=True, fmt='.2f', square=True))
                g.fig.subplots_adjust(top=0.9)
                g.fig.suptitle(f'Correlation Among GoF and EC (EC Method: {ECMethods[i]}, Dataset: {error_kind})')
                
                fig_name = "{0}'_method_'{1}'.png".format(error_kind, ECMethods[i])
                outfile = os.path.join(PLOT_CORR_REG_CLASS_OUTPUT_PATH, 'GoF_vs_EC/') + fig_name
                # plt.show()
                plt.savefig(outfile, format='png')
                plt.close()

        # corr_df = pd.concat(corrs, axis=1)
        # return corr_df

def correlation_analysis_EC_Methods() -> DataFrame:
        corrs = []
        for csv in CSVS:
            error_kind = csv.stem[: csv.stem.find("_")]
            main_df = pd.read_csv(csv).drop(columns=['Unnamed: 0', "MSqE", "MAE", "MAPE"])
            main_df.rename({"model": "Model"}, axis=1, inplace=True)
            g = sns.FacetGrid(main_df, col="Model", col_wrap=2, height=6)
            g.map_dataframe(lambda data, color:sns.heatmap(data.corr(), annot=True, fmt='.2f', square=True))
            g.fig.subplots_adjust(top=0.9)
            g.fig.suptitle(f'Correlation Among EC Methods (Dataset: {error_kind})', fontsize=20)
            
            fig_name = "{0}'_corr_ECs.png".format(error_kind)
            outfile = os.path.join(PLOT_CORR_REG_CLASS_OUTPUT_PATH, 'GoF_vs_EC/') + fig_name
            # plt.show()
            plt.savefig(outfile, format='png')
            plt.close()

        # # corr_df = pd.concat(corrs, axis=1)
        # # return corr_df



if __name__ == "__main__":
    # corr_GoF_vs_EC_df = correlation_analysis_GoF_vs_EC()

    corr_EC_Methods_df = correlation_analysis_EC_Methods()
    