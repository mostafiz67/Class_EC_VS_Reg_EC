# Collecting and saving description of the output file

import pandas as pd
from src.constants import OUT


# parkinson_output = OUT / "Diabetes_reg_error.csv",
# liver_output = OUT / "Liver_reg_error.csv",
# diabetes_output = OUT / "Parkinsons_reg_error.csv"


diabetes_output_df = pd.read_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Diabetes_reg_error.csv')
diabetes_output_df.groupby('model').describe().unstack(1).to_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Diabetes_reg_error_des.csv')

liver_output_df = pd.read_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Liver_reg_error.csv')
liver_output_df.groupby('model').describe().unstack(1).to_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Liver_reg_error_des.csv')

parkinson_output_df = pd.read_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Parkinsons_reg_error.csv')
parkinson_output_df.groupby('model').describe().unstack(1).to_csv('/home/mostafiz/Desktop/Final_Thesis/Regression-EC-Classification-EC/output/Parkinsons_reg_error_des.csv')