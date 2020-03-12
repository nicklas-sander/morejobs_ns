#import statements

import numpy as np
import pandas as pd
import sklearn
import warnings

warnings.filterwarnings("ignore")

#reading csv files into dataframes

df_users = pd.read_csv("C:/Users/GGLQY/Python/Decision Science/applicant_material/applicant_material/user.csv")

df_jobdesc = pd.read_csv("C:/Users/GGLQY/Python/Decision Science/applicant_material/applicant_material/job_desc.csv")

df_users.fillna(0, inplace=True)

df_joined = df_jobdesc.merge(df_users,left_on="user_id", right_on="user_id", how='left')

