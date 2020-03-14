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

df_joined = df_jobdesc.merge(df_users, left_on="user_id", right_on="user_id", how='left')

#removing leading and trailing spaces in job title
df_joined['job_title_full'] = df_joined['job_title_full'].str.strip()

#calculating the average salary by position and company for imputation of missing salaries
df_jobdesc_avgsalary = df_jobdesc.groupby(['job_title_full']).mean()
df_jobdesc_avgsalary.rename(columns={"salary":"job_avg"}, inplace=True)
df_jobdesc_avgcompany = df_jobdesc.groupby(['company']).mean()
df_jobdesc_avgcompany.rename(columns={"salary":"company_avg"}, inplace=True)
