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

#imputing salary by mean of average position and company salary
df_joined = pd.merge(df_joined, df_jobdesc_avgcompany,  how='left', left_on=['company'], right_on = ['company'])
df_joined = pd.merge(df_joined, df_jobdesc_avgsalary,  how='left', left_on=['job_title_full'], right_on = ['job_title_full'])
df_joined['job_avg'].fillna(df_joined['company_avg'], inplace=True) # if no job average is available consider only company average
df_joined['avg']= (df_joined['company_avg']+df_joined['job_avg'])/2 # calculate average of company and job salary
df_joined['salary'].fillna(df_joined['avg'], inplace=True)



