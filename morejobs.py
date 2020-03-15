#import statements

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
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

#scaling all input features including salary using StandardScaler
df_joined_final = df_joined.drop(columns=['company_avg','job_avg','avg'])
df_joined_feature = df_joined_final.drop(columns=['job_title_full','user_id','company','has_applied'])
df_joined_label = pd.DataFrame(df_joined_final['has_applied'])
# using standard scaler
scaler = StandardScaler()
scaler.fit(df_joined_feature)
scaled = scaler.transform(df_joined_feature)
df_joined_feature_scaled = pd.DataFrame(scaled, columns=df_joined_feature.columns, index=df_joined_feature.index)
# rejoining the scaled values with the label
df_joined_scaled = df_joined_label.join(df_joined_feature_scaled)

df_corr = df_joined_scaled.corr(method='pearson')
df_corr = df_corr.loc['has_applied']
df_corr.plot()
plt.show()

#feature selection after correlation check v25 and v30

df_selected = df_joined_scaled[['v25','v30','has_applied']]

x_train, x_test, y_train, y_test = train_test_split(df_joined_scaled[['v25','v30']], df_joined_scaled['has_applied'], test_size=0.3)

x_train_df = pd.DataFrame(x_train, columns={"v25", "v30"})
y_train_df = pd.DataFrame(y_train, columns={"has_applied"})

train_df = x_train_df.join(y_train_df)

plt.figure()
plt.scatter(train_df[train_df["has_applied"]==1]["v25"], train_df[train_df["has_applied"]==1]["v30"], marker='x', color="green")
plt.scatter(train_df[train_df["has_applied"]==0]["v25"], train_df[train_df["has_applied"]==0]["v30"], marker='o', color="red")
plt.legend(['has applied', 'has not applied'],bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title("Training Data in scatterplot")

plt.show()

# support vector machine for classification of binary label

# training the model based on the training data split
clf = svm.SVC()
clf.fit(x_train, y_train)

#predicting the training data to evaluate training performance
pred = clf.predict(x_train)
data = {"actual":y_train, "predicted":pred}
predictions_df = pd.DataFrame(data)
predictions_df["dif"] = abs(predictions_df["actual"]-predictions_df["predicted"])
print ("Training accuracy of Support Vector Machine is: {} percent".format(round((1-(sum(predictions_df["dif"])/len(predictions_df.index)))*100,1)))

#predicting the test data to evaluate test performance
pred = clf.predict(x_test)
data = {"actual":y_test, "predicted":pred}
predictions_df = pd.DataFrame(data)
predictions_df["dif"] = abs(predictions_df["actual"]-predictions_df["predicted"])
print ("Test accuracy of Support Vector Machine is: {} percent".format(round((1-(sum(predictions_df["dif"])/len(predictions_df.index)))*100,1)))

