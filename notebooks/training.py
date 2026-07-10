# %% [markdown]
# # PROBLEM STATEMENT

# %% [markdown]
# Dream Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban and rural areas. Customers first apply for a home loan after that company validates the customer’s eligibility for a loan. The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling out the online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, and others. To automate this process, they have given a problem to identify the customer segments, that are eligible for loan amounts so that they can specifically target these customers.

# %% [markdown]
# # SOLUTION

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# %%
filepath = Path.cwd() /'..'/'data'/'train_loan.csv'
data = pd.read_csv(filepath)

# %%
df = data.copy()

# %%
df.head()

# %% [markdown]
# # DATA ANALYSIS

# %% [markdown]
# **Univariate Analysis**

# %% [markdown]
# Analysis of a single feature/column

# %%
df.dtypes

# %%
df.columns

# %%
num_cols = [column for column in df.columns if df[column].dtypes != 'O']

# %%
df['Gender'].dtypes

# %%
num_cols_chk = [column for column in df.columns if df[column].dtype != 'O']

# %%
num_cols_chk

# %%
num_cols

# %%
df[num_cols]

# %%
df['Dependents'].value_counts()

# %%
df['Dependents'].nunique()

# %%
df['Dependents'].unique()

# %%
df['Dependents'].replace('3+',3,inplace=True)

# %%
df['Dependents'].value_counts()

# %%
df.head()

# %%
df[num_cols]

# %%
df.shape

# %%
df['ApplicantIncome'].value_counts()

# %%
df['Loan_Amount_Term'].value_counts()

# %%
df['ApplicantIncome'].nunique()

# %%
df['Loan_Amount_Term'].nunique()

# %%
df['Credit_History'].nunique()

# %% [markdown]
# Univariate Analysis of Continuous Numerical Columns

# %%
df.head()

# %%
sns.distplot(df['ApplicantIncome'])

# %%
sns.boxplot(df['ApplicantIncome'])

# %% [markdown]
# We have seen there is extremen outliers in ApplicantIncome, so not suggested to use Mean for Missing Value Handling

# %%
sns.distplot(df['CoapplicantIncome'])

# %%
sns.boxplot(df['CoapplicantIncome'])

# %%
sns.boxplot(df['LoanAmount'])

# %% [markdown]
# Univariate Analysis of Discrete Numerical Columns

# %% [markdown]
# Cardinality check of the columns

# %%
sns.countplot(x='Dependents', data=df)

# %%
sns.countplot(x='Loan_Amount_Term', data=df)

# %%
sns.countplot(x='Credit_History', data=df)

# %% [markdown]
# Univariate Analysis of Categorical Columns

# %%
df.columns

# %%
cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']

for columns in cat_cols:
  plt.figure(figsize=(6,4))
  sns.countplot(x=columns, data=df)
  plt.title(columns)
  plt.show()

# %% [markdown]
# **Bi-Variate Analysis**

# %%
df.head()

# %%
sns.countplot(x='Gender', hue='Loan_Status', data=df)

# %%
sns.histplot(data=df, x='Gender', hue='Loan_Status', multiple='fill', stat='percent')

# %%
pd.crosstab(df['Gender'], df['Loan_Status'], normalize='index')*100

# %%
df['Gender'].value_counts()

# %%
pd.crosstab(df['Gender'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)

# %%
pd.crosstab(df['Married'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)

# %%
df.head()

# %%
df.groupby('Married')['CoapplicantIncome'].mean()

# %%
df.groupby('Married')['ApplicantIncome'].mean()

# %%
pd.crosstab(df['Education'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)

# %%
pd.crosstab(df['Education'], df['Loan_Status'], normalize='index')*100

# %%
pd.crosstab(df['Self_Employed'], df['Loan_Status'], normalize='index').plot(kind='bar', stacked=True)

# %%
pd.crosstab(df['Self_Employed'], df['Loan_Status'], normalize='index')*100

# %% [markdown]
# **Multi-Variate Analysis**

# %%
df.head()

# %%
sns.histplot(data=df[df['Married']=='Yes'], x='Gender',hue='Loan_Status', multiple='fill')

# %%
sns.histplot(data=df[df['Married']=='No'], x='Gender',hue='Loan_Status', multiple='fill')

# %%
sns.histplot(data=df[df['Self_Employed']=='Yes'], x='Education',hue='Loan_Status', multiple='fill')

# %%
sns.displot(data=df, x='Education',hue='Loan_Status', col='Self_Employed', multiple='fill')

# %% [markdown]
# what is observation?

# %%
df.head()

# %% [markdown]
# # FEATURE ENGINEERING

# %%
df.head()

# %%
df.isna().sum()

# %%
df['Dependents'].unique()

# %%
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# %%
df.isna().sum()

# %%
df.dtypes

# %%
df['Dependents'] = df['Dependents'].astype('int')

# %%
df.dtypes

# %%
df.isna().sum()

# %%
df[df['Gender'].isna()]

# %%
pd.crosstab(df['Dependents'],[df['Married'],df['Gender']],normalize="index")*100

# %%
mode_gender_3D_Y = df.loc[(df['Married']=="Yes")&(df['Dependents']==3),"Gender"].mode()[0]

# %%
mode_gender_3D_Y

# %%
df.loc[(df['Gender'].isnull())&(df['Married']=='Yes')&(df['Dependents']==3),"Gender"] = mode_gender_3D_Y

# %%
df[df['Gender'].isna()]

# %%
mode_gender_2D_Y = df.loc[(df['Married']=="Yes")&(df['Dependents']==2),"Gender"].mode()[0]
df.loc[(df['Gender'].isnull())&(df['Married']=='Yes')&(df['Dependents']==2),"Gender"] = mode_gender_2D_Y

mode_gender_1D_Y = df.loc[(df['Married']=="Yes")&(df['Dependents']==1),"Gender"].mode()[0]
df.loc[(df['Gender'].isnull())&(df['Married']=='Yes')&(df['Dependents']==1),"Gender"] = mode_gender_1D_Y

mode_gender_0D_Y = df.loc[(df['Married']=="Yes")&(df['Dependents']==0),"Gender"].mode()[0]
df.loc[(df['Gender'].isnull())&(df['Married']=='Yes')&(df['Dependents']==0),"Gender"] = mode_gender_0D_Y

# %%
df.isna().sum()

# %%
df[df['Gender'].isna()]

# %%
mode_gender_3D_N = df.loc[(df['Married']=="No")&(df['Dependents']==3),"Gender"].mode()[0]
df.loc[(df['Gender'].isnull())&(df['Married']=='No')&(df['Dependents']==3),"Gender"] = mode_gender_3D_N

mode_gender_0D_N = df.loc[(df['Married']=="No")&(df['Dependents']==0),"Gender"].mode()[0]
df.loc[(df['Gender'].isnull())&(df['Married']=='No')&(df['Dependents']==0),"Gender"] = mode_gender_0D_N

# %%
df.isna().sum()

# %%
df[df['Married'].isna()]

# %%
df.loc[104,'Married'] = 'Yes'
df.loc[228,'Married'] = 'No'
df.loc[435,'Married'] = 'No'

# %%
df.isna().sum()

# %%
df[df['Self_Employed'].isna()]

# %%
high_income = df['ApplicantIncome'].median()

# %%
high_income

# %%
df.loc[(df['Self_Employed'].isnull())&(df['ApplicantIncome']>high_income),"Self_Employed"]="Yes"

# %%
df.isna().sum()

# %%
df['Self_Employed'] = df['Self_Employed'].fillna("No")

# %%
df.isna().sum()

# %%
df.head(2)

# %%
df[df['LoanAmount'].isna()]

# %%
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# %%
df.head()

# %%
df["IncomeBand"], income_bins = pd.qcut(df['TotalIncome'], 4, labels=False,retbins=True)

# %%
print(income_bins)

# %%
df.head()

# %%
df['LoanAmount'] = df['LoanAmount'].fillna(df.groupby("IncomeBand")["LoanAmount"].transform("median"))

# %%
df.isna().sum()

# %%
df.head(2)

# %%
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df.groupby("IncomeBand")["Loan_Amount_Term"].transform(lambda x:x.mode()[0]))

# %%
df.isna().sum()

# %%
df.head()

# %%
df['Credit_History'].unique()

# %%
df['Credit_History'] = df['Credit_History'].fillna(2)

# %%
df['Credit_History'].unique()

# %%
df.isna().sum()

# %%
df.head()

# %%
df['LoanIncomeRatio'] = df['LoanAmount']/df['TotalIncome']

# %%
df.head()

# %%
df['HasCoapplicant'] = np.where(df['CoapplicantIncome'] > 0,1,0)

# %%
df.head()

# %%
sns.distplot(df['ApplicantIncome'])

# %%
df['ApplicantIncome'] = np.log1p(df['ApplicantIncome'])

# %%
sns.distplot(df['ApplicantIncome'])

# %%
sns.distplot(df['CoapplicantIncome'])

# %%
df['CoapplicantIncome'] = np.log1p(df['CoapplicantIncome'])

# %%
sns.distplot(df['CoapplicantIncome'])

# %%
sns.distplot(df['LoanAmount'])

# %%
df['LoanAmount'] = np.log1p(df['LoanAmount'])

# %%
sns.distplot(df['LoanAmount'])

# %%
df.head()

# %%
df.drop('Loan_ID',axis=1,inplace=True)

# %%
df.head()

# %%
df.dtypes

# %%
cat_cols = [column for column in df.columns if df[column].dtypes == 'O']

# %%
cat_cols

# %%
target_col = "Loan_Status"
if target_col in cat_cols:
    cat_cols.remove(target_col)

# %%
df[cat_cols]

# %%
from sklearn.preprocessing import OneHotEncoder

# %%
ohe = OneHotEncoder(drop="first", sparse_output=False)

# %%
encoded_data = ohe.fit_transform(df[cat_cols])

# %%
encoded_data

# %%
encoded_df = pd.DataFrame(encoded_data,columns = ohe.get_feature_names_out(cat_cols), index=df.index)

# %%
encoded_df

# %%
df.head(1)

# %%
df = pd.concat([df,encoded_df],axis=1)

# %%
df.head(1)

# %%
df.drop(columns=cat_cols,axis=1,inplace=True)

# %%
df.head(10)

# %%
from sklearn.preprocessing import StandardScaler

# %%
scaling_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','TotalIncome','LoanIncomeRatio']

# %%
scaler = StandardScaler()

# %%
df[scaling_cols] = scaler.fit_transform(df[scaling_cols])

# %%
df.head()

# %% [markdown]
# # Feature Selection

# %%
df.head()

# %%
X = df.drop("Loan_Status",axis=1)
y =df["Loan_Status"]

# %%
X

# %%
y

# %%
# from sklearn.feature_selection import SelectKBest, mutual_info_classif

# selector = SelectKBest(mutual_info_classif,k=8)

# %%
# X_new = selector.fit_transform(X,y)

# %%
# X.columns[selector.get_support()]

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, stratify=y)

# %%
X_train.columns

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# %%
rf = RandomForestClassifier(random_state=42)

# %%
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict(X_test)

# %%
accuracy_score(y_test,y_pred)

# %%
confusion_matrix(y_test,y_pred)

# %%
print(classification_report(y_test,y_pred))

# %%
from sklearn.model_selection import RandomizedSearchCV

param_dist= {
    "n_estimators": [100,200,300,500],
    "max_depth" : [None, 5,10,15,20],
    "min_samples_split": [2,5,10],
    "min_samples_leaf" : [1,2,4],
    "max_features" : ["sqrt","log2"],
    "bootstrap" : [True, False]
}

# %%
random_search = RandomizedSearchCV(
    estimator = RandomForestClassifier(random_state=42),
    param_distributions= param_dist,
    n_iter=30,
    scoring="accuracy",
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=1
)

# %%
random_search.fit(X_train,y_train)

# %%
random_search.best_params_

# %%
random_search.best_score_

# %%
random_search.best_estimator_

# %%
best_rf_model = random_search.best_estimator_

# %%
y_pred_tuned = best_rf_model.predict(X_test)

# %%
accuracy_score(y_test, y_pred_tuned)

# %%
import joblib

# %%
# Path.cwd() /'..'/'data'/'train_loan.csv'
model_path = Path.cwd() /'..'/'utils'/'loan_rf_model.pkl'
joblib.dump(best_rf_model, model_path)

# %%
scaler_path =  Path.cwd() /'..'/'utils'/'scaler.pkl'
joblib.dump(scaler, scaler_path)

# %%
OHE_path = Path.cwd() /'..'/'utils'/'onehot_encoder.pkl'
joblib.dump(ohe, OHE_path)

# %%
model_cols_path = Path.cwd() /'..'/'utils'/'model_columns.pkl'
scaling_path = Path.cwd() /'..'/'utils'/'scaling_cols.pkl'

joblib.dump(X.columns.tolist(), model_cols_path)
joblib.dump(scaling_cols, scaling_path)

# %%