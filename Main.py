#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option("display.max_columns", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 1000)

# sklearn preprocessing for dealing with categorical variables
from sklearn.preprocessing import LabelEncoder

# File systems management
import os


# In[2]:


app_train = pd.read_csv('application_train.csv')
app_test = pd.read_csv('application_test.csv')
previous_application = pd.read_csv('previous_application.csv')


# In[3]:


app_train.head()


# In[4]:


app_train.shape


# In[5]:


app_train.columns


# In[6]:


#Checking the data type, if object have to confert it
app_train.info(verbose=True)


# In[7]:


#Check Mean, Median
app_train.describe()


# In[8]:


previous_application.head(2)


# In[9]:


previous_application.tail(2)


# In[10]:


previous_application.shape


# In[11]:


previous_application.columns


# In[12]:


#There are null valuse in the data
previous_application.info(verbose=True)


# In[13]:


previous_application.describe()


# In[14]:


app_train.isnull().sum().sort_values(ascending=False)


# In[15]:


round(app_train.isnull().sum()/app_train.shape[0]*100,2).sort_values(ascending=False)


# In[16]:


#Drop table with lots of null value more than 45%
app_train=app_train.loc[:, app_train.isnull().mean()<=0.45]


# In[17]:


app_test=app_test.loc[:, app_test.isnull().mean()<=0.45]


# In[18]:


app_train.shape


# In[19]:


app_test.shape


# In[20]:


app_train.columns


# In[21]:


app_test.columns


# In[22]:


round(app_train.isnull().sum()/app_train.shape[0]*100,2).sort_values(ascending=False)


# In[23]:


app_train['OCCUPATION_TYPE'].value_counts()


# In[24]:


app_train['OCCUPATION_TYPE'].isnull().sum()


# In[25]:


#Replace empty = 'Unknown'
app_train['OCCUPATION_TYPE'].replace(np.NaN, 'Unknown', inplace = True)


# In[26]:


app_train['OCCUPATION_TYPE'].isnull().sum()


# In[27]:


app_train['OCCUPATION_TYPE'].value_counts()


# In[28]:


#Checking the next highest null 
round(app_test.isnull().sum()/app_test.shape[0]*100,2).sort_values(ascending=False)


# In[29]:


app_test['OCCUPATION_TYPE'].value_counts()


# In[30]:


app_test['OCCUPATION_TYPE'].isnull().sum()


# In[31]:


#Replace empty = 'Unknown'
app_test['OCCUPATION_TYPE'].replace(np.NaN, 'Unknown', inplace = True)


# In[32]:


app_train['OCCUPATION_TYPE'].isnull().sum()


# In[33]:


# Droping the columns not required for Analysis:

NOT_REQ = ['FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_17',
          'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_12',
          'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_7', 
          'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_2', 
          'OBS_30_CNT_SOCIAL_CIRCLE' , 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
          'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_WEEK',
          'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_QRT']


# In[34]:


app_train.drop( labels = NOT_REQ , axis = 1, inplace = True)


# In[35]:


app_train.shape


# In[36]:


# Droping the columns not required for Analysis:

NOT_REQ_TEST = ['FLAG_DOCUMENT_21', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_17',
          'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_12',
          'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_7', 
          'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_2', 
          'OBS_30_CNT_SOCIAL_CIRCLE' , 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
          'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_WEEK',
          'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_QRT', 'EXT_SOURCE_1']


# In[37]:


app_test.drop( labels = NOT_REQ_TEST , axis = 1, inplace = True)


# In[38]:


app_test.shape


# In[39]:


round(app_train.isnull().sum()/app_train.shape[0]*100,2).sort_values(ascending=False)


# In[40]:


#VALUE COUNTS IN GRAPH
plt.figure(figsize=[20,5])

sns.barplot(x=app_train.OCCUPATION_TYPE.value_counts().index,
            y=app_train.OCCUPATION_TYPE.value_counts().values).set_title("OCCUPATION TYPE COUNTS",
                                                                        fontsize=30, color='Green', pad=20)
plt.xlabel('OCCUPATION TYPE', fontsize= 20, color='Brown')
plt.xticks(rotation=45)

plt.show


# In[41]:


plt.figure(figsize=[20,15])

sns.set_style('darkgrid')

plt.subplot(2,2,1)
sns.boxplot(app_train['EXT_SOURCE_2']).set_title('EXT_SOURCE_2', fontsize=20, color='Green', pad=20)

plt.subplot(2,2,2)
sns.boxplot(app_train['EXT_SOURCE_3']).set_title('EXT_SOURCE_3', fontsize=20, color='Green', pad=20)

plt.show()


# In[42]:


plt.figure(figsize=[20,15])

sns.set_style('darkgrid')

plt.subplot(2,2,1)
sns.displot(app_train['EXT_SOURCE_2'], color='g')


# In[43]:


plt.figure(figsize=[20,15])

sns.set_style('darkgrid')

plt.subplot(2,2,1)
sns.displot(app_train['EXT_SOURCE_3'], color='g')


# From the above graph, we can conclude that: there are no outliers There is a small amount of skewness

# Median can be used to replace the missing values here because of skewness

# In[44]:


# Replacing missing values of these 2 columns with its correspending Median

for column in ['EXT_SOURCE_2', 'EXT_SOURCE_3']:
    app_train[column].fillna(app_train[column].median(), inplace=True)


# In[45]:


# Replacing missing values of these 2 columns with its correspending Median

for column in ['EXT_SOURCE_2', 'EXT_SOURCE_3']:
    app_test[column].fillna(app_test[column].median(), inplace=True)


# ### Checking and Imputing AMT_GOODS_PRICE column

# In[46]:


# Checking the correlation between the loan amount demanded vs the good's price

sns.scatterplot(x=app_train['AMT_CREDIT'], y=app_train['AMT_GOODS_PRICE'], data = app_train)

plt.title("Correlation between the Loan amount and the price of goods for which loan was given\n",
         fontdict={'fontsize': 25, 'fontweight' : 5, 'color':'Brown'})
plt.xlabel("Loan Amount", fontdict={'fontsize': 20, 'fontweight' : 5, 'color':'Brown'})
plt.ylabel("Loan Amount", fontdict={'fontsize': 20, 'fontweight' : 5, 'color':'Brown'})

plt.show()


# Inference
# 
# Since there is a very linear and positive correlation between the Loan Amount and the Good's price, we can assume that, in most cases the loan amount demanded by the customer is slightly more than but mostly equal to the price of the article he/she wishes to purchase. For the AMT_GOODS_PRICE we can impute the same value of AMT_CREDIT for missing values (keeping in mind loan amount is usually same as good's price)

# In[47]:


# Imputing the above mentioned logic

app_train['AMT_GOODS_PRICE'] = np.where(app_train['AMT_GOODS_PRICE'].isnull() == True,
                                       app_train['AMT_CREDIT'], app_train['AMT_GOODS_PRICE'])


# In[48]:


app_train['AMT_GOODS_PRICE'].isnull().sum()


# In[49]:


import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff


# In[50]:


# Percentage of each category
go.Figure(data=[go.Pie(labels=app_train.NAME_TYPE_SUITE.value_counts().index,
                       values=app_train.NAME_TYPE_SUITE.value_counts().values, hole=.6, title = 'NAME_TYPE_SUITE VALUE COUNTS',
                       pull=[0,0.1,0.1,0.1,0.1,0.1,0.1])] )


# In[51]:


app_train['NAME_TYPE_SUITE'].isnull().sum()/app_train.shape[0]*100


# In[52]:


app_train['NAME_TYPE_SUITE'].isnull().sum()


# In[53]:


app_train['NAME_TYPE_SUITE'].mode()


# In[54]:


#Replacing missing values with MODE
app_train['NAME_TYPE_SUITE'].fillna(app_train['NAME_TYPE_SUITE'].mode()[0], inplace = True)


# In[55]:


round(app_train.isnull().sum()/app_train.shape[0]*100,4).sort_values(ascending=False)


# In[56]:


# REMAINING COLUMNS with negligible null values (LESS THAN 1)

NULL_COL = ['CNT_FAM_MEMBERS', 'AMT_ANNUITY', 'DAYS_LAST_PHONE_CHANGE']

for column in NULL_COL:
    app_train[column].fillna(app_train[column].median(), inplace=True)


# In[57]:


round(app_train.isnull().sum()/app_train.shape[0]*100,4).sort_values(ascending=False)


# In[58]:


app_train.info(verbose=True)


# CHANGING DAYS COLUMNS AND COUNT COLUMNS TO INTEGER

# In[59]:


dayandcount = ['CNT_FAM_MEMBERS' , 'DAYS_REGISTRATION' , 'DAYS_LAST_PHONE_CHANGE']

app_train.loc[:,dayandcount]=app_train.loc[:,dayandcount].apply(lambda x: x.astype('int64', errors='ignore'))


# In[60]:


app_train.info()


# #### CHANGINE ALL VALUES OF COLUMNS WITH DTYPE OBJECT TO STRING

# In[61]:


#LISTING OBJECT TYPE COLUMNS AND CONFIRMING THE VALUES TO BE IN STRING TYPE

obj_col = list(app_train.select_dtypes(include='object').columns)

app_train.loc[:,obj_col]=app_train.loc[:,obj_col].apply(lambda x: x.astype('str'))


# In[62]:


app_train.info()


# Checking values of other categorical columns
# Checking Gender Code Column

# In[63]:


# VALUE COUNTS OF GENDER CODE

app_train.CODE_GENDER.value_counts()


# In[64]:


# Checking the Gender Column

plt.figure(figsize=[10,5])

sns.barplot(x=app_train.CODE_GENDER, y=app_train.TARGET).set_title("Gender vs Target", fontsize=20, color='Green', pad=20)

plt.show()


# In[65]:


round(app_train.ORGANIZATION_TYPE.value_counts()/app_train.shape[0]*100,2)


# We notice that there are several sub-categories within Industry, Trade, Business, and Transport 

# In[66]:


# Therefore, we eliminate the sub_category with the overall category

app_train.ORGANIZATION_TYPE = app_train.ORGANIZATION_TYPE.apply(lambda x: 'Industry' if 'Industry' in x else x)
app_train.ORGANIZATION_TYPE = app_train.ORGANIZATION_TYPE.apply(lambda x: 'Trade' if 'Trade' in x else x)
app_train.ORGANIZATION_TYPE = app_train.ORGANIZATION_TYPE.apply(lambda x: 'Transport' if 'Transport' in x else x)
app_train.ORGANIZATION_TYPE = app_train.ORGANIZATION_TYPE.apply(lambda x: 'Business' if 'Business' in x else x)


# In[67]:


round(app_train.ORGANIZATION_TYPE.value_counts()/app_train.shape[0]*100,2)


# In[68]:


# Therefore, we eliminate the sub_category with the overall category

app_test.ORGANIZATION_TYPE = app_test.ORGANIZATION_TYPE.apply(lambda x: 'Industry' if 'Industry' in x else x)
app_test.ORGANIZATION_TYPE = app_test.ORGANIZATION_TYPE.apply(lambda x: 'Trade' if 'Trade' in x else x)
app_test.ORGANIZATION_TYPE = app_test.ORGANIZATION_TYPE.apply(lambda x: 'Transport' if 'Transport' in x else x)
app_test.ORGANIZATION_TYPE = app_test.ORGANIZATION_TYPE.apply(lambda x: 'Business' if 'Business' in x else x)


# In[69]:


round(app_test.ORGANIZATION_TYPE.value_counts()/app_train.shape[0]*100,2)


# In[70]:


# Checking the ORGANIZATION_TYPE Column

plt.figure(figsize=[20,5])

sns.barplot(x=app_train.ORGANIZATION_TYPE.value_counts().index,
            y=app_train.ORGANIZATION_TYPE.value_counts().values).set_title("Distribution within ORGANIZATION_TYPE",
                                                                        fontsize=20, color='Green', pad=20)
plt.xlabel('ORGANIZATION_TYPE', fontsize=20, color='Brown')

plt.xticks(rotation=90)
plt.show()


# INFERENCE:
# 
# People who is in business field applied more in number for the loan compared other fields.

# In[71]:


### Checking NAME_CONTRACT_TYPE column
app_train.NAME_CONTRACT_TYPE.value_counts()


# In[72]:


### CHECKING FLAG OWN CAR COLUMN
app_train.FLAG_OWN_CAR.value_counts()


# In[73]:


### CHECKING FLAG OWN REALTY COLUMN
app_train.FLAG_OWN_REALTY.value_counts()


# In[74]:


### CHECKING NAME TYPE SUITE COLUMN
app_train.NAME_TYPE_SUITE.value_counts()


# In[75]:


### CHECKING NAME EDUCATION TYPE COLUMN
app_train.NAME_EDUCATION_TYPE.value_counts()


# In[76]:


### CHECKING NAME INCOME TYPE COLUMN
app_train.NAME_INCOME_TYPE.value_counts()


# In[77]:


### CHECKING NAME FAMILY STATUS COLUMN
app_train.NAME_FAMILY_STATUS.value_counts()


# In[78]:


### CHECKING NAME HOUSING TYPE COLUMN
app_train.NAME_HOUSING_TYPE.value_counts()


# In[79]:


### CHECKING WEEKDAY_APPR_PROCESS COLUMN
app_train.WEEKDAY_APPR_PROCESS_START.value_counts()


# ### Analyzing Effect of Age on Repayment

# In[80]:


(app_train['DAYS_BIRTH'] / -365).describe()


# These age looks reasoneable. There are no outliers for the age on either high or low end. How about the days of employement

# In[81]:


(app_train['DAYS_EMPLOYED']/-365).describe()


# That doesn't look right! The maximum value (beside being positive) is about 1000 years!

# In[82]:


app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram');
plt.xlabel('Days Employment');


# In[83]:


#replace anomalies
#create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
#replace the anamalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days_Employment_Histogram');
plt.xlabel('Days Employment');


# In[84]:


app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)


# In[85]:


#  Find the correlation of the positive days since birth and target
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_train['DAYS_BIRTH'].corr(app_train['TARGET'])


# As the client get older, there is a negative linear relationship with the target meaning that as clients get older, they tend to repay their loans on time more often.
# 
#     Let's start looking at this variable. First, we make a histogram of the age. We will put the x axis in years to make the plot a little more understable

# In[86]:


#set the style of plots
plt.style.use('fivethirtyeight')

# Plot the distribution of age in years
plt.hist(app_train['DAYS_BIRTH'] / 365, edgecolor = 'k', bins =25)
plt.title('Age of Client'); plt.xlabel('Age (years)'); plt.ylabel('Count');


# "The distribution of age alone doesn't reveal much, other than the absence of outliers. To see how age affects the target variable, we'll create a kernel density estimation (KDE) plot colored by the target values. A KDE plot visualizes the distribution of a single variable, like a smoothed histogram. It's created by calculating a kernel, typically Gaussian, at each data point and then averaging them to form a smooth curve. We'll use seaborn's kdeplot function for this."

# In[87]:


plt.figure(figsize = (10,8))

# KDE plot of loans that were repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'] / 365, label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'] / 365, label = 'target == 1')
            
# Labelling of plot
plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');


# "The target == 1 curve skews towards the younger end of the range. Although this is not a significant correlation (-0.07 correlation coefficient), this variable is likely going to be useful in a machine learning model because it does affect the target. Let's look at this relationship in another way: average failure to repay loans by age bracket."

# #### Average Failure to Repay Loans by Age Bracket

# To make this graph, first we cut the age category into bins of 5 years each. Then, for each bin, we calculate the average value of the target, which tells us the ratio of loans that were not repaid in each age category

# In[88]:


# Age information into a seperate dataframe
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
age_data['YEARS_BIRTH'] = age_data['DAYS_BIRTH'] / 365

# Bin the age data
age_data['YEARS_BINNED'] = pd.cut(age_data['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
age_data.head(10)


# In[89]:


# Group by the bin and calculate averages
age_groups = age_data.groupby('YEARS_BINNED').mean()
age_groups


# In[90]:


plt.figure(figsize = (8,8))

# Graph the age bins and the average of the target as a bar plot
plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('Age Group (years)'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by Age Group')


# There is a clear trend: younger applicants are more likely to not repay the loan! The rate of failure to repay is above 10% for the youngest three age groups and below 5% for the oldest age group
# 
# This is information that could be directly used by the bank: because younger clients are less likely to repay the loan, maybe they should be provided with more guidance or financial planning tips. This does not mean the bank should discriminate against younger clients, but it would be smart to take the precautionary measures to help younger clients pay on time

# # Analyzing prev_credit

# In[91]:


previous_application.head()


# In[92]:


previous_application.shape


# In[93]:


previous_application.isnull().sum()


# In[94]:


# Keeping only the necessary columns for merge and analysis:

cols_n = ['SK_ID_CURR', 'AMT_APPLICATION', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
         'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE','CHANNEL_TYPE',
         'NAME_YIELD_GROUP']

previous_application=previous_application.loc[:,cols_n]


# In[95]:


previous_application.shape


# In[96]:


# checking for any missing values

previous_application.isnull().sum()


# In[97]:


previous_application.DAYS_DECISION.unique()


# In[98]:


previous_application['DAYS_DECISION'] = abs(previous_application['DAYS_DECISION'])


# In[99]:


previous_application.NAME_PAYMENT_TYPE.value_counts()


# In[100]:


previous_application.NAME_PORTFOLIO.value_counts()


# In[101]:


# CHANGING XNA to Unknown

XNA_col = ['NAME_PAYMENT_TYPE', 'NAME_CLIENT_TYPE', 'NAME_PORTFOLIO']

for i in XNA_col:
    previous_application[i] = previous_application[i].str.replace('XNA','Unknown')


# ### Univariate analysis on the previous_application columns

# In[102]:


#plotting graph for AMT_APPLICATION

plt.figure(figsize=[20,6])

plt.subplot(1,2,1)
sns.boxplot(previous_application['AMT_APPLICATION']).set_title("AMT_APPLICATION - BOXPLOT", fontsize =20, color='indigo',pad=20)

plt.subplot(1,2,2)
sns.distplot(previous_application['AMT_APPLICATION'], color='g').set_title("AMT_APPLICATION - DISTRIBUTION", fontsize = 20, color='indigo', pad=20)

plt.show()


# In[103]:


previous_application['AMT_APPLICATION'].max()


# Inference:
# 
# From the box plot and the histogram we can see that most of the clients have asked for credit worth less than 7 million. Most of credit being near about 1-3 million

# ### Mergin both new and old dataframes

# In[104]:


new_df = pd.merge( left=app_train, right=previous_application, how ='inner', on ='SK_ID_CURR')


# In[105]:


new_df_test = pd.merge( left=app_test, right=previous_application, how ='inner', on ='SK_ID_CURR')


# In[106]:


new_df.info()


# In[107]:


#% of Loan Payment Difficulties for NAME_CONTRACT_STATUS and NAME_CLIENT_TYPE

table = pd.pivot_table(new_df, values='TARGET', index=['NAME_CLIENT_TYPE'],
                      columns=['NAME_CONTRACT_STATUS'], aggfunc=np.mean)

cm = sns.light_palette("green", as_cmap=True)
table.style.background_gradient(cmap=cm)


# In[108]:


table.T.plot(kind='bar').set_ylabel('% of Loan_Payment Difficulties')

plt.title('% of Loan Payment Difficulties for NAME_CONTRACT_STATUS and NAME_CLIENT_TYPE', fontdict={'fontsize':18}, pad=20)

plt.show()


# Inference -
# 
# From the above data we can infer that new clients are more likely to cancel loans. Also new clients are more likely to get their loan amount refused. Repeater clients are more likely to get a loan refused.

# In[109]:


#% of Loan Payment Difficulties for NAME_CONTRACT_STATUS and NAME_CONTRACT_TYPE

table = pd.pivot_table(new_df, values='TARGET', index=['NAME_CONTRACT_TYPE'],
                      columns=['NAME_CONTRACT_STATUS'], aggfunc=np.mean)

cm = sns.light_palette("green", as_cmap=True)
table.style.background_gradient(cmap=cm)


# In[110]:


table.T.plot(kind='bar').set_ylabel('% of Loan_Payment Difficulties')

plt.title('% of Loan Payment Difficulties for NAME_CONTRACT_STATUS and NAME_CONTRACT_TYPE', fontdict={'fontsize':18}, pad=20)

plt.show()


# Inference -
# 
# Cash Loans are more likely to get cancelled or refused with a bigger margin of that revolving loans.

# In[111]:


# Create Label Encoder object
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in new_df:
    if new_df[col].dtype == 'object':
        #if 2 or fewer unique categories
        if len(list(new_df[col].unique())) <= 2:
            # Train on the training data
            le.fit(new_df[col])
            # Transform both training and testing data
            new_df[col] = le.transform(new_df[col])
            new_df_test[col] = le.transform(new_df_test[col])
            
            # Keep track of how many columns were label encoded
            le_count+=1
            
print('%d columns were label encoded.' % le_count)


# In[112]:


# one-hot encoding of categorical variables
new_df = pd.get_dummies(new_df)
new_df_test = pd.get_dummies(new_df_test)

print('Traing Features shape: ', new_df.shape)
print('Testing Features shape: ', new_df_test.shape)


# In[113]:


new_df


# ### Feature Engineering

# Kaggle competitions are won by feature engineering: those win are those who can create the most useful features out of the data. (This is true for the most part as the winning models, at least for structured data, all tend to be variants on gradient boosting).This represents one of the patterns in machine learning: feature engineering has a greater return on investment than model building and hyperparameter tuning. This is a great article on the subject). As Andrew Ng is fond of saying: "applied machine learning is basically feature engineering."
# 
# While choosing the right model and optimal settings are important, the model can only learn from the data it is given. Making sure this data is as relevant to the task as possible is the job of the data scientist (and maybe some automated tools to help us out).
# 
# Feature engineering refers to a geneal process and can involve both feature construction: adding new features from the existing data, and feature selection: choosing only the most important features or other methods of dimensionality reduction. There are many techniques we can use to both create features and select features.
# 
# We will do a lot of feature engineering when we start using the other data sources, but in this notebook we will try only two simple feature construction methods:
# ㆍ Polynomial features
# ㆍ Domain knowledge features

# ### Domain Knowledge Features
# 
# Maybe it's not entirely correct to call this "domain knowledge" because I'm not a credit expert, but perhaps we could call this "attempt at applying limited financial knowledge". In this frame of mind, we can make a couple feature that attempt to capture what we think may be important to telling wethere a client will default on a loan. Here I'm going to use five features:
# 
#     - CREDIT_INCOME_PERCENT : the percentage of the credit amount relative to a client's income
#     - ANNUITY_INCOME_PERCENT : the percentage of the loan annuity relative to a client's income 
#     - CREDIT_TERM : the length of the payment in months (since the annuity is the monthly amount due 
#     - DAYS_EMPLOYED_PERCENT : the percentage of the days employed relative to the client's age

# ### Train Model

# In[114]:


app_train_domain= app_train.copy()
app_test_domain= app_test.copy()

app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']

app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] /app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']


# In[115]:


plt.figure(figsize = (12, 20))
# iterate through the new features
for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):
        
        #create a new subplot for each source
        plt.subplot(4, 1, i+1)
        #plot repaid loans
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label = 'target == 0')
        #plot loans that were not repaid
        sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label = 'target == 1')
        
        # Label the plots
        plt.title('Distribution of %s by Target Value' % feature)
        plt.xlabel('%s' % feature); plt.ylabel('Density');
        
plt.tight_layout(h_pad = 2.5)


# It's hard to say ahead of time if these new features will be useful. The only way to tell for sure is to try them out!

# ### Logistic Regression & Random Forest Implementation

# In[116]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

app_train2 = new_df.dropna()
train = app_train2.drop(columns=['TARGET'])
label = app_train2['TARGET']

x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.25, random_state=2023)


# In[117]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Make the model with the specified regularization parameter
log_reg = RandomForestClassifier(random_state=42, max_depth=3, class_weight="balanced")

# Train on the training data
log_reg.fit(x_train, y_train)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(x_test)[:, 1]
log_reg_pred_train = log_reg.predict_proba(x_train)[:, 1]

print(roc_auc_score(y_test, log_reg_pred))
print(roc_auc_score(y_train, log_reg_pred_train))


# In[118]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Make the model with the specified regularization parameter
log_reg2 = LogisticRegression(random_state=42)

# Train on the training data
log_reg2.fit(x_train, y_train)

# Make predictions
# Make sure to select the second column only
log_reg_pred2 = log_reg.predict_proba(x_test)[:, 1]
log_reg_pred_train2 = log_reg2.predict_proba(x_train)[:, 1]

print(roc_auc_score(y_test, log_reg_pred2))
print(roc_auc_score(y_train, log_reg_pred_train2))


# In[119]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

train = new_df.drop(columns=['TARGET'])
label = new_df['TARGET']

x_train, x_test, y_train, y_test = train_test_split(train, label, test_size=0.25)

# Feature names
features = list(train.columns)

# Copy of the testing data
test = new_df_test.copy()

# Median imputation of missing values
imputer = SimpleImputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(x_train)

# Transform both training and testing data
x_train = imputer.transform(x_train)
x_test = imputer.transform(x_test)

# Repeat with the scaler
scaler.fit(x_train)
train = scaler.transform(x_train)
test = scaler.transform(x_test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)


# We will use LogisticRegression from Scikit-Learn for our first mode. The only change we will make from the default model settings is to lower the regularization parameter, C, which controls the amount of overfitting (a lower value should decrease overfitting). This will get us slightly better result than the default LogisticRegression, but it still will set a low bar for any future models.
# 
# Here we use the familiar Scikit-Learn modeling syntax: we first create the model, then we train the model using .fit and then we make predictions on the testing data using .predict_proba (remember that we want probabilities and not a 0 or 1).

# In[120]:


from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001)

# Train on the training data
log_reg.fit(train, y_train)


# In[121]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Make the model with the specified regularization parameter
log_reg = RandomForestClassifier(random_state=42, max_depth=3, class_weight="balanced")

# Train on the training data
log_reg.fit(x_train, y_train)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(x_test)[:, 1]
log_reg_pred_train = log_reg.predict_proba(x_train)[:, 1]

print(roc_auc_score(y_test, log_reg_pred))
print(roc_auc_score(y_train, log_reg_pred_train))


# In[122]:


# Extract feature importances
feature_importance_values = log_reg.feature_importances_
feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})


# ### Model Interpretation: Feature Importances
# 
# As a simple method to see which variables are the most relevant, we can look at the feature importances of the random forest. Given the correlations we saw in the exploratory data analysis, we should expect that the most important features are the EXT_SOURCE and the DAYS_BIRTH. We may use these feature importances as a method of dimesionality reduction in future work.

# In[123]:


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measured of
    feature importance provided that higher importance is better.
    
    Args:
        df (dataframe): feature importances. Must have the features in a column 
        called 'features' and the importances in a column called 'importance'
        
    Returns:
        shows a plot of the 15 most importance features
        
        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """
    
    # Sort features according to importance
    df = df.sort_values('importance', ascending = False).reset_index()
        
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
        
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize = (10,6))
    ax = plt.subplot()
        
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align = 'center', edgecolor = 'k')
        
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    
    # Plot Labelling
    plt.xlabel('Normalized Importance'); plt.title('Feature Importances')
    plt.show()
    
    return df


# In[124]:


from sklearn.ensemble import RandomForestClassifier

# Make the random forest classifier
random_forest = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose =1, n_jobs=-1)


# In[125]:


# Show the feature importances for the default features
feature_importances_sorted = plot_feature_importances(feature_importances)


# The demographics that are less likely to default on loans include:
# - Students
# - Pensioners
# - People with higher education degrees
# 
# The analysis also found that people in the age group of 20 to 30 are more likely to default on loans than people above the age of 45.
# 
# The bank can use this information to target its loan applications to people who are less likely to default, which will help to reduce the number of defaults and improve the bank's bottom line.
# 
# It is important to note that this is just one analysis of loan data, and there may be other factors that contribute to loan defaults. Additionally, this analysis does not take into account individual circumstances, so it is important to always evaluate each loan application on a case-by-case basis.
