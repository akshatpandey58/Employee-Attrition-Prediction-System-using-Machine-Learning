#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv(r'C:\Users\Acer\OneDrive\Documents\emplo.csv') 
df.head(7)


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.isna().sum()


# In[6]:


df.isnull().values.any()


# In[7]:


df.describe()


# In[8]:


df['Attrition'].value_counts()


# In[9]:


fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(x='Age', hue='Attrition', data = df, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));


# In[10]:


for column in df.columns:
    if df[column].dtype == object:
        print(str(column) + ' : ' + str(df[column].unique()))
        print(df[column].value_counts())
        print("_________________________________________________________________")


# In[11]:


df = df.drop('EmployeeNumber', axis = 1) 
df = df.drop('StandardHours', axis = 1) 
df = df.drop('EmployeeCount', axis = 1) 
df = df.drop('Over18', axis = 1) 


# In[12]:


df.corr()


# In[13]:


plt.figure(figsize=(14,14))  #14in by 14in
sns.heatmap(df.corr(), annot=True, fmt='.0%')


# In[14]:


for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])


# In[15]:


df['Age_Years'] = df['Age']


# In[16]:


df = df.drop('Age', axis = 1)


# In[17]:


df


# In[18]:


X = df.iloc[:, 1:df.shape[1]].values 
Y = df.iloc[:, 0].values


# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# In[20]:


forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train,Y_train)


# In[21]:


forest.score(X_train, Y_train)


# In[22]:


cm = confusion_matrix(Y_test, forest.predict(X_test))
  
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
  
print(cm)
print('Model Testing Accuracy = "{}!"'.format(  (TP + TN) / (TP + TN + FN + FP)))
print()


# In[22]:


importances = pd.DataFrame({'feature':df.iloc[:, 1:df.shape[1]].columns,'importance':np.round(forest.feature_importances_,3)}) #Note: The target column is at position 0
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances


# In[23]:


importances.plot.bar()


# In[ ]:




