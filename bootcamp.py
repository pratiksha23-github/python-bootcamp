#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[6]:


data=pd.DataFrame(pd.read_csv("train.csv"))


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.shape


# In[10]:


data.isnull().sum()


# In[13]:


a=data.isnull().sum()
drop_col=a[a>(35/100*data.shape[0])]
drop_col


# In[14]:


drop_col.index


# In[15]:


data.drop(drop_col.index,axis=1,inplace=True)
data.isnull().sum()


# In[21]:


data.fillna(data.mean(),inplace=True)
data.isnull().sum()


# In[17]:





# In[20]:


data['Embarked'].describe()


# In[22]:


data['Embarked'].fillna('S',inplace=True)


# In[23]:


data.isnull().sum()


# In[24]:


data.corr()


# In[26]:


data['FamilySize']=data['SibSp']+data['Parch']
data.drop(['SibSp','Parch'],axis=1,inplace=True)
data.corr()


# In[27]:


data['Alone']=[0 if data['FamilySize'][i]>0 else 1 for i in data.index]
data.head()


# In[28]:


data.groupby(['Alone'])['Survived'].mean()


# In[29]:


data[['Alone','Fare']].corr()


# In[30]:


data['Sex']=[0 if data['Sex'][i]=='male' else 1 for i in data.index]
data.groupby(['Sex'])['Survived'].mean()


# In[32]:


data.groupby(['Embarked'])['Survived'].mean()


# # CONCLUSION:-

# 1) Female passengers were prioritized over men.
# 
# 2) People with high class or rich people have higher survival rate than others. The hierarichy might have been followed while saving the passengers.
# 
# 3) Passengers travelling with their family have higher survival rate.
# 
# 4) Passengers who borded the ship at Cherbourg, survived more in proportion then the others.

# In[ ]:




