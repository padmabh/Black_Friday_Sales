#!/usr/bin/env python
# coding: utf-8

# # Black Friday Dataset EDA

# # Cleaning and preparing the data

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Problem Statement

# In[8]:


#A retail company “ABC Private Limited” wants to understand the customer purchase behaviour (specifically, purchase amount)
#against various products of different categories. 
#They have shared purchase summary of various customers for selected high volume products from last month.
#The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city),
#product details (product_id and product category) and Total purchase_amount from last month.


# In[56]:


#importing the file
df_train=pd.read_csv("C:\\Users\\Lenovo\\Downloads\\Black_Friday_Sale_train.csv")

df_train.head()


# In[42]:


df_test=pd.read_csv('C:\\Users\\Lenovo\\Downloads\\Black_Friday_Sale_test.csv')
df_test.head()


# In[57]:


df=df_train.append(df_test)

df.head()


# In[58]:


df.info()


# In[59]:


df.describe()


# In[60]:


df.drop('User_ID',axis=1,inplace=True)


# In[61]:


df.head()


# In[48]:


# handling categorical feature by gender
df["Gender"]=df["Gender"].map({'F':0,'M':1})
df


# In[62]:


df['Age'].unique()


# In[63]:


# handle categorical feature age
df["Age"]=df["Age"].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})

df


# In[64]:


#Fixing categorical city_category

df_city=pd.get_dummies(df['City_Category'],drop_first=True)

df_city


# In[68]:


df=pd.concat([df,df_city],axis=1)
df


# In[ ]:


#drop city Category
df.drop('City_Category',axis=1,inplace=True)


# In[72]:


df


# In[73]:


#Missing values
df.isnull().sum()


# In[74]:


# Focus onn replacing missing values
df['Product_Category_1'].unique()


# In[75]:


df['Product_Category_1'].value_counts()


# In[105]:


#Replacing a missing value with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[101]:


df['Product_Category_2'].mode()[0]


# In[106]:


df['Product_Category_2'].isnull().sum()


# In[107]:


#Product category 3 replace missing values.
df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[108]:


df['Product_Category_3'].isnull().sum()


# In[110]:


df.shape


# In[111]:


df['Stay_In_Current_City_Years'].unique()


# In[112]:


#replacing + with ''(nothing)
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+',' ')


# In[113]:


df['Stay_In_Current_City_Years']


# In[114]:


df


# In[116]:


#Convert object into integers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)

df.info()


# In[117]:


# covert 'b' and 'c' into integers

df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[118]:


df.info()


# # Visualizations

# In[122]:


#visualization Age VS Purcahse
sns.barplot(x='Age',y='Purchase',hue='Gender',data=df)


# In[123]:


#Observation from 1 visual :- Purchasing of men is higher than women.


# In[126]:


#visualization of purchase with occupation
sns.barplot(x='Occupation',y='Purchase',hue='Gender',data=df)


# In[127]:


sns.barplot(x='Product_Category_1',y='Purchase',hue='Gender',data=df)


# In[128]:


sns.barplot(x='Product_Category_2',y='Purchase',hue='Gender',data=df)


# In[129]:


sns.barplot(x='Product_Category_3',y='Purchase',hue='Gender',data=df)


# In[130]:


#Feature Scaling

df_test=df[df['Purchase'].isnull()]


# In[131]:


df_train=df[~df['Purchase'].isnull()]


# In[132]:


X=df_train.drop('Purchase',axis=1)


# In[133]:


X.head()


# In[134]:


X.shape


# In[135]:


y=df_train['Purchase']


# In[136]:


y.shape


# In[137]:


y


# In[ ]:




