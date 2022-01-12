#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[43]:


company= pd.read_csv("C:/Users/User/Downloads/Company_Data.csv")


# In[44]:


company


# In[45]:


company.info()


# In[46]:


from sklearn import preprocessing


# In[47]:


label_encoder= preprocessing.LabelEncoder()


# In[48]:


company["ShelveLoc"]= label_encoder.fit_transform(company["ShelveLoc"])


# In[49]:


company["US"]= label_encoder.fit_transform(company["US"])


# In[50]:


company["Urban"]= label_encoder.fit_transform(company["Urban"])


# In[51]:


company


# In[52]:


featurecols=['CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','US']


# In[53]:


company["High"]=company.Sales.map(lambda x:1 if x>8 else 0)


# In[54]:


company


# In[55]:


x= company.drop(['Sales','High'],axis=1)


# In[56]:


x= company[featurecols]


# In[57]:


y=company.High


# In[58]:


print(x)


# In[59]:


print(y)


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 40)


# In[62]:


print(x_test)


# In[63]:


print(y_test)


# In[64]:


print(x_train)


# In[65]:


print(y_train)


# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


SC= StandardScaler()


# In[68]:


x_train= SC.fit_transform(x_train)


# In[69]:


x_test= SC.transform(x_test)


# In[70]:


from sklearn.ensemble import RandomForestClassifier 


# In[71]:


classifier= RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=40)
classifier.fit(x_train, y_train)


# In[72]:


classifier.fit(x_train, y_train)


# In[73]:


classifier.score(x_test,y_test)


# In[74]:


y_pred= classifier.predict(x_test)


# In[75]:


y_pred


# In[76]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[77]:


CM= confusion_matrix(y_test,y_pred)


# In[78]:


print(CM)


# In[79]:


accuracy_score(y_test,y_pred)


# In[80]:


classifier= RandomForestClassifier(n_estimators=100,criterion='gini')


# In[82]:


classifier.fit(x_test,y_test)


# In[83]:


classifier.score(x_test,y_test)


# In[ ]:




