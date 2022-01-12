#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


fraud= pd.read_csv("C:/Users/User/Downloads/Fraud_check.csv")


# In[3]:


fraud


# In[4]:


fraud.info()


# In[5]:


fraud["TaxInc"] = pd.cut(fraud["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])


# In[6]:


fraud["TaxInc"]


# In[7]:


fraud= fraud.drop(columns=["Taxable.Income"])


# In[8]:


fraud


# In[9]:


Fraud= pd.get_dummies(fraud.drop(columns=["TaxInc"]))


# In[10]:


FC= pd.concat([Fraud,fraud["TaxInc"]],axis=1)


# In[11]:


colnames= list(FC.columns)


# In[12]:


colnames


# In[13]:


predictors= colnames[:9]


# In[14]:


predictors


# In[15]:


target= colnames[9]


# In[16]:


target


# In[17]:


X= FC[predictors]


# In[19]:


X.shape


# In[20]:


Y=FC[target]


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 40)


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


SC= StandardScaler()


# In[25]:


X_train= SC.fit_transform(X_train)


# In[26]:


X_test= SC.transform(X_test)


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


classifier= RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=40)


# In[31]:


classifier.fit(X_train,Y_train)


# In[32]:


classifier.fit(X_train,Y_train)


# In[33]:


classifier.score(X_train,Y_train)


# In[34]:


y_pred = classifier.predict(X_test)


# In[35]:


y_pred


# In[36]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[37]:


cm = confusion_matrix(Y_test, y_pred)


# In[38]:


print(cm)


# In[39]:


accuracy_score(Y_test, y_pred)


# In[40]:


classifier = RandomForestClassifier(n_estimators=100, criterion='gini')


# In[43]:


classifier.fit(X_train, Y_train)


# In[ ]:




