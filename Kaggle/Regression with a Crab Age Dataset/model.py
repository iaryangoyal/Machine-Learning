#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[3]:


train.head()


# In[4]:


train.info()


# In[5]:


test.head()


# In[6]:


test.info()


# In[7]:


test.describe()


# In[8]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[9]:


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# no null point in both datas


# In[10]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[11]:


train.drop(['Sex'],axis=1,inplace=True)


# In[12]:


train = pd.concat([train,sex],axis=1)


# In[13]:


train.head()


# In[14]:


sex = pd.get_dummies(test['Sex'],drop_first=True)


# In[15]:


test.drop(['Sex'],axis=1,inplace=True)


# In[16]:


test = pd.concat([test,sex],axis=1)


# In[17]:


test.head()


# In[18]:


test.describe()


# In[19]:


train.describe()


# In[20]:


X_train=train.drop('Age',axis=1)
y_train=train['Age']


# In[21]:


#from sklearn.linear_model import LogisticRegression


# In[22]:


#lm=LogisticRegression()


# In[23]:


#lm.fit(X_train,y_train)


# In[24]:


#predict=lm.predict(test)


# In[25]:


#submission_df = pd.DataFrame(data = {'id': test['id'], 'Age': predict})
#submission_df.to_csv('submission.csv', index = False)
#submission_df.head()


# In[26]:


#from sklearn.tree import DecisionTreeClassifier


# In[27]:


#dtree = DecisionTreeClassifier()


# In[28]:


#dtree.fit(X_train,y_train)


# In[29]:


#predictions = dtree.predict(test)


# In[35]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout


# In[36]:


regressor = Sequential()


# In[44]:


# First LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#sec LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#third LSDM layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#forth LSDM layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#outputs
regressor.add(Dense(units=1))


# In[40]:


regressor.compile(optimizer='adam', loss='mean_squared_error')

#fillting
regressor.fit(X_train,y_train, epochs=100, batch_size=32)


# In[41]:


predicted = regressor.predict(test)


# In[49]:


submission_df = pd.DataFrame(data = {'Age': predicted})
submission_df.to_csv('submission.csv', index = False)
submission_df.head()


# In[ ]:




