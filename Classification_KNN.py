#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from sklearn import metrics
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix
import pickle


# In[2]:

pd.set_option("display.max_columns", None)


# # 데이터

# In[9]:

with open(f"./datasets/X_samples.pickle", "rb") as f:
    X_samples = pickle.load(f)
    
with open(f"./datasets/y_samples.pickle", "rb") as f:
    y_samples = pickle.load(f)
    
with open(f"./datasets/X_test.pickle", "rb") as f:
    X_test = pickle.load(f)
    
with open(f"./datasets/y_test.pickle", "rb") as f:
    y_test = pickle.load(f)


# In[10]:

# samples는 dict,test는 df 
print('X_samples key : ' , X_samples.keys()) 
print('y_samples key : ' , y_samples.keys())
print(X_test.head())
print(y_test.head())



clf = KNeighborsClassifier(n_neighbors = 5)

data_list = ['Raw', 'SMOTE', 'ADASYN', 'CNN', 'SMOTE + ENN', 'ADASYN + ENN']
cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC']
# print(X_samples.get(data_list[0]))
# print(y_samples.get(data_list[0]))

data_dict = {}
for i in data_list:
    print(i)
    clf.fit(X_samples.get(i), y_samples.get(i))
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_pred, y_test)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    data_dict[i] = [acc, precision, recall, f1, roc_auc]


df_knn = pd.DataFrame(data_dict, index = cols)
print(df_knn)


# In[ ]:


df_knn.plot(kind = 'bar', figsize = (10, 5))
plt.legend(loc=(1.01, 0.))
plt.xticks(rotation = 0)
plt.show()

