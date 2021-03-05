#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
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


# In[ ]:

data_list = ['Raw', 'SMOTE', 'ADASYN', 'CNN', 'SMOTE + ENN', 'ADASYN + ENN']
cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC']

# In[ ]:

print(X_samples.get(data_list[0]))
print(y_samples.get(data_list[0]))


# # KNN

# 각 분류 모델의 성능을 평가(model selection) 방법
# 1. Accuracy
# 2. Confusion matrix
# 3. Precision, Recall and F-measure(f1score, f-beta-score) 
# 4. Receiver operating characteristic (AUC-ROC)

# In[ ]:

from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 5)

data_dict = {}
for i in data_list:
    print(i)
#     clf.fit(X, y)
    clf.fit(X_samples.get(i), y_samples.get(i))
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
print("classifier created")

# In[ ]:


#프로모션 평가 코드
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

def evaluate(test, pred):

    confusion_mtx = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba[:, 1])

    print('오차 행렬',confusion_mtx)
    print('\n정확도: {:.4f}, 정밀도: {:.4f}, 재현율: {:.4f}, fl 스코어: {:.4f}, roc_auc : {:.4f}'.format(accuracy, precision, recall, f1, roc_auc))
    
print(evaluate(y_test,pred))


# # 현주님 코드
#
# # In[ ]:
#
# print('현주님 코드')
# for key, sample in X_samples.items(): #y_samples 해도 상관x
#     y = y_samples[key]
#     total = len(y)
#     counts = y.value_counts()
#
#     print('=' * 100)
#     print(key)
#     print('=' * 100)
#     print(sample.head())
#     print('-' * 100)
#     print(counts)
#     print('-' * 30)
#     print(f"Total : {total}")
#     for idx in counts.index:
#         print(f"{idx} 비율 : {counts[idx] / total * 100:6.2f} %")
#     print('=' * 100)
#
#
# # In[ ]:
#
#
# def show_data_info(y):
#     total = len(y)
#     counts = y.value_counts()
#     print('=' * 80)
#     print(counts)
#     print('-' * 30)
#     for idx in counts.index:
#         print(f"{idx} 비율 : {counts[idx] / total * 100:6.2f} %")
#     print('=' * 80)
#
# def evaluate_model(clf, x_test, y_test):
#     y_proba = clf.predict_proba(x_test)
#     y_pred = clf.predict(x_test)
#
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     fl = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_proba[:, 1])
#
#     print('=' * 80)
#     print('Confusion Matrix')
#     print(confusion_matrix(y_test, y_pred))
#     print('-' * 60)
#     print(f'Accuracy  : {accuracy}')
#     print(f'Precision : {precision}')
#     print(f'Recall    : {recall}')
#     print(f'F1-Score  : {fl}')
#     print('-' * 60)
#     print(classification_report(y_test,y_pred))
#     print('-' * 60)
#     print(f'ROC AUC : {roc_auc}')
#     print('=' * 80)
#
#     return accuracy, precision, recall, fl, roc_auc
#
#
# # In[ ]:
#
#
# class_weights = ['Balanced', 'SqrtBalanced']
#
# for class_weight in class_weights:
#     X, y = X_samples['Raw'], y_samples['Raw']
#
#     print('=' * 80)
#     print('Raw')
#     show_data_info(y)
#
#     clf = KNeighborsClassifier(n_neighbors = 5,n_jobs=-1)
#     clf.fit(X, y)
#     print("classifier created")
#
#     print(f'KNN {key}')
#     result = evaluate_model(clf, X_test, y_test)
#
#     clf[class_weight] = clf
#     results[class_weight] = result
#
#
# # In[ ]:
#
#
# cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC_AUC']
#
# evaluation = pd.DataFrame(results, index=cols)
# evaluation.index.set_names('Score', inplace=True)
# print(evaluation)
#
#
# # In[ ]:
#
#
# evaluation.plot(y=X_samples.keys(), kind="bar", figsize=(10, 5))
# plt.legend(loc=(1.01, 0.))
# plt.show()
#
