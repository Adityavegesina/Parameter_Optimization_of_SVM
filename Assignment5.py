#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Aditya Vegesina 102017171

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



# In[2]:
data = pd.read_csv("C:\\Users\\adity\\Downloads\\Crowdsourced Mapping\\training.csv");
X = data.drop(['class'], axis=1)
y = data["class"]

X_trainlist = []
X_testlist = []
y_trainlist = []
y_testlist = []

for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i+1)
    X_trainlist.append(X_train)
    X_testlist.append(X_test)
    y_trainlist.append(y_train)
    y_testlist.append(y_test)


# In[3]:


param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 11, 102],
    'degree':np.arange(3,6), 
}


# In[4]:


best_params_list = []
accuracy_list = []
convergence_list = []
for i in range(10):
    X_train_sample = X_trainlist[i]
    X_test_sample = X_testlist[i]
    y_train_sample = y_trainlist[i]
    y_test_sample = y_testlist[i]

    svm = SVC(max_iter=1000)

    grid_search = GridSearchCV(svm, param_grid, cv=5)
    grid_search.fit(X_train_sample, y_train_sample)
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    svm_best = SVC(**best_params, max_iter=1000)
    accuracy_iteration = []

    for iter in range(1, 1001):
        svm_best.fit(X_train_sample, y_train_sample)
        y_pred = svm_best.predict(X_test_sample)
        accuracy = accuracy_score(y_test_sample, y_pred)
        accuracy_iteration.append(accuracy)

    accuracy_list.append(accuracy_iteration)
    convergence_list.append(accuracy_iteration[-1])

max_acc_index = np.argmax(convergence_list)
best_params_max_acc = best_params_list[max_acc_index]
accuracy_max_acc = accuracy_list[max_acc_index]

Accuracy = []
for i in range(10):
    Accuracy.append(accuracy_list[i][999])


# In[7]:


df_best_params = pd.DataFrame(best_params_list)
df_best_params.index.name = 'Sample'
df_best_params['Accuracy'] = Accuracy
df_best_params.columns = ['Epsilon', 'Nu', 'Kernel','Best  Accuracy']


# In[8]:
print("Table: Best Parameters")
print(df_best_params)

# Create line plot of accuracy convergence for sample with maximum accuracy
plt.plot(range(1, 1001), accuracy_max_acc)
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Accuracy Convergence (Sample {})'.format(max_acc_index+1))
plt.show()










