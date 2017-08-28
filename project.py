
# coding: utf-8

# # The Sherlock Holmes investigative AI

# # Mostafa Akrami

# ## Problem Set-up
# 

# ### I start by using some useful Python packages 
#

# In[2]:

# Import a bunch of libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
import re
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import pydotplus 
from matplotlib.colors import ListedColormap
get_ipython().magic(u'matplotlib inline')


# ###

# In[3]:

# ###

# In[4]:

train = pd.read_csv('data.csv')


# In[5]:

train['dateofbirth'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
train['dateofdeath'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
train = train.dropna()
print(train.head())


# In[6]:

train.info()


# ###

# In[7]:

train[train['religionLabel'] == 0].head(10)


# ### Some feature engineering

# In[8]:

def text2number(X_word, words):
    numbers = range(len(words))
    
    X_number = X_word.copy()
    for n, w in zip(numbers, words):
        X_number[X_word == w] = n
    return X_number

nativelanguageLabel_words = list(set(train['nativelanguageLabel']))
train['nativelanguageLabel_number'] = text2number(train['nativelanguageLabel'], nativelanguageLabel_words)

sexLabel_words = list(set(train['sexLabel']))
train['sexLabel_number'] = text2number(train['sexLabel'], sexLabel_words)

religionLabel_words = list(set(train['religionLabel']))
train['religionLabel_number'] = text2number(train['religionLabel'], religionLabel_words)

causeofdeathLabel_words = list(set(train['causeofdeathLabel']))
train['causeofdeathLabel_number'] = text2number(train['causeofdeathLabel'], causeofdeathLabel_words)

placeofbirthLabel_words = list(set(train['placeofbirthLabel']))
train['placeofbirthLabel_number'] = text2number(train['placeofbirthLabel'], placeofbirthLabel_words)

placeofdeathLabel_words = list(set(train['placeofdeathLabel']))
train['placeofdeathLabel_number'] = text2number(train['placeofdeathLabel'], placeofdeathLabel_words)

countryofcitizenshipLabel_words = list(set(train['countryofcitizenshipLabel']))
train['countryofcitizenshipLabel_number'] = text2number(train['countryofcitizenshipLabel'], countryofcitizenshipLabel_words)

placeofintermentLabel_words = list(set(train['placeofintermentLabel']))
train['placeofintermentLabel_number'] = text2number(train['placeofintermentLabel'], placeofintermentLabel_words)

occupationLabel_words = list(set(train['occupationLabel']))
train['occupationLabel_number'] = text2number(train['occupationLabel'], occupationLabel_words)

positionheldLabel_words = list(set(train['positionheldLabel']))
train['positionheldLabel_number'] = text2number(train['positionheldLabel'], positionheldLabel_words)

worklocationLabel_words = list(set(train['worklocationLabel']))
train['worklocationLabel_number'] = text2number(train['worklocationLabel'], worklocationLabel_words)

memberofpoliticalpartyLabel_words = list(set(train['memberofpoliticalpartyLabel']))
train['memberofpoliticalpartyLabel_number'] = text2number(train['memberofpoliticalpartyLabel'], memberofpoliticalpartyLabel_words)

militaryrankLabel_words = list(set(train['militaryrankLabel']))
train['militaryrankLabel_number'] = text2number(train['militaryrankLabel'], militaryrankLabel_words)

mannerofdeathLabel_words = list(set(train['mannerofdeathLabel']))
train['mannerofdeathLabel_number'] = text2number(train['mannerofdeathLabel'], mannerofdeathLabel_words)

conflictLabel_words = list(set(train['conflictLabel']))
train['conflictLabel_number'] = text2number(train['conflictLabel'], conflictLabel_words)

print(train.head())


#
# 
#

# In[13]:

fig = plt.figure(figsize=(15, 15))

for idx, f in enumerate(['nativelanguageLabel_number','sexLabel_number', 'religionLabel_number', 'causeofdeathLabel_number', 'placeofbirthLabel_number', 'placeofdeathLabel_number', 'countryofcitizenshipLabel_number', 'placeofintermentLabel_number', 'occupationLabel_number']):
    ax = fig.add_subplot(3,3,idx+1)
    ax.hist(train[f], 15)
    ax.set_xlabel(f)
    ax.set_ylabel('Nr of people')
    
plt.show()


# ###

# In[14]:

import seaborn as sns
sns.set_style('whitegrid')
sns.countplot(x='causeofdeathLabel_number',hue='sexLabel_number',data=train[train['sexLabel']=='male'])


# In[15]:

#sns.set_style('whitegrid')
sns.countplot(x='causeofdeathLabel_number',hue='sexLabel_number',data=train[train['sexLabel']=='female'])




#

# In[34]:

##
X_num = train['causeofdeathLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'causeofdeathLabel_' + str(num)
    train[name] = X_cat


X_num = train['religionLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'religionLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['nativelanguageLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'nativelanguageLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['placeofbirthLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'placeofbirthLabel_' + str(num)
    train[name] = X_cat

X_num = train['placeofdeathLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'placeofdeathLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['countryofcitizenshipLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'countryofcitizenshipLabel_' + str(num)
    train[name] = X_cat

X_num = train['placeofintermentLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'placeofintermentLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['occupationLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'occupationLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['positionheldLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'positionheldLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['worklocationLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'worklocationLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['memberofpoliticalpartyLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'memberofpoliticalpartyLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['militaryrankLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'militaryrankLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['mannerofdeathLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'mannerofdeathLabel_' + str(num)
    train[name] = X_cat
    

X_num = train['conflictLabel_number']
nmax = max(X_num)+1

for num in range(nmax):
    X_cat = np.zeros_like(X_num)
    for idx, entry in enumerate(X_num):
        if entry == num:
            X_cat[idx] = 1
    name = 'conflictLabel_' + str(num)
    train[name] = X_cat

train.head()




# In[57]:

X = train[['religionLabel_number', 'nativelanguageLabel_number', 'dateofbirth', 'dateofdeath', 'placeofbirthLabel_number', 'placeofdeathLabel_number', 'countryofcitizenshipLabel_number', 'placeofintermentLabel_number', 'occupationLabel_number', 'positionheldLabel_number', 'worklocationLabel_number', 'memberofpoliticalpartyLabel_number', 'militaryrankLabel_number', 'conflictLabel_0', 'conflictLabel_1', 'conflictLabel_2', 'conflictLabel_3', 'conflictLabel_4', 'conflictLabel_5', 'conflictLabel_6', 'conflictLabel_7', 'conflictLabel_8', 'conflictLabel_9']].values
Y = train['causeofdeathLabel'].values        


M = X.shape[0]
M_train = int(M*0.8)

X_train = X[:M_train, :]
X_dev = X[M_train:, :]
Y_train = Y[:M_train]
Y_dev = Y[M_train:]


#

# In[58]:

dt = DecisionTreeClassifier(criterion="entropy", max_depth=10)
dt.fit(X_train, Y_train)
print('Accuracy (a decision tree):', dt.score(X_dev, Y_dev))


# In[59]:

vec_acc = []
for num_est in [5,10,50,100, 500, 800, 1000, 1500]:
    rfc = RandomForestClassifier(criterion = "entropy", n_estimators=num_est)
    rfc.fit(X_train, Y_train)
    acc = rfc.score(X_dev, Y_dev)
    vec_acc.append(acc)
    print('Accuracy (a random forest):', rfc.score(X_dev, Y_dev))
plt.plot(vec_acc)




# In[60]:

k_vec = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
acc_vec = []
for k in k_vec:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    preds = knn.predict(X_dev)
    acc = np.mean(preds == Y_dev)
    acc_vec.append(acc)
    #print('Accuracy for %i is %f' %(k, acc))
    
best_k = k_vec[np.argmax(acc_vec)]
print('Best k is: ', best_k, ' with accuracy ', max(acc_vec))

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, Y_train)
knn_preds = knn.predict(X_dev)


# In[62]:

c_vec = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
acc_vec = []
for c in c_vec:
    lr2 = LogisticRegression(penalty='l2', C=c)
    lr2.fit(X_train, Y_train)
    preds = lr2.predict(X_dev)
    acc = np.mean(preds == Y_dev)
    acc_vec.append(acc)
    print('Accuracy for %i is %f' %(c, acc))
    
best_c = c_vec[np.argmax(acc_vec)]
print('Best c is: ', best_c, ' with accuracy ', max(acc_vec))

lr2 = LogisticRegression(penalty='l2', C=best_c)
lr2.fit(X_train, Y_train)
lr2_preds = lr2.predict(X_dev)

# In[63]:

c_vec = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
acc_vec = []
for c in c_vec:
    lr1 = LogisticRegression(penalty='l1', C=c)
    lr1.fit(X_train, Y_train)
    preds = lr1.predict(X_dev)
    acc = np.mean(preds == Y_dev)
    acc_vec.append(acc)
    print('Accuracy for %i is %f' %(c, acc))
    
best_c = c_vec[np.argmax(acc_vec)]
print('Best c is: ', best_c, ' with accuracy ', max(acc_vec))

lr1 = LogisticRegression(penalty='l1', C=best_c)
lr1.fit(X_train, Y_train)
lr1_preds = lr1.predict(X_dev)

# In[64]:

d_vec = [1,3,5,7,9,11,13,15,17,19]
acc_vec = []
for d in d_vec:
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=d)
    dt.fit(X_train, Y_train)
    preds = dt.predict(X_dev)
    acc = np.mean(preds == Y_dev)
    acc_vec.append(acc)
    print('Accuracy for %i is %f' %(d, acc))
    
best_d = d_vec[np.argmax(acc_vec)]
print('Best depth is: ', best_d, ' with accuracy ', max(acc_vec))

dt = DecisionTreeClassifier(criterion="entropy", max_depth=best_d)
dt.fit(X_train, Y_train)
dt_preds = lr1.predict(X_dev)


