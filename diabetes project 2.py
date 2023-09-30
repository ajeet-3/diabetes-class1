
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
df=pd.read_csv("diabetes.csv")

df.head()
df.rename(columns={'Pregancies':'Preg','BloodPressure':'BP','SkinThickness':'ST','DiabetesPedigreeFunction':'DPF'},inplace=True)
print(df.head())
print(df.shape)

#CHECK FOR MISSING DATA

print(df.isnull().sum())

# no. of classes 

print(df.Outcome.value_counts())

df.iloc[:,1:-1].hist(bins=30,figsize=(15,10))
plt.show()

# replace or drop zero values
print(df.iloc[:,1:-1].isin([0]).sum())
print(df.iloc[:,1:-3].describe())

# replace zero values with median(can take any other option)
for col in ['Glucose','BP','ST','Insulin','BMI']:
    df[col]=df[col].replace({0:df[col].median()})
print(df.iloc[:,1:-3].isin([0]).sum())
print(df.head())


# FEATURE SELECTION USING RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
x=df.iloc[:,:-1].values
y=df.Outcome.values

rf =RandomForestClassifier(n_estimators=20,random_state=0)
rf.fit(x,y)

f= rf.feature_importances_

print(f)

for c,k in zip(df.columns[:-1],f):
    print(c,':',k)

df_new= df[['Glucose','BMI','Age','DPF','Insulin']]
print(df_new.shape)

# split data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

xtrain,xtest,ytrain,ytest=train_test_split(df_new.values,df.Outcome.values,test_size=.2,random_state=12)

print(xtrain.shape)

# Feature scaling

from sklearn.preprocessing import StandardScaler
std=StandardScaler()
std_train=std.fit_transform(xtrain)
std_test=std.transform(xtest)

print(std_train.shape)
print(df.Outcome.value_counts())


#LOGISTIC REGRESSION 
log=LogisticRegression()
log.fit(std_train,ytrain)

train_score= log.score(std_train,ytrain)
test_score = log.score(std_test,ytest)
print('------------------------------------------------------------------------------------')

print(train_score)
print(test_score)

print(np.bincount(ytest))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest,log.predict(std_test)))


# SVM

clf_svm=SVC(kernel ='linear',C=10)
clf_svm.fit(std_train,ytrain)
train_score = clf_svm.score(std_train,ytrain)
test_score = clf_svm.score(std_test,ytest)
print('------------------------------------------------------------------------------------')

print(train_score)
print(test_score)
print(confusion_matrix(ytest,clf_svm.predict(std_test)))

# KNN 

knn =KNeighborsClassifier(n_neighbors=3)
knn.fit(std_train,ytrain)

train_score=knn.score(std_train,ytrain)
test_score=knn.score(std_test,ytest)
print('------------------------------------------------------------------------------------')

print(train_score)
print(test_score)

print(confusion_matrix(ytest,knn.predict(std_test)))


# HYPERTUNING 

from sklearn.model_selection import GridSearchCV
p={'C':[.001,.01,.1,10]}
gd=GridSearchCV(SVC(kernel='linear'),param_grid=p,cv=5,scoring='accuracy')

gd.fit(std_train,ytrain)
print('------------------------------------------------------------------------------------')

print(gd.best_score_)

print(gd.best_params_)

clf_final =gd.best_estimator_
print(clf_final.score(std_test,ytest))