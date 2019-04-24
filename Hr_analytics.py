 # -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 07:11:41 2019

@author: Gowtham G
"""
import numpy as n
import pandas as pd
import statistics as S
import matplotlib.pyplot as plt

testhr=pd.read_csv("test_HR.csv")
trainhr=pd.read_csv("train_HR.csv")

trainhr['department'].value_counts()

#finding null values in testhr and trainhr

testhr.describe()
trainhr.describe()

testhr.info()
trainhr.info()

trainhr.columns[trainhr.isnull().any()]
#$print(null_col)
trainhr['previous_year_rating']=trainhr['previous_year_rating'].fillna('0')
print(trainhr['education'].unique(),trainhr['department'].unique())
#trainhr['education']=trainhr['education'].astype('object')
trainhr['education']=trainhr.groupby(['department','age','region'])['education'].apply(lambda x: x.fillna(x.value_counts()))



train_Age_Q1 = trainhr['age'].quantile(0.25)
train_Age_Q3 = trainhr['age'].quantile(0.75)
train_Age_IQR = train_Age_Q3 - train_Age_Q1
print(train_Age_Q1,train_Age_Q3,train_Age_IQR)
trainhr = trainhr[~((trainhr.age<(train_Age_Q1-1.5*train_Age_IQR))|(trainhr.age>(train_Age_Q3+1.5*train_Age_IQR)))]

sns.countplot(x=trainhr['education'])
y=trainhr['is_promoted']
trainhr=trainhr.drop(['is_promoted'],axis=1)


X=pd.get_dummies(trainhr)
#xtest=pd.get_dummies(testhr)



from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(
        X,y,random_state=123,test_size=0.3)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
regression = LogisticRegression()
regression.fit(x_train,y_train)
y_pred = regression.predict(x_test)

y_predhr=regression.predict(X_testhr)

y_pred_df=pd.DataFrame(y_pred,columns=['count'])
pd.crosstab(index=y_pred_df["count"],columns="count") 


y_test_df=pd.DataFrame(y_test,columns=['is_promoted'])
pd.crosstab(index=y_test_df['is_promoted'],columns="is_prmoted")

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)


from sklearn.metrics import f1_score
f1_score(y_pred,y_test)


from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


from sklearn.externals import joblib
joblib.dump(regression,"Hr_analyticsModel2.py")
clf1 = joblib.load("Hr_analyticsModel2.py")
clf1.predict([testhr[0]])



import seaborn as sns

sns.boxplot(x=trainhr['no_of_trainings'])
sns.boxplot(x=trainhr['age'])
sns.countplot(y_pred)


from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test,y_pred)


#testing 

testhr.columns[testhr.isnull().any()]
testhr['previous_year_rating']=testhr['previous_year_rating'].fillna('0')
print(testhr['education'].unique(),testhr['department'].unique())
#trainhr['education']=trainhr['education'].astype('object')
testhr['education']=testhr.groupby(['department','age','region'])['education'].apply(lambda x: x.fillna(x.value_counts()))

sns.countplot(testhr['education'])


x1_testhr=pd.get_dummies(testhr)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_testhr=sc.fit_transform(x1_testhr)
#x_testhr=sc.transform(X_testhr)
y_predhr=regression.predict(X_testhr)

y_predhrdf=pd.DataFrame(y_predhr,columns=["isPromoted"])
pd.crosstab(index=y_predhrdf["isPromoted"],columns="isPromoted") 

isPromoted = pd.DataFrame(y_predhrdf,columns=['isPromoted'])
f=[testhr['employee_id'],isPromoted]
finaldata=pd.concat(f,axis=1)

finaldata.to_csv("D:\Capstone Project\PredictedOutput.csv",index=False)





