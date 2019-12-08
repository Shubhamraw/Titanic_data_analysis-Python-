# 1.Collecting data

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import math

titanic_data = pd.read_csv("titanic.csv")
titanic_data.head(10)

print("no. of passengers in original data:" +str(len(titanic_data.index)))

# 2.Analyzing Data
#Creating different plot to check relationship between variables
#Implement logistic Regression    
sns.countplot(x="Survived", data = titanic_data)
sns.countplot(x="Survived", hue="Sex", data=titanic_data)
sns.countplot(x="Survived", hue="Pclass", data=titanic_data)
titanic_data['Age'].plot.hist() #analysis on the age column
titanic_data['Fare'].plot.hist(bins=20, figsize=(10,5))

titanic_data.info

sns.countplot(x="SibSp",data=titanic_data)

# 3.Data Wrangling
# Clean the data by removing the Nan values and unnecessary columns in the dataset
#Implement logistic Regression  
#step 1:
titanic_data.isnull()
titanic_data.isnull().sum()
sns.heatmap(titanic_data.isnull(), yticklabels=False, cmap='viridis')
sns.boxplot(x='Pclass', y='Age',data=titanic_data)
titanic_data.head(5)
titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.head(5)
titanic_data.dropna(inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False) 
titanic_data.isnull().sum()

titanic_data.head(2)
pd.get_dummies(titanic_data['Sex'])
sex=pd.get_dummies(titanic_data['Sex'],drop_first=True)
sex.head(5)
embark = pd.get_dummies(titanic_data['Embarked'])
embark.head(5)
embark = pd.get_dummies(titanic_data['Embarked'],drop_first=True)
embark.head(5)
Pcl = pd.get_dummies(titanic_data['Pclass'],drop_first=True)
Pcl.head(5)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data.head(5)
titanic_data.drop(['Sex','Embarked', 'PassengerId','Name','Ticket'],axis=1,inplace=True)
titanic_data.head(5)
titanic_data.drop('Pclass',axis=1,inplace=True)
titanic_data.head(5)

# 4.Training and testing the data
# Build the model on the train data and predict the output on the test data
X= titanic_data.drop('Survived',axis=1)
y= titanic_data['Survived']
from sklearn.model_selection import train_test_split
#help train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state = 1)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
classification_report(y_test,predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)