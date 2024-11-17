# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Ashwin Kumar A
RegisterNumber: 212223040021 
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows=1252')
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

![image](https://github.com/user-attachments/assets/a32d3c08-e5ca-405f-8b5b-4db7357db890)

## data.head()
![image](https://github.com/user-attachments/assets/d8e89f33-3f4c-48b9-a94a-577ad07c7db2)

## data.info()
![image](https://github.com/user-attachments/assets/4c1f48a1-ead2-42a1-a070-a82791ffcc2f)
## data.isnull().sum()
![image](https://github.com/user-attachments/assets/58f0eeb5-4148-4773-994e-40c3081962b7)
## Prediction y
![image](https://github.com/user-attachments/assets/06358c50-be41-4aeb-b862-c11945f925ab)
## Accuracy
![image](https://github.com/user-attachments/assets/b4224b6d-4cfc-4923-98f6-9a2a70e8d4f9)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
