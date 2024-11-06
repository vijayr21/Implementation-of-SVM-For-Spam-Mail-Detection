# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program.

Step 2: Import the required packages.

Step 3: Import the dataset to operate on.

Step 4: Split the dataset.

Step 5: Predict the required output.

Step 6: End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIJAY R
RegisterNumber: 212223240178
*/
```

```
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
#countvectorizer is a method to convert text to numerical data. The text is transformed to a sparse matrix
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

from sklearn.metrics import confusion_matrix,classification_report
con=confusion_matrix(y_test,y_pred)
print(con)
cl=classification_report(y_test,y_pred)
print(cl)
```

## Output:

data.head:

![image](https://github.com/user-attachments/assets/5409a3f5-a4c8-4834-b09f-e3d53434a021)

data.info:

![image](https://github.com/user-attachments/assets/f2ff642a-c01f-4af9-a678-d2f060abc118)

data.isnull:

![image](https://github.com/user-attachments/assets/04d86014-2f49-4070-9bec-28842bb5dc3f)

Accuracy :

![Screenshot 2024-11-06 114235](https://github.com/user-attachments/assets/0884b7fd-b4a2-40bf-b480-3c1330030167)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
