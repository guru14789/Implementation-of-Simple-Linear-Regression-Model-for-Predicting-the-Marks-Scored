# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter Notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representation in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given data.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SREEKUMAR S
RegisterNumber:212223240157

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("/content/student_scores.csv")
df.head()
```
```
df.tail()
```
```
x=df.iloc[:,:-1].values
x
```
```
y=df.iloc[:,:1].values
y
```
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
```
```
Y_test
```
```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,regressor.predict(X_train),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
plt.scatter(X_train,Y_train,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
```
mse=mean_squared_error(ytest,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(ytest,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
## df.head()
![Screenshot 2024-02-20 111916](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/b5eeddc8-d488-4302-9f0f-350908578af5)
## df.tail() 
![Screenshot 2024-02-20 112048](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/07670bc7-d28b-47d9-a8cd-d094f00042ae)
## X:
![Screenshot 2024-02-20 113243](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/fa06e7f2-a18a-4505-a936-19d78ab725d7)
## Y:
![Screenshot 2024-02-20 113322](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/122405fe-e83e-4b6c-96ec-3c6197b04347)
## Y_pred:
![Screenshot 2024-02-20 113352](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/9d3851b8-fdf8-451e-bee1-18cc6173806f)
## Y_test:
![Screenshot 2024-02-20 113430](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/5518b70e-f33b-47d3-9a6c-d68ce90001a5)

## training dataset:
![Screenshot 2024-02-20 113505](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/71ed4092-195b-45b1-b746-dbf7cfc656d4)

## testing dataset:
![Screenshot 2024-02-20 113618](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/38f2f32c-f889-4d8f-8cdc-920e2da7c98a)

## mse,mae,rmse:
![Screenshot 2024-02-20 113718](https://github.com/guru14789/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/151705853/1551603e-9015-41cc-a1a1-573d142d7df8)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
