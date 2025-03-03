# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages.
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: THIRUMURUGAN R
RegisterNumber:212223220118
*/
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)

```

## Output:
To read head and tail files

![image](https://github.com/user-attachments/assets/48df5cf8-00dd-4445-b1fb-94d00ee811b6)

![image](https://github.com/user-attachments/assets/505cbbf4-6bb1-4680-ae37-25bebf7710c2)

Compare Dataset

![image](https://github.com/user-attachments/assets/d8fdd3a5-7f7e-4536-bbbd-bc2be5f2b9ce)

Predicted Value

![image](https://github.com/user-attachments/assets/03062372-562c-496a-a3c0-847191df63f0)

Graph For Training Set 

![image](https://github.com/user-attachments/assets/e7d5a744-b98a-4de0-8985-94580e88dde5)

Graph For Testing Set

![image](https://github.com/user-attachments/assets/a49a6bc4-ee43-45f6-ab9f-f4b6650bc5c1)

Error

![image](https://github.com/user-attachments/assets/549fdd55-5f28-4798-aea6-29d6df0a2c91)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
