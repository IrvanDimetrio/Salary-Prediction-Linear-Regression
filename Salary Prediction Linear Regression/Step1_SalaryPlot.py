"""
@author : Muhamad Irvan Dimetrio
NIM     : 18360018
Teknik Informatika
Institut Sains dan Teknologi Nasional
"""
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Visualising the Training set results
plt.figure(1) #n must be a different integer for every window
plt.scatter(X, Y, color = 'red')
plt.title('Salary VS Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()