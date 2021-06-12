"""
@author : Muhamad Irvan Dimetrio
NIM     : 18360018
Teknik Informatika
Institut Sains dan Teknologi Nasional
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

#Splitting the dataset into the Training set and Test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 1/3, random_state=0)

#Get Linear Regression as model training the model using X_train, Y_train data
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#How much accuracy of the model
akurasi = regressor.score(X_train, Y_train)
print("Akurasi dari model adalah : {}".format(akurasi))

#Visualising the training set results
plt.figure(1) #n must be a different integer for every windows
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

#Visualising the training set results
plt.figure(2) #n must be a different integer for every windows
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary VS Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

plt.show() # dengan matplotlib figure semua window digambar dan ditampilkan melalui plt.show()