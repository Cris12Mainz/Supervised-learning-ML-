import csv
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np


model = Perceptron()
#model = svm.SVC()
#model = KNeighborsClassifier(n_neighbors=1)
#model = GaussianNB()

# Read data in from file
with open("data.csv") as a:
    csv_reader = csv.reader(a)
    next(csv_reader)

    data = []
    for row in csv_reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
#            "label": "Authentic" if row[4] == "0" else "Counterfeit"
            "label": [float(cell) for cell in row[4]],
        })

# Separate data into training and testing groups
holdout = int(0.40* len(data))#we split the data in two groups.
random.shuffle(data) #the data is shuffled in order to make the scenario more realistic.
testing = data[:holdout] # the first # % is the data that will be used as a test.
training = data[holdout:]# the remaining (100-#)% is the data the will be used to train the model.

# Train model on training set. We divide the five columns into two groups. For instance, this model can represent when it will or not rain. Then, the first four colummns represent the conditions: temperature, speed of the wind, pressure and altitude. And, the fifth columm will indicate whether it will rain or not.
X_training = [row["evidence"] for row in training] # conditions for rains.
y_training = [row["label"] for row in training] #it will rain or not
model.fit(X_training, y_training) #This function trains the "model" with the trainning data and then it can predict data.

# Make predictions on the testing set
X_testing = [row["evidence"] for row in testing] # the first four colums are the evidence
y_testing = [row["label"] for row in testing] # the fifth colum is the result or prediction
predictions = model.predict(X_testing)# this function "predict" predicts the fifth colum (it will or not rain)

# Compute how well we performed
correct = 0
incorrect = 0
total = 0
for actual, predicted in zip(y_testing, predictions): #counting how many of the predictions are correct!
    total += 1
    if actual == predicted:
        correct += 1
    else:
        incorrect += 1

#Relation between two conditions for a rainy day.
data1= [i[1] for i in X_training]
data2= [i[2] for i in X_training]
data3= [i[3] for i in X_training]
color = [i[0] for i in y_training]
predict1= [i[1] for i in X_testing]
predict2= [i[2] for i in X_testing]
predict3= [i[3] for i in X_testing]
predict = [i[0] for i in y_testing]
#color = [i[4] for i in y_training]#this array contains both "1" and "0"
#PLOT IN 2D
#plt.scatter(data1,data2,s=30, c=color,cmap="bwr",marker=1)
#plt.scatter(predict1,predict2,s=30, c=predict, cmap= "PiYG",marker=2)
#plt.ylabel("skewness")
#plt.xlabel("variance")
#plt.title("GaussianNB")
#plt.show()

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
 
# Creating plot
ax.scatter3D(data1, data2, data3,  c=color,cmap="bwr",marker=1)
ax.scatter3D(predict1, predict2, predict3,  c=predict, cmap="PiYG",marker=2)
plt.title(f"{type(model).__name__}")
#ax.set_xlabel('variance', fontweight ='bold') #0
ax.set_xlabel('skewness', fontweight ='bold')#1
ax.set_ylabel('curtosis', fontweight ='bold')#2
ax.set_zlabel('entropy', fontweight ='bold')#3
plt.show()

#PLOT IN 3D
# Print results
print(f"Results for model {type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")


