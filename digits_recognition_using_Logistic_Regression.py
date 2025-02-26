
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# loading digits dataset from sklearn
dataset = load_digits()

print(dataset.keys())

# sample images
for i in range(3):
    plt.imshow(dataset.images[i], cmap="gray")
    plt.show()

# value of the sample images
print(dataset.target[0:3])

# splitting data for train and test (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(dataset.data, dataset.target, test_size=0.2 ,  random_state=42)

print(len(X_train))
print(len(Y_test))

# loading the model
model = LogisticRegression(max_iter=10000)

# training the model
model.fit(X_train, Y_train)

# score
print(model.score(X_test, Y_test))

# testing on a random image
sample = dataset.data[97]

print(model.predict([sample]))
print(dataset.target[97])

from sklearn.metrics import confusion_matrix

# printing the confusion matrix
Y_predicted = model.predict(X_test)
c_matrix = confusion_matrix(Y_test, Y_predicted)
sns.heatmap(c_matrix, annot=True, cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


