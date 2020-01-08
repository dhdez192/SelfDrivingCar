# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 18:55:54 2020

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn import datasets
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Function that plots the NN decision boundary
def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)


np.random.seed(0)

n_pts = 500 # number of points for the data
# grabbing the data set and setting them into X and y
X, y = datasets.make_circles(n_samples=n_pts, random_state = 123, noise = 0.1, factor = 0.2)

#print(X)
#print(y)

# Function will plot the data set based on the clusters.
plt.scatter(X[y==0, 0], X[y==0, 1]) # where X[y==0,0] plots all the x values and X[y==0,1] plots all y values
plt.scatter(X[y==1,0], X[y==1,1])

# Creating a NN to classify the model
model = Sequential()
model.add(Dense(4, input_shape = (2,), activation='sigmoid')) # creating a layer that has 4 nodes and 2 inputs
model.add(Dense(1, activation='sigmoid'))   # Creating the output layer of the NN
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics=['accuracy'])              # compile the model using Adam 
model.fit(x=X , y=y, verbose = 1, batch_size = 20, epochs=100, shuffle='true')           # function trains model to fit the data

#plotting the accuracy of the NN per epoch
#plt.plot(h.history['accuracy'])
#plt.xlabel('epoch')
#plt.legend(['accuracy'])
#plt.title('accuracy')
#
##plotting the loss function
#plt.plot(h.history['loss'])
#plt.xlabel('epoch')
#plt.legend(['loss'])
#plt.title('loss')

# using the plot decision boundary over the dataset
#plot_decision_boundary(X,y,model)
#plt.scatter(X[:n_pts,0], X[:n_pts,1])
#plt.scatter(X[n_pts:,0], X[n_pts:,1])

# predict where a new point will lie in the data set
plot_decision_boundary(X,y,model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 0.1     # create a point x
y = 0.75    # create a point y
point = np.array([[x,y]])   # save two point as an array to pass through as an argument
prediction = model.predict(point)   # calculate the prediction value
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("Prediciton is: ", prediction)