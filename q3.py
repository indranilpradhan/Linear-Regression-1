import numpy as np
import pandas as pd
from numpy.random import RandomState
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

class Airfoil:
    def __init__(self, learning_rate = 0.0003,epochs = 100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def fit_test(self, X_test):
        self.X_test = X_test
    
    def fit_coef(self, coef):
        self.coef = coef
        
    def prediction(self, X_test, newcoeff):
        return self.X_test.dot(newcoeff)
      
    def cost_calculation(self, X_train, Y_train, Coefficient):
        leng = len(Y_train)
        sumcost = np.sum((self.X_train.dot(Coefficient) - self.Y_train) ** 2)
        cost = sumcost/(2 * leng)
        return cost
    
    def calculate_gradient(self, weights,X_train, Y_train):
        leng = len(self.Y_train)
        return self.X_train.T.dot(weights-self.Y_train) / leng
    
    def calculate_coefficients(self, coefficients, learning_rate, gradient):
        cal_grad = learning_rate * gradient
        penalty = coefficients - cal_grad
        return penalty
    
    def gradient_descent(self, X_train, Y_train, coefficients, learning_rate, epochs):
        costs = []
        for i in range(epochs):
            weights = self.X_train.dot(coefficients)
            gradient = self.calculate_gradient(weights, X_train, Y_train)
            coefficients = self.calculate_coefficients(coefficients, learning_rate, gradient)
            cost = self.cost_calculation(X_train, Y_train, coefficients)
            costs.append(cost)
        return coefficients, costs
    
    def train(self, path):
        df = pd.read_csv(str(path),header = None)
        df=(df-df.min())/(df.max()-df.min())
        train = df
        X_train,Y_train_temp = train.iloc[:,1:], train.iloc[:,-1:]
        Y_train = []
        for i in np.array(Y_train_temp):
            Y_train.append(i[0])
        X_train = np.array(X_train).astype('float')
        Y_train = np.array(Y_train).astype('float')
        
        self.fit(X_train,Y_train)
        
        coefficients = np.zeros(self.X_train.shape[1])

        newcoeff, cost = self.gradient_descent(self.X_train, self.Y_train, coefficients, self.learning_rate, self.epochs)
        self.fit_coef(newcoeff)
        
    def predict(self, path):
        df = pd.read_csv(str(path),header = None)
        train = df
        X_test = train.iloc[:,1:]
        X_test = np.array(X_test).astype('float')
        self.fit_test(X_test)
        Y_predict = self.prediction(self.X_test, self.coef)
        return Y_predict

