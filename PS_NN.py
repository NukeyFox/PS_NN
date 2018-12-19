
import numpy as np
import scipy
import time
import os


def sigmoid(x):
#    return 1.0/(1+np.exp(-x/110))
    return np.where(x <= 0, 0, x^2)
def d_sigmoid(x):
    #return sigmoid(x/110)/(110*(1-sigmoid(x/110)))
    return np.where(x <= 0, 0, x)
    
class PS_NeuralNet(object):    
    def __init__(self,file1, file2, file3):
        if file1 == "":  
            self.weights1 = np.matrix(np.random.rand(4000,600))
            self.weights2 = np.matrix(np.random.rand(600,50))
            self.weights3 = np.matrix(np.random.rand(50,8))
        else:
            self.weights1 = np.load(file1)
            self.weights2 = np.load(file2)
            self.weights3 = np.load(file3)
        self.output = np.array([0,0,0,0,0,0,0,0])
    
    def feedforward(self,X):
        self.layer1 = sigmoid(np.dot(X,self.weights1)) 
        self.layer2 = sigmoid(np.dot(self.layer1,self.weights2)) #(1,700)*(600,50) = (1,50)
        self.output = sigmoid(np.dot(self.layer2,self.weights3)) #(1,50)*(50,8) = (1,8)
        return self.output
    
    def backprop(self,X,Y,O):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights3 = np.dot(self.layer2.T, (2*(O - Y) * d_sigmoid(O)).T)
        d_weights2 = np.dot(self.layer1.T,  (np.dot((2*(O - Y) * d_sigmoid(O)).T, self.weights3.T) * d_sigmoid(self.layer2.T)))
        d_weights1 = np.dot(X.T, (np.dot(np.dot((2*(O - Y) * d_sigmoid(O)).T, self.weights3.T), self.weights2.T )* d_sigmoid(self.layer1).T).T)
        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3
        
    def train(self,X,O):
        Y = self.feedforward(X)
        self.backprop(X,Y,O)
        print(np.sum(np.square(Y-O)))
    
    def predict(self,X):
        result = []
        output = self.feedforward(X)
        return output
    

