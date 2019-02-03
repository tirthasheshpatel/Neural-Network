# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore');
import os
import sys

# Importing and cleaning our dataset
os.chdir("C:\\Users\\Hilak\\Desktop\\INTERESTS\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)");
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_test, X_CV, y_test, y_CV = train_test_split(X, y, test_size = 0.5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.T
X_test = X_test.T
X_CV = X_CV.T

# ANN -- Hand Wooven
def sigmoid(z) : 
    return 1./(1 + np.exp(-z))
def sigmoid_prime(z) :
    return sigmoid(z)*(1-sigmoid(z))
def ReLU(z) : 
    return (z*(z > 0))
def ReLU_prime(z) :
    return 1*(z>=0)
def lReLU(z) : 
    return np.maximum(z/10,z)
def lReLU_prime(z) :
    z = 1*(z>=0)
    z[z==0] = 1/10
    return z

class NeuralNet : 
    def __init__(self, layers, X, y, ac_func='sigmoid', init_method='gaussian', loss_func='b_ce', W=np.array([]), B=np.array([])) : 
        self.layers = layers
        self.W = None
        self.B = None
        self.m = X.shape[1]
        self.n = [X.shape[0], *layers]
        self.X = X
        self.y = y
        self.cost = []
        self.acc = 0
        self.ac_func = ac_func
        self.loss = loss_func
        if len(W) and len(B) :
            self.W = W
            self.B = B
        else : 
            if init_method=='gaussian': 
                self.W = [np.random.randn(self.n[nl], self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.random.randn(nl,1) for nl in self.layers]
            elif init_method == 'random':
                self.W = [np.random.rand(self.n[nl], self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.random.rand(nl,1) for nl in self.layers]
            elif init_method == 'zeros':
                self.W = [np.zeros((self.n[nl], self.n[nl-1]), 'float32') for nl in range(1,len(self.n))]
                self.B = [np.zeros((nl,1), 'float32') for nl in self.layers]
    
    def startTraining(self, epochs, alpha, interval=100):
        start = time.time()
        for i in range(epochs+1) : 
            z,a = self._feedForward()
            cost_val = self._cost_func(a[-1])
            self.cost.append(cost_val)
            delta = self._cost_derivative(a[-1])
            for l in range(1,len(z)) : 
                delta_w = np.matmul(delta, a[-l-1].T)
                delta_b = sum(sum(delta))
                self.W[-l] = self.W[-l] - (alpha/self.m)*delta_w
                self.B[-l] = self.B[-l] - (alpha/self.m)*delta_b
                if self.ac_func == 'sigmoid' : 
                    delta = np.matmul(self.W[-l].T, delta)*sigmoid_prime(z[-l-1])
                elif self.ac_func == 'relu' : 
                    delta = np.matmul(self.W[-l].T, delta)*ReLU_prime(z[-l-1])
                elif self.ac_func == 'lrelu' : 
                    delta = np.matmul(self.W[-l].T, delta)*lReLU_prime(z[-l-1])
            if not i%interval :
                aa = self.predict(self.X)
                aa = aa > 0.5
                self.acc = sum(sum(aa == self.y)) / self.m
            sys.stdout.write(f'\rEpoch[{i}] : Cost = {cost_val:.2f} ; Acc = {(self.acc*100):.2f}% ; Time Taken = {(time.time()-start):.2f}s')
        print('\n')
        return None
    
    def predict(self, X_test) : 
        if self.ac_func == 'sigmoid' : 
            a = sigmoid(np.matmul(self.W[0], X_test) + self.B[0])
            for l in range(1,len(self.layers)):
                a = sigmoid(np.matmul(self.W[l], a) + self.B[l])
            return a
        elif self.ac_func == 'relu' : 
            a = ReLU(np.matmul(self.W[0], X_test) + self.B[0])
            for l in range(1,len(self.layers)):
                if l == (len(self.layers)-1) : 
                    a = np.matmul(self.W[l], a) + self.B[l]
                else : 
                    a = ReLU(np.matmul(self.W[l], a) + self.B[l])
            return a
        elif self.ac_func == 'lrelu' : 
            a = lReLU(np.matmul(self.W[0], X_test) + self.B[0])
            for l in range(1,len(self.layers)):
                if l == (len(self.layers)-1) : 
                    a = np.matmul(self.W[l], a) + self.B[l]
                else : 
                    a = lReLU(np.matmul(self.W[l], a) + self.B[l])
            return a
            
    
    def _feedForward(self):
        z = [];a = []
        if self.ac_func == 'sigmoid' : 
            z.append(np.matmul(self.W[0], self.X) + self.B[0])
            a.append(sigmoid(z[0]))
            for l in range(1,len(self.layers)):
                z.append(np.matmul(self.W[l], a[l-1]) + self.B[l])
                a.append(sigmoid(z[l]))
            return z,a
        elif self.ac_func == 'relu' :
            z.append(np.matmul(self.W[0], self.X) + self.B[0])
            a.append(ReLU(z[0]))
            for l in range(1,len(self.layers)):
                z.append(np.matmul(self.W[l], a[l-1]) + self.B[l])
                a.append(ReLU(z[l]))
            a[-1] = sigmoid(z[-1])
            return z,a
        elif self.ac_func == 'lrelu' :
            z.append(np.matmul(self.W[0], self.X) + self.B[0])
            a.append(lReLU(z[0]))
            for l in range(1,len(self.layers)):
                z.append(np.matmul(self.W[l], a[l-1]) + self.B[l])
                a.append(lReLU(z[l]))
            a[-1] = sigmoid(z[-1])
            return z,a
    
    def _cost_func(self, a):
        return ( (-1/self.m)*np.sum(np.nan_to_num(self.y*np.log(a) + (1-self.y)*np.log(1-a))) )

    def _cost_derivative(self, a) : 
        return a-self.y
   
    @property
    def summary(self) :
        return self.cost, self.acc, self.W,self.B
    def __repr__(self) : 
        return f'<UNDER CONST>'


# Testing our ANN
neural_net = NeuralNet([16,16,1], X_train, y_train, ac_func = 'sigmoid')
neural_net.startTraining(1000, 1, 100)
aa = neural_net.predict(X_test)
aa = aa > 0.5
acc = (sum(sum(aa == y_test)) / y_test.size)*100
print(f'Accuracy on test set : {acc}%')

# Plotting our results
metrics = neural_net.summary
plt.plot(range(len(metrics[0])), metrics[0])
plt.title('Cost')
plt.xlabel('Epochs')
plt.ylabel('Cost Value')
plt.show()

# Comparing with Fully Optimized Neural Network Library
from keras.models import Sequential
from keras.layers import Dense
X_train, X_test, X_CV = X_train.T, X_test.T, X_CV.T
classifier = Sequential()
classifier.add(Dense(input_dim=11, units = 16, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units = 16, kernel_initializer = "uniform", activation="relu"))
classifier.add(Dense(units = 16, kernel_initializer = "uniform", activation="relu"))
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 5, epochs = 10)
y_pred = classifier.predict(X_test)
y_pred = 1*(y_pred > 0.5)
test_acc = sum(sum(y_pred.T == y_test)) / y_test.size
print(f"Test set Accuracy : {test_acc*100}%")
X_train, X_test, X_CV = X_train.T, X_test.T, X_CV.T