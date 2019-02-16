# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore');
import os
import sys

# Importing our dataset
os.chdir("C:/Users/Hilak/Desktop/INTERESTS/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression");
training_set = pd.read_csv("Social_Network_Ads.csv");

# Splitting our dataset into matrix of features and output values.
X = training_set.iloc[:, 1:4].values
y = training_set.iloc[:, 4].values

# Encoding our object features.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
X[:,0] = le_x.fit_transform(X[:,0])
ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()

# Performing Feature scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X[:,2:4] = ss.fit_transform(X[:, 2:4])

# Splitting the dataset into train and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
X_train = X_train.T
X_test = X_test.T

def sigmoid(z) :
    """ Reutrns the element wise sigmoid function. """
    return 1./(1 + np.exp(-z))
def sigmoid_prime(z) :
    """ Returns the derivative of the sigmoid function. """
    return sigmoid(z)*(1-sigmoid(z))
def ReLU(z) : 
    """ Reutrns the element wise ReLU function. """
    return (z*(z > 0))
def ReLU_prime(z) :
    """ Returns the derivative of the ReLU function. """
    return 1*(z>=0)
def lReLU(z) : 
    """ Reutrns the element wise leaky ReLU function. """
    return np.maximum(z/100,z)
def lReLU_prime(z) :
    """ Returns the derivative of the leaky ReLU function. """
    z = 1*(z>=0)
    z[z==0] = 1/100
    return z
def tanh(z) :
    """ Reutrns the element wise hyperbolic tangent function. """
    return np.tanh(z)
def tanh_prime(z) : 
    """ Returns the derivative of the tanh function. """
    return (1-tanh(z)**2)

# A dictionary of our activation functions
PHI = {'sigmoid':sigmoid, 'relu':ReLU, 'lrelu':lReLU, 'tanh':tanh}

# A dictionary containing the derivatives of our activation functions
PHI_PRIME = {'sigmoid':sigmoid_prime, 'relu':ReLU_prime, 'lrelu':lReLU_prime, 'tanh':tanh_prime}


class NeuralNet : 
    """
    This is a class for making Artificial Neural Networks. L2 and Droupout are the 
    default regularization methods implemented in this class. It takes the following parameters:
    
    1. layers      : A python list containing the different number of neurons in each layer.
                     (containing the output layer)
                     Eg - [64,32,16,16,1]
                
    2. X           : Matrix of features with rows as features and columns as different examples.
    
    3. y           : Numpy array containing the ouputs of coresponding examples.
    
    4. ac_funcs    : A python list containing activation function of each layer.
                     Eg - ['relu','relu','lrelu','tanh','sigmoid']
    
    5. init_method : Meathod to initialize weights of the network. Can be 'gaussian','random','zeros'.
    
    6. loss_func   : Currently not implemented
    
    7. W           : Weights of a pretrained neural network with same architecture.
    
    8. W           : Biases of a pretrained neural network with same architecture.
    """
    def __init__(self, layers, X, y, ac_funcs, init_method='gaussian', loss_func='b_ce', W=np.array([]), B=np.array([])) :
        """ Initialize the network. """
        # Store the layers of the network
        self.layers = layers
        # ----
        self.W = None
        self.B = None
        # Store the number of examples in the dataset as m
        self.m = X.shape[1]
        # Store the full layer list as n
        self.n = [X.shape[0], *layers]
        # Save the dataset
        self.X = X
        # Save coresponding output
        self.y = y
        # List to store the cost of the model calculated during training
        self.cost = []
        # Stores the accuracy obtained on the test set.
        self.acc = 0
        # Activation function of each layer
        self.ac_funcs = ac_funcs
        self.loss = loss_func
        # Inittialize the weights by provided methods if not provided.
        if len(W) and len(B) :
            self.W = W
            self.B = B
        else : 
            if init_method=='gaussian': 
                self.W = [np.random.randn(self.n[nl], self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.zeros((nl,1), 'float32') for nl in self.layers]
            elif init_method == 'random':
                self.W = [np.random.rand(self.n[nl], self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.random.rand(nl,1) for nl in self.layers]
            elif init_method == 'zeros':
                self.W = [np.zeros((self.n[nl], self.n[nl-1]), 'float32') for nl in range(1,len(self.n))]
                self.B = [np.zeros((nl,1), 'float32') for nl in self.layers]
    
    def startTraining(self, epochs, alpha, _lambda, keep_prob=0.5, interval=100):
        """
        Start training the neural network. It takes the followng parameters : 
        
        1. epochs : Number of epochs for which you want to train the network.
        
        2. alpha  : The learning rate of your network.
        
        3. _lambda : L2 regularization parameter or the penalization parameter.
        
        4. keep_prob : Dropout regularization parameter. The probability of neurons to deactivate.
                       Eg - 0.8 means 20% of the neurons have been deactivated.
        
        5. interval : The interval between updates of cost and accuracy.
        """
        start = time.time()
        for i in range(epochs+1) : 
            z,a = self._feedForward(keep_prob)
            delta = self._cost_derivative(a[-1])
            for l in range(1,len(z)) : 
                delta_w = np.dot(delta, a[-l-1].T) + (_lambda)*self.W[-l]
                delta_b = np.sum(delta, axis=1, keepdims=True)
                delta = np.dot(self.W[-l].T, delta)*PHI_PRIME[self.ac_funcs[-l-1]](z[-l-1])
                self.W[-l] = self.W[-l] - (alpha/self.m)*delta_w
                self.B[-l] = self.B[-l] - (alpha/self.m)*delta_b
            delta_w = np.dot(delta, self.X.T ) + (_lambda)*self.W[0]
            delta_b = np.sum(delta, axis=1, keepdims=True)
            self.W[0] = self.W[0] - (alpha/self.m)*delta_w
            self.B[0] = self.B[0] - (alpha/self.m)*delta_b
            if not i%interval :
                aa = self.predict(self.X)
                if self.loss == 'b_ce':
                    aa = aa > 0.5
                    self.acc = sum(sum(aa == self.y)) / self.m
                    cost_val = self._cost_func(a[-1], _lambda)
                    self.cost.append(cost_val)
                elif self.loss == 'c_ce':
                    aa = np.argmax(aa, axis = 0)
                    yy = np.argmax(self.y, axis = 0)
                    self.acc = np.sum(aa==yy)/(self.m)
                    cost_val = self._cost_func(a[-1], _lambda)
                    self.cost.append(cost_val)
            sys.stdout.write(f'\rEpoch[{i}] : Cost = {cost_val:.2f} ; Acc = {(self.acc*100):.2f}% ; Time Taken = {(time.time()-start):.2f}s')
        print('\n')
        return None
    
    def predict(self, X_test) :
        """ Predict the labels for a new dataset. Returns probability. """
        a = PHI[self.ac_funcs[0]](np.dot(self.W[0], X_test) + self.B[0])
        for l in range(1,len(self.layers)):
            a = PHI[self.ac_funcs[l]](np.dot(self.W[l], a) + self.B[l])
        return a
            
    
    def _feedForward(self, keep_prob):
        """ Forward pass """
        z = [];a = []
        z.append(np.dot(self.W[0], self.X) + self.B[0])
        a.append(PHI[self.ac_funcs[0]](z[-1]))
        for l in range(1,len(self.layers)):
            z.append(np.dot(self.W[l], a[-1]) + self.B[l])
            # a.append(PHI[self.ac_funcs[l]](z[l]))
            _a = PHI[self.ac_funcs[l]](z[l])
            a.append( ((np.random.rand(_a.shape[0],1) < keep_prob)*_a)/keep_prob )
        return z,a
    
    def _cost_func(self, a, _lambda):
        """ Binary Cross Entropy Cost Function """
        return ( (-1/self.m)*np.sum(np.nan_to_num(self.y*np.log(a) + (1-self.y)*np.log(1-a))) + (_lambda/(2*self.m))*np.sum([np.sum(i**2) for i in self.W]) )

    def _cost_derivative(self, a) : 
        """ The derivative of cost w.r.t z """
        return (a-self.y)
   
    @property
    def summary(self) :
        return self.cost, self.acc, self.W,self.B
    def __repr__(self) : 
        return f'<Neural Network at {id(self)}>'
    
# Initializing our neural network
neural_net_sigmoid = NeuralNet([32,16,1], X_train, y_train, ac_funcs = ['relu','relu','sigmoid'])
# Staring the training of our network.
neural_net_sigmoid.startTraining(5000, 0.01, 0.2, 0.5, 100)
# Predicting on new dataset using our trained network.
preds = neural_net_sigmoid.predict(X_test)
preds = preds > 0.5
acc = (sum(sum(preds == y_test)) / y_test.size)*100
# Accuracy (metric of evaluation) obtained by the network.
print(f'Test set Accuracy ( r-t-s ) : {acc}%')

# Plotting our cost vs epochs relationship
sigmoid_summary = neural_net_sigmoid.summary
plt.plot(range(len(sigmoid_summary[0])), sigmoid_summary[0], label='Sigmoid Cost')
plt.title('Cost')
plt.show()

# Comparing our results with the library keras.
from keras.models import Sequential
from keras.layers import Dense
X_train, X_test = X_train.T, X_test.T
classifier = Sequential()
classifier.add(Dense(input_dim=4, units = 32, kernel_initializer="uniform", activation="relu"))
classifier.add(Dense(units = 16, kernel_initializer = "uniform", activation="relu"))
# classifier.add(Dense(units = 16, kernel_initializer = "uniform", activation="tanh"))
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
classifier.fit(X_train, y_train, batch_size = 1, epochs = 30)
y_pred = classifier.predict(X_test)
y_pred = 1*(y_pred > 0.5)
test_acc = sum(sum(y_pred.T == y_test)) / y_test.size
print(f"Test set Accuracy : {test_acc*100}%")
X_train, X_test = X_train.T, X_test.T
