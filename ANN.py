# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')
import os
import sys
from concurrent.futures import ProcessPoolExecutor as ppe

os.chdir("C:\\Users\\Hilak\\Desktop\\INTERESTS\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)");
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, [13]].values

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

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

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
def softmax(z) :
    t = np.exp(z)
    return (t/np.sum(t,axis=0))
def softmax_prime(z):
    return softmax(z)*(1-softmax(z))
    

# A dictionary of our activation functions
PHI = {'sigmoid':sigmoid, 'relu':ReLU, 'lrelu':lReLU, 'tanh':tanh, 'softmax':softmax}

# A dictionary containing the derivatives of our activation functions
PHI_PRIME = {'sigmoid':sigmoid_prime, 'relu':ReLU_prime, 'lrelu':lReLU_prime, 'tanh':tanh_prime, 'softmax':softmax_prime}


class NeuralNet : 
    """
    This is a class for implementing Artificial Neural Networks. L2 and Droupout are the 
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
    
    8. B           : Biases of a pretrained neural network with same architecture.
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
        self.X_mini = None
        self.y_mini = None
        self.m_mini = None
        # List to store the cost of the model calculated during training
        self.cost = []
        # Stores the accuracy obtained on the test set.
        self.acc = 0
        # Activation function of each layer
        self.ac_funcs = ac_funcs
        self.loss = loss_func
        # Initialize the weights by provided method.
        if len(W) and len(B) :
            self.W = W
            self.B = B
        else : 
            if init_method=='gaussian': 
                self.W = [np.random.randn(self.n[nl], self.n[nl-1])*np.sqrt(2/self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.zeros((nl,1), 'float32') for nl in self.layers]
            elif init_method == 'random':
                self.W = [np.random.rand(self.n[nl], self.n[nl-1]) for nl in range(1,len(self.n))]
                self.B = [np.random.rand(nl,1) for nl in self.layers]
            elif init_method == 'zeros':
                self.W = [np.zeros((self.n[nl], self.n[nl-1]), 'float32') for nl in range(1,len(self.n))]
                self.B = [np.zeros((nl,1), 'float32') for nl in self.layers]
        self.vdw = [np.zeros(i.shape) for i in self.W]
        self.vdb = [np.zeros(i.shape) for i in self.B]
        self.sdw = [np.zeros(i.shape) for i in self.W]
        self.sdb = [np.zeros(i.shape) for i in self.B]
    
    def startTraining(self, batch_size, epochs, alpha, decay_rate, _lambda, keep_prob, beta1=0.9, beta2=0.999, interval=10, print_metrics = True, evaluate=False, X_test=None, y_test=None):
        """
        Start training the neural network. It takes the followng parameters : 
        
        1. batch_size : Size of your mini batch. Must be greater than 1.
        
        2. epochs     : Number of epochs for which you want to train the network.
        
        3. alpha      : The learning rate of your network.
        
        4. decay_rate : The rate at which you want to decrease your learning rate.

        5. _lambda    : L2 regularization parameter or the penalization parameter.

        6. keep_prob  : Python List. Dropout regularization parameter. The percentage of neurons to keep activated.
                        Eg - 0.8 means 20% of the neurons have been deactivated.
        
        7. beta1      : Momentum. default=0.9
        
        8. beta2      : RMSprop Parameter. default=0.999
        
        9. interval   : The interval between updates of cost and accuracy.
        """
        dataset_size = self.X.shape[1]
        k=1
        cost_val = 0
        for j in range(1, epochs+1):
            start = time.time()
            alpha = alpha/( 1 + decay_rate*(j-1) )
            for i in range(0, dataset_size-batch_size+1, batch_size):
                self.X_mini = self.X[:, i:i+batch_size]
                self.y_mini = self.y[:, i:i+batch_size]
                self.m_mini = self.y_mini.shape[1]
                self._miniBatchTraining(alpha, beta1, beta2, _lambda, keep_prob,k)
                aa = self.predict(self.X)
                if not k%interval:
                    if self.loss == 'b_ce':
                        aa_ = aa > 0.5
                        self.acc = np.sum(aa_ == self.y) / self.m
                        cost_val = self._cost_func(aa, _lambda)
                        self.cost.append(cost_val)
                    elif self.loss == 'c_ce':
                        aa_ = np.argmax(aa, axis = 0)
                        yy = np.argmax(self.y, axis = 0)
                        self.acc = np.sum(aa_==yy)/(self.m)
                        cost_val = self._cost_func(aa, _lambda)
                        self.cost.append(cost_val)
                if print_metrics:
                    sys.stdout.write(f'\rEpoch[{j}] {i+batch_size}/{self.m} : Cost = {cost_val:.4f} ; Acc = {(self.acc*100):.2f}% ; Time Taken = {(time.time()-start):.0f}s')
                    print("\n")
                k+=1
        if evaluate:
            print(f"For batch_size = {batch_size}, epochs = {epochs}, alpha = {alpha}, decay_rate = {decay_rate}, _lambda = {_lambda}, keep_prob = {keep_prob}")
            aa = self.predict(X_test)
            if self.loss == 'b_ce':
                aa_ = aa > 0.5
                acc = np.sum(aa_ == y_test) / X_test.shape[1]
                print(f"Accuracy on training set: {self.acc}")
                print(f"Accuracy on test set: {acc}\n")
            elif self.loss == 'c_ce':
                aa_ = np.argmax(aa, axis = 0)
                yy = np.argmax(y_test, axis = 0)
                acc = np.sum(aa_==yy)/(X_test.shape[1])
                print(f"Accuracy on training set: {self.acc}")
                print(f"Accuracy on test set: {acc}\n")
        
    
    def _miniBatchTraining(self, alpha, beta1, beta2, _lambda, keep_prob,i):
        epsilon = 1e-8
        z,a = self._feedForward(keep_prob)
        delta = self._cost_derivative(a[-1])
        for l in range(1,len(z)) : 
            delta_w = (1/self.m_mini)*(np.dot(delta, a[-l-1].T) + (_lambda)*self.W[-l])
            delta_b = (1/self.m_mini)*(np.sum(delta, axis=1, keepdims=True))
            self.vdw[-l] = (beta1*self.vdw[-l] + (1-beta1)*delta_w)
            vdw_corrected = self.vdw[-l]/(1 - beta1**i)
            self.vdb[-l] = (beta1*self.vdb[-l] + (1-beta1)*delta_b)
            vdb_corrected = self.vdb[-l]/(1 - beta1**i)
            self.sdw[-l] = (beta2*self.sdw[-l] + (1-beta2)*(delta_w**2))
            sdw_corrected = self.sdw[-l]/(1 - beta2**i)
            self.sdb[-l] = (beta2*self.sdb[-l] + (1-beta2)*(delta_b**2))
            sdb_corrected = self.sdb[-l]/(1 - beta2**i)
            delta = np.dot(self.W[-l].T, delta)*PHI_PRIME[self.ac_funcs[-l-1]](z[-l-1])
            self.W[-l] = self.W[-l] - (alpha)*(vdw_corrected/(np.sqrt(sdw_corrected)+epsilon))
            self.B[-l] = self.B[-l] - (alpha)*(vdb_corrected/(np.sqrt(sdb_corrected)+epsilon))
        delta_w = (1/self.m_mini)*(np.dot(delta, self.X_mini.T ) + (_lambda)*self.W[0])
        delta_b = (1/self.m_mini)*(np.sum(delta, axis=1, keepdims=True))
        self.vdw[0] = (beta1*self.vdw[0] + (1-beta1)*delta_w)
        vdw_corrected = self.vdw[0]/(1 - beta1**i)
        self.vdb[0] = (beta1*self.vdb[0] + (1-beta1)*delta_b)
        vdb_corrected = self.vdb[0]/(1 - beta1**i)
        self.sdw[0] = (beta2*self.sdw[0] + (1-beta2)*(delta_w**2))
        sdw_corrected = self.sdw[0]/(1 - beta2**i)
        self.sdb[0] = (beta2*self.sdb[0] + (1-beta2)*(delta_b**2))
        sdb_corrected = self.sdb[0]/(1 - beta2**i)
        self.W[0] = self.W[0] - (alpha)*(vdw_corrected/(np.sqrt(sdw_corrected)+epsilon))
        self.B[0] = self.B[0] - (alpha)*(vdb_corrected/(np.sqrt(sdb_corrected)+epsilon))
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
        z.append(np.dot(self.W[0], self.X_mini) + self.B[0])
        a.append(PHI[self.ac_funcs[0]](z[-1]))
        for l in range(1,len(self.layers)-1):
            z.append(np.dot(self.W[l], a[-1]) + self.B[l])
            # a.append(PHI[self.ac_funcs[l]](z[l]))
            _a = PHI[self.ac_funcs[l]](z[l])
            a.append( ((np.random.rand(*_a.shape) < keep_prob[l-1])*_a)/keep_prob[l-1] )
        z.append(np.dot(self.W[-1], a[-1]) + self.B[-1])
        a.append(PHI[self.ac_funcs[-1]](z[-1]))
        return z,a
    
    def _cost_func(self, a, _lambda):
        """ Binary Cross Entropy Cost Function """
        if self.ac_funcs[-1] == 'sigmoid':
            return ( (-1/self.m)*np.sum(np.nan_to_num(self.y*np.log(a+10e-8) + (1-self.y)*np.log(1-a+10e-8))) + (_lambda/(2*self.m))*np.sum([np.sum(i**2) for i in self.W]) )
        return ( (-1/self.m)*np.sum(np.nan_to_num(self.y*np.log(a))) + (_lambda/(2*self.m))*np.sum([np.sum(i**2) for i in self.W]) )
        
    def _cost_derivative(self, a) : 
        """ The derivative of cost w.r.t z """
        return (a-self.y_mini)
   
    @property
    def summary(self) :
        return self.cost, self.acc, self.W,self.B
    def __repr__(self) : 
        return f'<Neural Network at {id(self)}>'


class HyperParameterTuning:

    def __init__(self):
        pass

    def GridSearch(self, layers, X, y, ac_funcs, params, X_test, y_test):
        if __name__ == '__main__':
            models=[]
            with ppe(max_workers = len(params)) as pool:
                for param in params:
                    models.append(NeuralNet(layers, X, y, ac_funcs))
                    pool.submit(models[-1].startTraining, batch_size=param['batch_size'], epochs=param['epochs'], alpha=param['alpha'], decay_rate=param['decay_rate'], _lambda=param['_lambda'], keep_prob=param['keep_prob'], print_metrics=False, evaluate=True, X_test=X_test, y_test=y_test)

    def RandomizedGridSearch(self, layers, X, y, ac_funcs, params_range, nb_models, X_test, y_test):
    	if __name__ == '__main__':
	    	models = []
	    	params = []
	    	for i in range(nb_models):
	    		params.append({
	    			'batch_size' : int(np.round(np.random.rand()*(params_range['batch_size'][1]-params_range['batch_size'][0]) + params_range['batch_size'][0])),
	    			'epochs' : int(np.round(np.random.rand()*(params_range['epochs'][1]-params_range['epochs'][0]) + params_range['epochs'][0])),
	    			'alpha' : 10**(np.random.rand()*(np.log10(params_range['alpha'][1])-np.log10(params_range['alpha'][0])) + np.log10(params_range['alpha'][0])),
	    			'decay_rate' : 10**(np.random.rand()*(np.log10(params_range['decay_rate'][1])-np.log10(params_range['decay_rate'][0])) + np.log10(params_range['decay_rate'][0])),
	    			'_lambda' : 10**(np.random.rand()*(np.log10(params_range['_lambda'][1])-np.log10(params_range['_lambda'][0])) + np.log10(params_range['_lambda'][0])),
	    			'keep_prob' : [(np.random.rand()*(params_range['keep_prob'][1]-params_range['keep_prob'][0]) + params_range['keep_prob'][0]) for j in range(len(layers)-1)]
	    			})
	    	with ppe(max_workers = len(params)) as pool:
	    		for param in params:
	    			models.append(NeuralNet(layers, X, y, ac_funcs))
	    			pool.submit(models[-1].startTraining, batch_size=param['batch_size'], epochs=param['epochs'], alpha=param['alpha'], decay_rate=param['decay_rate'], _lambda=param['_lambda'], keep_prob=param['keep_prob'], print_metrics=False, evaluate=True, X_test=X_test, y_test=y_test)
        

    def __repr__(self):
        return f'<HPT at {id(self)}>'


# Hyperparameter Tuning for batch_size, epochs, alpha, decay_rate, _lambda, keep_prob
params = [
            {'batch_size':100,'epochs':20,'alpha':0.1,'decay_rate':0,'_lambda':0.5,'keep_prob':[0.86,0.9,1]},
            {'batch_size':200,'epochs':40,'alpha':0.1,'decay_rate':0.001,'_lambda':0.7,'keep_prob':[0.8,0.9,1]},
            {'batch_size':300,'epochs':60,'alpha':0.07,'decay_rate':0.001,'_lambda':0.3,'keep_prob':[0.9,0.9,1]},
            {'batch_size':400,'epochs':80,'alpha':0.05,'decay_rate':0,'_lambda':1,'keep_prob':[0.75,0.95,1]},
            {'batch_size':500,'epochs':100,'alpha':0.04,'decay_rate':0,'_lambda':0.8,'keep_prob':[0.5,0.95,1]},
            {'batch_size':600,'epochs':120,'alpha':0.03,'decay_rate':0,'_lambda':0.9,'keep_prob':[0.85,0.85,1]},
            {'batch_size':700,'epochs':150,'alpha':0.03,'decay_rate':0,'_lambda':1,'keep_prob':[0.8,0.9,1]},
            {'batch_size':800,'epochs':200,'alpha':0.02,'decay_rate':0,'_lambda':1.5,'keep_prob':[0.9,0.9,1]}
         ]

params_range = {'batch_size':[100, 1000],'epochs':[20,200],'alpha':[0.001, 0.02],'decay_rate':[0.000001, 0.0001],'_lambda':[0.1, 1.5],'keep_prob':[0.75, 1]}

hpt = HyperParameterTuning()
# hpt.GridSearch([16,16,1], X_train, y_train, ['relu','relu','sigmoid'], params, X_test, y_test)
hpt.RandomizedGridSearch([16,16,1], X_train, y_train, ['relu','relu','sigmoid'], params_range, 20, X_test, y_test)