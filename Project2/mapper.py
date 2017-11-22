#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import math
from sklearn import preprocessing

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)   # The classes that we are trying to predict.

#Initialize coefficient vector
w = np.ones([DIMENSION,1],dtype=np.float64)
lambda_param = .5 # Regularization Parameter. This descides how strong we weight misclassifications. This needs to be tuned!
it_count = 0
min_max_scaler = preprocessing.MinMaxScaler()

def grad_hinge_loss(y,w,x):
    #Hinge loss not differentiable for w'x = 0, use Subgradients
    temp = np.multiply(y,np.dot(w.transpose(),x))
    if(temp < 1):
       return np.multiply(-1,np.multiply(y,x))
    else:
        return 0

def next_step_hinge(w,step_size,x,y):
    #Next step with gradient
    w_next = np.subtract(w,np.multiply(step_size,grad_hinge_loss(y,w,x)))
    #Project
    return np.multiply(w_next, min(1,1/(np.linalg.norm(w_next,ord=2)*math.sqrt(lambda_param))))



def transform(x_original):
    x = np.asarray(x_original,dtype=np.float64)
    x = x.reshape(DIMENSION,1)
    #Arthur: Stochastic Gradient Descent is sensitive to feature scaling
    #Maybe we should transform the features and scale them to a range [0,1]:
    x = min_max_scaler.fit_transform(x) 
    return x



for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  
        it_count = it_count + 1
        w = next_step_hinge(w,(1/math.sqrt(it_count)),x,label)
    
w_opt = ' '.join(str(coeff) for coeff in w.flatten())
key = "w_opt"
print("%s\t%s" % (key, w_opt))

#Todo: look for bugs, tune regularization parameter (cross validation?)







