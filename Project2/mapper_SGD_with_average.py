#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import math
from sklearn import preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier

DIMENSION = 400  # Dimension of the original data.
NUM_RANDOM_FEATURES = 5000
CLASSES = (-1, +1)   # The classes that we are trying to predict.

#Initialize coefficient vector
it_count = 0
min_max_scaler = preprocessing.scale

def transform(x_original):
    #x = np.asarray(x_original,dtype=float)
    #x = x.reshape(DIMENSION, 1)
    #Arthur: Stochastic Gradient Descent is sensitive to feature scaling
    #Maybe we should transform the features and scale them to a range [0,1]:
    x = min_max_scaler(x_original)
    return np.append(x, 1)


if __name__=="__main__":

    SGD = SGDClassifier(alpha=0.0000001, average=1000, fit_intercept=False)
    #SGD = PassiveAggressiveClassifier(fit_intercept=False)
    x_data = []
    y_label = []
    for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')

        x = transform(x_original)


        #if (it_count < 50000):
        SGD.partial_fit(x, np.array([label]), np.array([-1, 1]))
        #else:
        #    x_data.append(x)
        #    y_label.append(label)

        it_count = it_count + 1

        #if (it_count % 60000 == 0):
        #    break

        #if (it_count % 1000 == 0):
        #    print(it_count)


    w = SGD.coef_
    #print(SGD.score(x_data, y_label))



    w_opt = ' '.join(str(coeff) for coeff in w.flatten())
    key = str(it_count)
    print("%s\t%s" % (key, w_opt))


#Todo: look for bugs, tune regularization parameter (cross validation?)



