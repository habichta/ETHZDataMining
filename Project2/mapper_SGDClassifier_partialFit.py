#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
import math
from sklearn import preprocessing
from sklearn import linear_model

DIMENSION = 400  # Dimension of the original data.
CLASSES = (-1, +1)  # The classes that we are trying to predict.

#Initialize coefficient vector
min_max_scaler = preprocessing.MinMaxScaler()

lambda_param = 0.001 # Regularization Parameter. This descides how strong we weight misclassifications. This needs to be tuned!
clf = linear_model.SGDClassifier(alpha=lambda_param, average=False, class_weight=None, epsilon=0.1,
        eta0=0.0, fit_intercept=True, l1_ratio=0.15,
        learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.5, random_state=None, shuffle=True,
        verbose=0, warm_start=False)

def transform(x):
    x = min_max_scaler.fit_transform(x) 
    #x = preprocessing.scale(x) 
    return x


if __name__=="__main__":
   for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = [int(label)]
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  
        
	clf.partial_fit(x,label,classes = np.unique(CLASSES))
        
	
  
   w_opt = ' '.join(str(coeff) for coeff in clf.coef_.flatten())
   key = "w_opt"
   print("%s\t%s" % (key, w_opt))

#Todo: look for bugs, tune regularization parameter (cross validation?)







