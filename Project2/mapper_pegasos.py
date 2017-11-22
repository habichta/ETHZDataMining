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
lambda_param = 0.001 # Regularization Parameter. This descides how strong we weight misclassifications. This needs to be tuned!
it_count = 0
min_max_scaler = preprocessing.scale

def transform(x_original):
    x = np.asarray(x_original,dtype=float)
    x = x.reshape(DIMENSION,1)
    #Arthur: Stochastic Gradient Descent is sensitive to feature scaling
    #Maybe we should transform the features and scale them to a range [0,1]:
    x = min_max_scaler(x) 
    return x


if __name__=="__main__":
   for line in sys.stdin:
        line = line.strip()
        (label, x_string) = line.split(" ", 1)
        label = int(label)
        x_original = np.fromstring(x_string, sep=' ')
        x = transform(x_original)  
        it_count = it_count + 1
	step_s = 1/(lambda_param*it_count)

	if(np.multiply(label,np.dot(w.transpose(),x))<1):
	    #Gradient
	    w = np.add(np.multiply(w,1-step_s*lambda_param),np.multiply(step_s,np.multiply(label,x)))
	else:
	    w = np.multiply(w,1-step_s*lambda_param)

	#Project
	w = np.multiply(w, min(1,1/(np.linalg.norm(w,ord=2)*math.sqrt(lambda_param))))

   w_opt = ' '.join(str(coeff) for coeff in w.flatten())
   key = "w_opt"
   print("%s\t%s" % (key, w_opt))

#Todo: look for bugs, tune regularization parameter (cross validation?)







