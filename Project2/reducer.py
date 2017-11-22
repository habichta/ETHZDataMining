#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import logging
import sys
import numpy as np

w = np.zeros([1,400],dtype=np.float64)
models = 0

for line in sys.stdin:
    line = line.strip()
    key, rest = line.split("\t")
    rest_int = np.fromstring(rest, dtype=np.float64, sep=" ").astype(np.float64)
    w = w + rest_int
    models=models+1

w = w/models	
w_opt = ' '.join(str(coeff) for coeff in w.flatten())
print("%s" % (w_opt))
	
    
