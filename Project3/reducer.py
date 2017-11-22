#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans

buffer = []
kmeans = MiniBatchKMeans(n_clusters = 100, batch_size = 100, init = "k-means++")
  
for line in sys.stdin:
    line = line.strip()
    feature = np.fromstring(line,dtype=np.float64,sep=" ").astype(np.float64)
    buffer.append(feature)
    
    if len(buffer) == 100:
      kmeans.partial_fit(buffer)
      buffer=[]
  
if len(buffer) > 0:
    kmeans.partial_fit(buffer)


centroids = kmeans.cluster_centers_

for i in range (0,100):
	#np.set_printoptions(precision=10)
	#np.set_printoptions(suppress=True)
	np.set_printoptions(linewidth=500)
	w_opt = ' '.join(str(coeff) for coeff in centroids[i,:].flatten())
	print w_opt
