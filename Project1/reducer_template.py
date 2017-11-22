#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

last_key = None
key_count = 0
duplicates = []
shngls = []

def jaccard(x,y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def print_duplicates(videos):
    #unique = np.unique(videos)
    for i in xrange(len(videos)):
        for j in xrange(i + 1, len(videos)):
            similarity = jaccard(shngls[i],shngls[j])
            if similarity >= 0.9:
            	print "%d\t%d" % (min(videos[i], videos[j]),
                              	  max(videos[i], videos[j]))



for line in sys.stdin:
    line = line.strip()
    key, rest = line.split("\t")
    rest_int = np.fromstring(rest, dtype=int, sep=" ").astype(int)
    video_id = rest_int[0]
    shngls.append(rest_int[1:])
	
    if last_key is None:
        last_key = key

    if key == last_key:
        duplicates.append(video_id)
    else:
        # Key changed (previous line was k=x, this line is k=y)
        print_duplicates(duplicates)
        duplicates = [video_id]
	shngls = []
	shngls.append(rest_int[1:])
        last_key = key

if len(duplicates) > 0:
    print_duplicates(duplicates)
