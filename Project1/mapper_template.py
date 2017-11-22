#!/local/anaconda/bin/python
# IMPORTANT: leave the above line as is.

import numpy as np
import sys

# Parameters used in this script
max_value_in_shingles = 20010   # Shingles contain values in the range [0,max_value_in_shingles-10]
                                # (values get temporarily shifted to [10, max_value_in_shingles in the algorithm)
num_hash_functions = 960 		# Number of rows in the signature matrix
band_size = 16                  # Number of rows in one band from the signature matrix
num_bands = 60               	# Number of bands in the signature matrix (such that band_size*num_bands = num_hash_functions)


# VERY IMPORTANT:
# Make sure that each machine is using the
# same seed when generating random numbers for the hash functions.
np.random.seed(seed=42)


# Random values used for the linear hashing of the shingles  ( h(x) = a*x + b mod n )
n_in_hash_signature_matrix = 20011                     # A prime number n greater than max_value_in_shingles (used for hashing)
b_in_has_signature_matrix = 0
ai_in_hash_signature_matrix =  \
    np.random.randint(max_value_in_shingles/4,         # The a_i values used for linear hashing (i=1,...,num_hash_functions)
                      max_value_in_shingles,
                      size=num_hash_functions)

# Random values used for the linear hashing of the bands of the signature matrix
n_in_hash_bands = 32452843                             # Very large prime number to reduce false collisions
ai_in_hash_bands =  \
    np.random.randint(n_in_hash_bands*10+1,            # The a_i values used for the band hashing (i=1,...,num_hash_functions)
                      n_in_hash_bands*100,
                      size=num_hash_functions)
for ai in ai_in_hash_bands:                            # If (ai % n_in_has_bands == 0), we have h(x) = b for all x! (Very bad)
    while ai % n_in_hash_bands == 0 :
        ai = np.random.randint(max_value_in_shingles*10+1,
                              max_value_in_shingles*100)

b_in_hash_bands = np.random.randint(10, max_value_in_shingles,
                                    size=num_bands)    # The b values used for the band hasing (i=1,...,num_bands)

# Calculates the column
def construct_column_from_shingles(column_signature_matrix, source_shingles):
    for shingle in source_shingles:
        for row in range(num_hash_functions):
            hash_value = ai_in_hash_signature_matrix[row]*(shingle + 10) % n_in_hash_signature_matrix  # h_i(x) = a_i*(x+ 10) mod n
            if hash_value < column_signature_matrix[row]:
                column_signature_matrix[row] = hash_value

def hash_bands(column_from_the_signature_matrix):
    hash_values = np.zeros(num_bands, dtype=np.int64)
    for band in range(num_bands):
        ax = np.dot(column_signature_matrix[(band_size*band):(band_size*(band+1))],   # calculates the sum over a_i*x_i for the band
                    ai_in_hash_bands[(band_size*band):(band_size*(band+1))])
        hash_values[band] = (ax + b_in_hash_bands[band]) % n_in_hash_bands
    return hash_values.astype(int)              # Return the int-version of the array

if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        video_id = int(line[6:15])
        shingles = np.fromstring(line[16:], dtype=int, sep=" ").astype(int)

        # Constructs one column (i.e. 1 video) from the signature matrix (min-hashing of the shingles)
        column_signature_matrix =  np.full(num_hash_functions, n_in_hash_signature_matrix, dtype=np.int64)
        construct_column_from_shingles(column_signature_matrix, shingles)

        # Calculates a hash value for each band in the signature matrix
        hash_values = hash_bands(column_signature_matrix)

        # For each video and hash_band the format of the output is the following:
        # band_number hash_bucket_number \TAB VIDEO_ID hash_values[0] .... hash_values[num_bands-1] shingles[0]...shingles[singles.size()-1]
        shingles = np.sort(shingles)                                 # Shingles get sorted (in ascending order)
        shingles = ' '.join(str(shingle) for shingle in shingles)    # Formatted for output
        for band_number in range(num_bands):
            print("%s %s \t%s %s" %
                    (band_number,
                     hash_values[band_number],
                     video_id,
                     shingles))
