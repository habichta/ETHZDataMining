import numpy.random
import numpy as np
import sys

alpha = 0.3

articles_indices = {}        # The indices of the articles in the articles_collection. (A dictionary)
num_articles = 0
features = 0                # The feature vector (user features) which we use to update our model (see the update-Function).
num_total_features = 6      # The dimension of our "feature" vector
vector_b = np.empty         # Corresponds to the collection of all vector b_x from Slide 16 of the Lecture Slides (Lecture 12)
vector_b_tmp = np.empty     # We update vector_b only periodically (because we also update matrix_M only periodically)

matrix_M = np.empty         # Corresponds to the collection of all matrices M_x from Slide 16 of the Lecture Slides (Lecture 12)
matrix_M_inv = np.empty
num_updates = np.empty      # The inverse matrix of each article gets only updated periodically (to save time).
chosen_arrayindex = 0       # The Index of the array index corresponding to the chosen article. Needed by "matrix_M" and "vector_b" from the function "update" and modified by the function "reccomend".


def set_articles(dictionary):
    global num_articles, vector_b, articles_indices, matrix_M, matrix_M_inv, num_updates, vector_b_tmp

    keys = dictionary.keys()
    num_articles = len(keys)
    for i in range(num_articles):
        articles_indices[keys[i]] = i           # A dictionary for the "article->chosen_arrayindex" Mapping.

    # Initializes the vector_b and the matrix_M.
    vector_b = np.zeros((num_articles,num_total_features))
    matrix_M = np.zeros((num_articles, num_total_features, num_total_features))     # Collection of all matrices M_x.
    matrix_M_inv = np.zeros((num_articles, num_total_features, num_total_features)) # Collection of the inverse-Matrices
    num_updates = np.zeros((num_articles))

    # Fills the diagonal of matrix_M with ones.
    for i in range(num_articles):
        matrix_M[i,:,:] = np.eye(num_total_features, num_total_features)
        matrix_M_inv[i,:,:] = np.eye(num_total_features, num_total_features)


def update(reward):         # See Slides 16 from the Lecture Slides 12
    global features, chosen_arrayindex, matrix_M_inv, vector_b_tmp, vector_b
    if (reward != -1):      # Needed because of the "evaluator.py" file. Our result on the server is worse without this line.
        num_updates[chosen_arrayindex] += 1
        vector_b[chosen_arrayindex,:] += features*reward
        matrix_M[chosen_arrayindex,:,:] += numpy.outer(features, features)

        # Updates matrix_M_inv and vector_b only if the current article has been updated 10 times since the last update of matrix_M_inv and vector_b.
        if (num_updates[chosen_arrayindex] == 10):
            num_updates[chosen_arrayindex] = 0
            matrix_M_inv[chosen_arrayindex, :, :] = np.linalg.inv(matrix_M[chosen_arrayindex,:,:])



def reccomend(time, user_features, articles):
    global chosen_arrayindex
    global features

    features = np.array(user_features)
    chosen_arrayindex = -1    # The index in "matrix_M" and "vector_b" corresponding the article "chosen_article".
    chosen_article = -1       # The ID of the article which is "best" according to the UCB-criterion.
    max_UCB = -sys.maxint-1                         # The maximal UCB value calculated until now.

    # Loop over the possible articles (given by the function argument "articles").
    for i in range(len(articles)):                             # I want to put here "range(len(articles))", but that seems to be way too slow.
        #if (not articles[i] in articles_indices):  # Only necessary if the program is run on the local computer. (Necessary, because some articles from "log.txt" are not given in the "article.txt" file)
        #    continue                               # ""
        #else:                                      # ""
            articles_array_index = articles_indices[articles[i]]

            # Calculating UCB_x (where x = articles_array_index)    (See Slide 16 from the Lecture Slides 12)
            #matrix_M_2d = np.linalg.inv(np.array(matrix_M[articles_array_index,:,:]))
            matrix_M_2d = matrix_M_inv[articles_array_index,:,:]
            w_x = numpy.dot(matrix_M_2d,vector_b[articles_array_index,:].T)
            a = alpha*np.sqrt(numpy.dot(numpy.dot(features,matrix_M_2d), features))
            UCB_x = numpy.dot(features, w_x) + a

            # Is the "UCB_x" value corresponding to articles[i] bigger than our previously biggest?
            if (UCB_x > max_UCB):
                chosen_arrayindex = articles_array_index
                chosen_article = articles[i]
                max_UCB = UCB_x
    #return numpy.random.choice(articles, size=1)
    return chosen_article
