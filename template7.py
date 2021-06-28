"""
  Today we will cover
  1. Create clustering data
  2. Clustering - kmeans (build a class)
  3. Clustering - spectral clustering (internal library)
  4. Sliding window approach (Sobel edge)
"""
import sklearn.cluster

"""
  ####### Import area
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import completeness_score as skcs

"""
  ####### Preamble
"""

ex01 = True
ex02 = True
ex03 = True
ex04 = True

"""
  ####### 1. Create data
"""
if ex01:
  # We will first create multivariate_normal data using the means and covariances
  # listed in the pdf.
  X0 = np.random.multivariate_normal([0,0], [[1,0], [0,1]], 100)
  X1 = np.random.multivariate_normal([4,1], [[1,0], [0,1]], 100)
  X2 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 100)
  # now create the data and associated label sets from these subsets
  # D0 = [X0, X1]; using np.vstack
  D0 = np.vstack([X0, X1])
  D1 = np.vstack([X0, X1, X2])
  # L0 = [0, 0, ..., 1,1,...]; using np.zeros, np.ones and np.vstack
  L0 = np.vstack([np.zeros(100), np.ones(100)])
  # D1 = [X0, X1, X2]; using np.concatenate along axis 0
  D1 = np.concatenate([X0, X1, X2], axis = 0)

  # L0 = [0, 0, ..., 1, 1,..., 2, 2, ...]; using np.concatenate and np.zeros, np.ones, np.ones*2
  L1 = np.concatenate([np.zeros(100), np.ones(100), np.ones(100)*2])

  # now let's do the scatter plots of the two different subsets. What do we need to import?
  # First let's plot D0 which is comprised of X0 and X1, use the scatter plot and don't forget
  # to change the colour of each plot...
  plt.figure()
  plt.scatter(X0[:,0], X0[:,1], c = 'red', label = "X0")
  plt.scatter(X1[:,0], X1[:,1], c = 'green', label = "X1")
  #plt.show()
  # Okay now we have our data for exercise 2 and 3.
  pass
  sklearn.cluster.KMeans
"""
  ####### 2. Our KMeans class
"""

if ex02:
  # Now we will create a KMeans class based on the algorithm in the pdf and the template below.
  # First let's create a class called KMeans
  # I would recommend for the assignment that you would put this in a module.
  class KMeans:
    # Next we need to create the __init__ function that takes as input the number of
    # clusters (n_clusters), and the max iterations (imax) set to 100 as default.
    # We could add a distance metric here too, do you know what it would do?
    def __init__(self, n_cluster, imax = 100):
      # instantiate the inputs
      self.n_cluster = n_cluster
      self.imax = imax
    # Now let's create a the Euchlidean distance calculator (euchlid) that takes some data (X)
    # and a center value (self.C[c]) as input. This is based on (sum( (X-C)^2 ))^(1/2.) where the resulting vector
    # will have the same number of columns as the input X.
    def euclid_dist(self, X, c):
      diff = X - c # [N,2] - [1,2] = [N,2]
      sqrd = diff ** 2 # [N,2]
      smmd = np.sum(sqrd, axis = 1) # [N,2] + axis = 1 = [N]
      return np.sqrt(smmd) # [N]

    # Next is the main part of the code, this is based on the algorithm in the pdf.
    # See if you can work it out from the sudo code supplied. But call the function "fit"
    def fit(self, X):
      # first we need to randomly create the cluster centers.
      # random dpoint selection
      cstart = np.random.randint(0,X.shape[0], self.n_cluster)
      self.C = X[cstart,:]

      # Now we need to iterate around the EM algorithm for the number of self.imax
      for _ in range(self.imax): # underscore for just running and no variable
        # create an empty data matrix
        dist = np.zeros((X.shape[0], self.n_cluster)) # [N, n_cluster]

        # calculate the distances per center.
        for i in range(self.n_cluster):
          dist[:,i] = self.euclid_dist(X, self.C[i])

        # assign the data to one of the centroids. Remember we want the minimum distance,
        # between the datapoint and the Centroid.
        X_assign = np.argmin(dist, axis=1)

        # Just in case we want to use the distance metric later let's calculate the
        # total distance of the new assignments to the it's assigned center.
        dist_metric = np.sum(dist[:,X_assign])
        print(dist_metric)
        # Now the final step, let's update the self.C locations. We will use the mean
        # of the assigned points to that cluster.
        for i in range(self.n_cluster):
          c_samples = X[X_assign==i]
          self.C[i] = np.mean(c_samples, axis=0)


    # Finally let's create a predict method too. This is basically just the distance
    # calculation, and assignment of an input matrix X
    def predict(self, X):
      # create an empty distance matrix
      dist = np.zeros((X.shape[0], self.n_cluster))  # [N, n_cluster]
      # calculate the distances
      for i in range(self.n_cluster):
        dist[:, i] = self.euclid_dist(X, self.C[i])
      # return the assignments.
      return np.argmin(dist, axis=1)

  # Let's use this class to cluster some data (D0 from exercise 1) with 4 clusters to start with
  # and an imax of zero (we only randomly assign centers).
  # Create the object
  kmeans = KMeans(2, imax=1)
  X = D0
  kmeans.fit(X)
  Y = kmeans.predict(X)
  # let's plot what this looks like
  # but first we want to know the  unique values in D0a so we aren't constantly having
  # to change the label values, you'll need np.unique
  Yu = np.unique(Y)
  # now scatter plt based on the predictions
  plt.figure()
  for i in Yu:
    plt.scatter(X[Y==i, 0],X[Y==i, 1], label ='Y{}'.format(i))
    plt.title('X clustering')
    plt.legend()
  plt.show()
  plt.close()
  # Now what's wrong with what we did? We fit and predicted on the same  data.
  # Go back and Fit with D0 and predict with D1... How does that look?
  # Now we need to evaluate this,  for that we will use
  # from sklearn.metrics import completeness_score as skcs
  # Which is a metric designed expressly for clustering.
  # You will need to reshape the L vectors to be np.shape = (N,)
  print(D0.shape)
  print(L0)
  accuracy = skcs(L0.reshape((-1,)), Yu)
  print("The accuracy of clustering was " + repr(round(accuracy*100)) + "%.")


"""
  ####### 3. Spectral Clustering with sklearn
"""
if ex03:
  # This exercise is just a simple homework exercise to get you familiar with reading the
  # sklearn documentation. We are after Spectral clustering (urls are in the pdf). This
  # is just another method of clustering and actually can use KMeans. We will use this method
  # of clustering with the "discretize" variable. You'll find that in the documentation.
  # You'll also need to import the appropriate library, I'll let you look it up.
  # Create the object
  spectral_cluster = SpectralClustering(n_clusters=2, assign_labels='discretize')

  # Fit and predict the data (D0 or D1 whichever)
  Y = spectral_cluster.fit_predict(D0)

  # Plot the result
  plt.figure()
  for label in Y:
    plt.scatter(D0[Y==label,0], D0[Y==label,1])
  plt.show()

  # Calculate the metric.
  accuracy = skcs(L0.reshape((-1,)), Y)
  print("The accuracy of clustering was " + repr(round(accuracy*100)) + "%.")


"""
  ####### 4. Sobel edge detection - sliding window
"""
# if ex04:
  # First we are going to be loading and saving images, so we need to import something.
  # Now load the image from week 5 irregular.jpg and convert it to a greyscale image (you
  # will need another library for that).


  # Now we need a base (zeros) horizontal and vertical image of the same size as the gry scale image


  # Next we need to create the two kernels we will use - see the pdf.


  # Now we will slide over the input image (greyscale). But keep in mind we aren't padding
  # the image at this point so for a 3*3 kernel where do we need to start iterating from,
  # and where do we need to stop iterating at so that we don't throw out of bound errors?
  # Now iterate over the image and update the horizontal and vertical image pixel by multiplying
  # the kernel by a snippet of the image.


  # calculate the output magnitude of the image.


  # Now let's use subplots to plot the horizontal, vertical, and magnitude images vertically.


  # Now let's compare that to the inbuilt skimage.filters sobel version.


  # show our version versus this version in a vertical subplot.
