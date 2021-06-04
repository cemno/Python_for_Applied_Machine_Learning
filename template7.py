"""
  Today we will cover
  1. Create clustering data
  2. Clustering - kmeans (build a class)
  3. Clustering - spectral clustering (internal library)
  4. Sliding window approach (Sobel edge)
"""

"""
  ####### Import area
"""

import numpy as np


"""
  ####### Preamble
"""

ex01 = False
ex02 = False
ex03 = False
ex04 = True

"""
  ####### 1. Create data
"""
# if ex01:
  # We will first create multivariate_normal data using the means and covariances
  # listed in the pdf.

  # now create the data and associated label sets from these subsets
  # D0 = [X0, X1]; using np.vstack


  # L0 = [0, 0, ..., 1,1,...]; using np.zeros, np.ones and np.vstack


  # D1 = [X0, X1, X2]; using np.concatenate along axis 0


  # L0 = [0, 0, ..., 1, 1,..., 2, 2, ...]; using np.concatenate and np.zeros, np.ones, np.ones*2


  # now let's do the scatter plots of the two different subsets. What do we need to import?
  # First let's plot D0 which is comprised of X0 and X1, use the scatter plot and don't forget
  # to change the colour of each plot...


  # Okay now we have our data for exercise 2 and 3.

"""
  ####### 2. Our KMeans class
"""
# if ex02:
  # Now we will create a KMeans class based on the algorithm in the pdf and the template below.
  # First let's create a class called KMeans
  # I would recommend for the assignment that you would put this in a module.

    # Next we need to create the __init__ function that takes as input the number of
    # clusters (n_clusters), and the max iterations (imax) set to 100 as default.
    # We could add a distance metric here too, do you know what it would do?


      # instantiate the inputs


    # Now let's create a the Euchlidean distance calculator (euchlid) that takes some data (X)
    # and a center value (self.C[c]) as input. This is based on (sum( (X-C)^2 ))^(1/2.) where the resulting vector
    # will have the same number of columns as the input X.


    # Next is the main part of the code, this is based on the algorithm in the pdf.
    # See if you can work it out from the sudo code supplied. But call the function "fit"


      # first we need to randomly create the cluster centers.
      # random dpoint selection


      # Now we need to iterate around the EM algorithm for the number of self.imax


        # create an empty data matrix


        # calculate the distances per center.


        # assign the data to one of the centroids. Remember we want the minimum distance,
        # between the datapoint and the Centroid.


        # Just in case we want to use the distance metric later let's calculate the
        # total distance of the new assignments to the it's assigned center.


        # Now the final step, let's update the self.C locations. We will use the mean
        # of the assigned points to that cluster.



    # Finally let's create a predict method too. This is basically just the distance
    # calculation, and assignment of an input matrix X


      # create an empty distance matrix


      # calculate the distances


      # return the assignments.



  # Let's use this class to cluster some data (D0 from exercise 1) with 4 clusters to start with
  # and an imax of zero (we only randomly assign centers).
  # Create the object


  # let's plot what this looks like
  # but first we want to know the  unique values in D0a so we aren't constantly having
  # to change the label values, you'll need np.unique


  # now scatter plt based on the predictions


  # Now what's wrong with what we did? We fit and predicted on the same  data.
  # Go back and Fit with D0 and predict with D1... How does that look?
  # Now we need to evaluate this,  for that we will use
  # from sklearn.metrics import completeness_score as skcs
  # Which is a metric designed expressly for clustering.
  # You will need to reshape the L vectors to be np.shape = (N,)



"""
  ####### 3. Spectral Clustering with sklearn
"""
# if ex03:
  # This exercise is just a simple homework exercise to get you familiar with reading the
  # sklearn documentation. We are after Spectral clustering (urls are in the pdf). This
  # is just another method of clustering and actually can use KMeans. We will use this method
  # of clustering with the "discretize" variable. You'll find that in the documentation.
  # You'll also need to import the appropriate library, I'll let you look it up.
  # Create the object


  # Fit and predict the data (D0 or D1 whichever)


  # Plot the result


  # Calculate the metric.



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


  # Now let's use subplots to plot the horizonta, vertical, and magnitude images vertically.


  # Now let's compare that to the inbuilt skimage.filters sobel version.


  # show our version versus this version in a vertical subplot.
