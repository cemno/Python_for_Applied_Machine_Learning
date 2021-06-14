"""
  This file will hold our Gaussian based classses.
  1. Is the full MVG
  2. IS the diagonal MVG
  3. Is the class for a multiclass Gaussion mixture model.
"""

"""
  Import area
"""
import numpy as np
from sklearn.mixture import GaussianMixture
"""
  0. Base class for MVG and diagonal MVG written for you.
"""
# Your two MVG type classes will inherit from this class.
# In your classes you will also need a _precalculations method that isn't in the base class.
# In this instance it's not really necessary but it's good practice.
class Distribution:
  def __init__( self, ):
    self.name = "A distribution"

  def log_likelihood( self, X ):
    print( "Return the log-likelihood" )

  def train( self, X ):
    print( "Update and train the model" )


"""
  1. MVG
"""
# First create a MultivariateGaussian class, don't forget to inherit from Distribution.
# The base class provides an overview of the methods we will use here.
class MultivariateGaussian( Distribution ):
  # Create the __init__ function, you will also need to initialise the base class: super().__init__()
  # This class will also take two inputs: mu and sigma which default to an empty list each.
  # If both of these members are not empty you should run the _precalculations method
  # which we will code up next.
  def __init__(self, mu = [], sigma = []):
    super().__init__()
    self.mu = mu
    self.sigma = sigma
    if (not (self.sigma == []) and (not (self.mu == []))):
      self._precalculations()

  # When we perform the log likelihood calculation we need to calculate some values
  # including the Sigma^-1 and |Sigma| as you can see in the pdf. Along with these
  # values we will also precompute the constant values from the pdf.
  # Create a method called _precalculations with no inputs.
  def _precalculations(self):
    # How many dimensions do we have?
    n = self.mu.shape[1]

    # Calculate the inverse matrix using np.linalg.inv and store as a member
    self.inv_sigma = np.linalg.inv(self.sigma)

    # calculate the two constant values from the pdf.
    # the log determinant can be calculated by np.linalg.slogdeg()
    log_two_pi = -n / 2. * np.log(2 * np.pi)
    log_det = -0.5 * np.linalg.slogdet(self.sigma)[1]

    # now sum these two constants together and store them as a member.
    self.constant = log_two_pi + log_det

  # Next we will overwrite the log_likelihood method from the base class.
  def log_likelihood( self, X ):
    # get the shape of the data (m samples, n dimensions)
    m, n = X.shape

    # create an empty log likelihood output to the shape of m
    llike = np.zeros((m, ))

    # calculate the residuals X - mu
    resids = X - self.mu

    # iterate over the number of data points (m) in residuals and calculate the log likelihood for each.
    # equation in the pdf, using the members created in _precalculations.
    # Hopefully, you see the benefit of precalculating the constants and inverse.
    for i in range(m):
      llike[i] = self.constant - resids[i, :] @ self.inv_sigma @ resids[i,:].T

    # return the log likelihood values
    return llike


  # Now we will overwrite the train function.
  def train( self, X ):
    # get the shape of the data
    m,n = X.shape

    # step 1 estimate the mean values. X is of size (m,n) and take the sum over m samples.
    # then divide by the total number of samples.
    mu = np.sum(X, axis = 0)/float(m)
    mu = np.reshape(mu, (1,n))

    # Step 2 calculate the covariance matrix
    # residuals
    norm_X = X - mu

    # covariance n,n = (n,m @ m,n) / float( m )
    sigma = (norm_X.T @ norm_X) / float( m )

    # Assign class values and compute internals
    self.mu = mu
    self.sigma = sigma

    # step 3 precalcuate the internals for log likelihood
    self._precalculations()


"""
  2. Diagonal MVG
"""
# This is a homework exercise. It will follow the standard MVG so I won't comment as much.
# Again we will create a Class DiagonalMultivariateGaussian that inherits from Distribution.


  # Create the __init__ function exactly the same as above


  # Compute the precalculations but this is slightly different


    # calculate  the dimensions


    # inverse sigma is just 1/sigma diagonals


    # calculate the two constants


    # the determinant is sigma.diagonal().prod()


    # add them together to create a single constant



  # log likelihood is the same as MVG apart from when where I comment.


    # From the pdf we know that the residuals are squared.


    # Use the equation in the pdf to calculate the log likelihood. This is different
    # to the MVG version...


  # Calculate the statistics. In exactly the same way as MVG

    # step 1 estimate the mean values. X is of size (m,n) and take the sum over m samples.


    # Step 2 calculate the covariance matrix


    # Assign class values and compute internals



"""
  3. Multiple GMM class
"""
# import GaussianMixture from sklearn.mixture
# Create a class called MultiGMM we aren't inheriting in this case.
class MultiGMM():
  # Create the __init__ method with the number of mixtures as an input that we create a
  # member from. You should also instantiate gmms as empty dictionary members.
  def __init__(self, n_mixtures):
    self.n_mixtures = n_mixtures
    self.gmms = {}

  # fit method.
  # Input is a dictionary of keys (classes) and values (matrix(m,n))
  # We will iterate over the dictionary and create a GaussianMixture( number of mixtures )
  # model for each key. Where the GMM.fit( values )
  # You will need to import GaussianMixture from sklearn.mixture
  def fit(self, X):
    for k, v in X.items():
      self.gmms[k] = GaussianMixture(self.n_mixtures).fig(v)
      self.gmms[k].fit( v )

  # A handly little method for some classes is a reset method that resets the primary
  # members. In our case we will also input the number of mixtures.
  def rest(self, n_mixtures):
    self.n_mixtures = n_mixtures
    self.gmms = {}

  # And finally we will predict an input where the input is a matrix (m,n).
  # We will iterate through the gmm members and classify the matrix for each gmm member.
  # Create the predict method with input X
  def predict ( self, X ):
    # create a vector of scores (m of X, number of gmms)
    scores = np.zeroes( (X.shape[0], len( self.gmms) ) )

    # iterate over the gmms and use the score_samples function to calculate the similarity of each
    # point in X to the gmm. In this case we will use enumerate rather than having
    # an iterator that we manually add to to index into scores. In this case we will have:
    # for itr, (keys, values) in enumerate( gmms.items() ):
    # in this case enumerate returns an iterator integer and the keys and values as a tuple.
    for i, (k, g) in enumerate(self.gmms.items()):
      scores[:,i] = g.score_samples(X)

    # for i, g in enumerate( self.gmms ):
    #   scores[:,i] = g.score_samples( X )
    # Use argmax to classify the scores
    classify = np.argmax(scores, axis = 1)

    # return the classification score with the correct (m,1) dimensionality
    return classify.reshape((-1,1))