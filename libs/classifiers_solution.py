import numpy as np

# KL divergence based template matching.
class templatematch():
  def __init__( self, bovw ):
    self.bovw = bovw

  # Input: X as a dictionary of classes with a feature vector in each
  #       needs to be of the same requirements as the bovw classifier input to init
  # Will train a template (histogram) for each class.
  def fit( self, X ):
    self.templates = {}
    for k, v in X.items():
      h = self.bovw.predict( v )
      self.templates[k] = h / np.sum( h )

  # KL divergence member
  # Inputs are the template and the image histogram
  # output the kl score
  def KL_divergence( self, P, Q, eps=0.000001 ):
    lP = np.log2( P + eps )
    lQ = np.log2( Q + eps )
    kl = np.sum( P*(lP-lQ) )
    return kl

  # prediction per image.
  # X is an image feature vector
  # outputs the best score based on the templates.
  def predict( self, X ):
    # get the histogram of the input
    print( X.shape )
    h = self.bovw.predict( X )
    h = h / np.sum( h )
    scores = np.zeros( (1, len( self.templates )) )
    for i, (k, t) in enumerate( self.templates.items() ):
      scores[0, i] = self.KL_divergence( t, h )
    return np.argmin( scores ), scores
