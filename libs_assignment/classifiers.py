import numpy as np

class templatematch():
  def __init__( self, bovw ):
    self.bovw = bovw

  def fit( self, X ):
    self.templates = {}
    for k, v in X.items():
      h = self.bovw.predict( v )
      self.templates[k] = h / np.sum( h )

  def KL_divergence( self, P, Q, eps=0.000001 ):
    lP = np.log2( P + eps )
    lQ = np.log2( Q + eps )
    kl = np.sum( P*(lP-lQ) )
    return kl

  def predict( self, X ):
    h = self.bovw.predict( X )
    h = h / np.sum( h )
    scores = np.zeros( (1, len( self.templates )) )
    for i, (k, t) in enumerate( self.templates.items() ):
      scores[0, i] = self.KL_divergence( t, h )
    return np.argmin( scores ), scores
