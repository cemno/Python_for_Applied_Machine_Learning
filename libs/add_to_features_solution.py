import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.cluster import KMeans

# Extract the hog information from a file path, coverts to greyscale.
# Input: filname
#        hog orientation
#        hog pixel per
#        hog cells per
#       ALSO: visualize=False and feature_vector=False
# Output: feature vector of size (-1, orientations)
def extract_hog_matrix( f, o, p, c ):
  # convert to greyscale
  gry = rgb2gray( imread( f ) )
  # calculate the HOG representation
  feat = hog( gry,
              orientations=o,
              pixels_per_cell=p,
              cells_per_block=c,
              visualize=False,
              feature_vector=False )
  return feat.reshape( (-1, o) )


# Now let's convert the training dictionary into a single feature matrix for training
# the kmeans classifier. We can use the extraction function we just created to save time.
# Remember the output will be a numpy array of size (-1,orientations)
# Input is a dictionary of lists of filenames
#       orientations, pixel per, cells per
# output will be the full feature vector for kmeans
def extract_full_hog_features( X, o, p, c ):
  # iterate over the dictionary
  firstval = True
  for k, v in X.items():
    for f in v:
      # extract hog from the file
      feat = extract_hog_matrix( f, o, p, c )
      # concatenate the features
      if firstval:
        fullvec = feat
        firstval = False
      else:
        fullvec = np.vstack( (fullvec, feat) )
  # Return the full vecetor
  return fullvec

# Extract the hog information per class for the average histogram calculator.
# In this case the input will be dictionary of classes with a list of file locations for each.
# The output will be a full feature vector per class.
def extract_class_hog_features( X, o, p, c):
  classvec = {}
  for k, v in X.items():
    firstval = True
    for f in v:
      # extract hog from the file
      feat = extract_hog_matrix( f, o, p, c )
      # now per class concatenate the features
      if firstval:
        classvec[k] = feat
        firstval = False
      else:
        classvec[k] = np.vstack( (classvec[k], feat) )
  return classvec


# A kmeans based BoVW classifier
# Create the BoVW class
class BoVW():
  # Initialise with the number of clusters and store the member
  def __init__( self, num_clusters ):
    self.num_clusters = num_clusters
  # the fit function to fit our kmeans to a feature vector of size (-1, dimensions)
  def fit( self, X ):
    # create and fit the kmeans object
    self.kmeans = KMeans( self.num_clusters )
    self.kmeans.fit( X )
  # The predict function will return a histogram based on the kmeans algorithm and the number of clusters
  def predict( self, X ):
    fv = self.kmeans.predict( X )
    # print( fv.shape ); sys.exit( 1 )
    # you can use np.histogram to get the histogram just be careful of the output...
    h, _ = np.histogram( fv, bins=self.num_clusters )
    return h
