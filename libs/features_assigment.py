"""
  This is a file that will do some basic lbp stuff in a library file.
  Import your libraries directly below this string.
"""
from skimage.feature import local_binary_pattern as lbp
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.cluster import KMeans

def extract_lbp_feature(file,  # the string location of the image
                        radius=1,  # the radius about which to look
                        npoints=8,  # the number of points around the radius.
                        nbins=128,  # for plotting the histogram
                        range_bins=(0, 255)):  # the range for plotting the histogram
    rgb = file
    gry = rgb2gray(rgb)
    feat = lbp(gry, R = radius, P = npoints)
    feats, edges = np.histogram( feat, bins = nbins, range = range_bins)
    return feat, edges



# Extract the hog information from a file path, coverts to greyscale.
# Input: filname
#        hog orientation
#        hog pixel per
#        hog cells per
#       ALSO: visualize=False and feature_vector=False
# Output: feature vector of size (-1, orientations)
def extract_hog_matrix(f, o, p, c):
  # convert to greyscale
  gry = rgb2gray(f)
  # calculate the HOG representation
  feat = hog(gry, orientations = o,
             pixels_per_cell = p,
             cells_per_block = c,
             visualize = False,
             feature_vector = False)
  # return a feature of shape (-1, orientations)
  return feat.reshape((-1, o)) # m,8, h/p*w/p*c*0 (8 is default in script 10)

# Now let's convert the training dictionary into a single feature matrix for training
# the kmeans classifier. We can use the extraction function we just created to save time.
# Remember the output will be a numpy array of size (-1,orientations)
# Input is a dictionary of lists of filenames
#       orientations, pixel per, cells per
# output will be the full feature vector for kmeans
def extract_full_hog_features(X, o, p, c):
  # iterate over the dictionary
  firstfile = True
  for t, v in X.items():
    for f in v:
      # extract hog from the file
      feat  = extract_hog_matrix(f, o, p, c)
      # concatenate the features
      if firstfile:
        fullvec = feat
        firstfile = False
      else:
        fullvec = np.vstack((fullvec, feat))
  # Return the full vecetor
  return fullvec


# A kmeans based BoVW classifier
# Create the BoVW class
class BoVW():
  # Initialise with the number of clusters and store the member
  def __init__(self, num_clusters):
    self.num_clusters = num_clusters

  # the fit function to fit our kmeans to a feature vector of size (-1, dimensions)
  def fit(self, X):
    # create and fit the kmeans object
    self.kmeans = KMeans(self.num_clusters)
    self.kmeans.fit(X)

  # The predict function will return a histogram based on the kmeans algorithm and the number of clusters
  def predict (self, X):
    fv = self.kmeans.predict(X)
    # you can use np.histogram to get the histogram, just be careful of the output...
    h, _ = np.histogram(fv, bins = self.num_clusters)
    return h
