"""
  This is a file that will do some basic lbp stuff in a library file.
  Import your libraries directly below this string.
"""
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern as lbp
import numpy as np
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


def extract_hog_matrix(img, o, p, c, visualize = False):
  # convert to greyscale
  gry = rgb2gray(img)
  # calculate the HOG representation
  if visualize:
    feat, map = hog(gry, orientations=o,
               pixels_per_cell=p,
               cells_per_block=c,
               visualize=True,
               feature_vector=False)
    plt.figure
    plt.subplot(211)
    plt.imshow(gry)
    plt.subplot(212)
    plt.imshow(map)
    plt.show()
  else:
    feat = hog(gry, orientations=o,
               pixels_per_cell=p,
               cells_per_block=c,
               visualize=False,
               feature_vector=False)
  # return a feature of shape (-1, orientations)
  return feat.reshape((-1, o))

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
  # Return the full vector
  return fullvec

def extract_class_hog_features(X, o, p, c):
    classvec = {}
    for t, v, in X.items():
        firstfile = True
        for f in v:
            feat = extract_hog_matrix(f, o, p, c)
            if firstfile:
                classvec[t] = feat
                firstfile = False
            else:
                classvec[t] = np.vstack((classvec[t], feat))
    return classvec

# kmeans based BoVW classifier
# Createing the BoVW class
class BoVW():
  # Initialise with the number of clusters
  def __init__(self, num_clusters):
    self.num_clusters = num_clusters

  # the fit function to fit kmeans to a feature vector of size (-1, dimensions)
  def fit(self, X):
    # create and fit the kmeans object
    self.kmeans = KMeans(self.num_clusters)
    self.kmeans.fit(X)

  # The predict function will return a histogram based on the kmeans algorithm and the number of clusters
  def predict (self, X):
    fv = self.kmeans.predict(X)
    h, _ = np.histogram(fv, bins = self.num_clusters)
    return h
