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



# This is the basic lbp file. You will fill out the rest.
def extract_lbp_feature(file,  # the string location of the image
                        radius=1,  # the radius about which to look
                        npoints=8,  # the number of points around the radius.
                        nbins=128,  # for plotting the histogram
                        range_bins=(0, 255)):  # the range for plotting the histogram
    # Read in the file. Because this is different to the template.py you will have to
    # Import all your libraries that you want to use again.

    rgb = imread(file)
    gry = rgb2gray(rgb)

    # Convert it to greyscale.

    # Now you need to import local_binary_pattern from skimage.features in the import area.
    # In much the same way that HOG does lbp has some basic parameters.
    # R is the radius around a central pixels that it will extend. In this case 1 means
    # that it will extend out from the center pixel 1 pixels (or there abouts), as the number
    # increases we scan a larger area.
    # P - number of points, to look at on the radius, for a radius of 1 it makes sense to look
    # at 8 pixels, why do you think?
    # Using this information we can use the local_binary_pattern in the following way:
    # features = local_binary_pattern( input image, R=radius, P=npoints )
    # When you are finished print the features shape.
    feat = lbp(gry, R = radius, P = npoints)
    # Let's go back to template.py and run this function.

    # So you have run it, what do you notice? The output is 140*140 which was the size of
    # our input image. So each pixel has a value associated with it! These values for the
    # standard lbp range from 0->255 based on a combination of pixels. For more information
    # on exactly how this is computed see: https://towardsdatascience.com/face-recognition-how-lbph-works-90ec258c3d6b
    # For here though, we are just going to assume that for each pixel in the image we have
    # a value on the range [0,255]. Next we need to create a feature vector based on these
    # values. We will use np.histogram to classify our values.
    # np.histogram takes as input the features you want to bin, the number of bins and the range
    # of the input.
    # By default we will classify the features into 128 bins on the range 0-255.
    feats, edges = np.histogram( feat, bins = nbins, range = range_bins)

    # now we need to return feats and edges and then go back to the template.py
    return feat, edges

# Addition from week 10

# Extract the hog information from a file path, coverts to greyscale.
# Input: filname
#        hog orientation
#        hog pixel per
#        hog cells per
#       ALSO: visualize=False and feature_vector=False
# Output: feature vector of size (-1, orientations)
def extract_hog_matrix(f, o, p, c):
  # convert to greyscale
  gry = rgb2gray(imread(f))
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

# Extract the hog information per class for the average histogram calculator.
# In this case the input will be dictionary of classes with a list of file locations for each.
# The output will be a full feature vector per class.
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
