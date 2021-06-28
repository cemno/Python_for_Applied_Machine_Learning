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


  # convert to greyscale


  # calculate the HOG representation


  # return a feature of shape (-1, orientations)



# Now let's convert the training dictionary into a single feature matrix for training
# the kmeans classifier. We can use the extraction function we just created to save time.
# Remember the output will be a numpy array of size (-1,orientations)
# Input is a dictionary of lists of filenames
#       orientations, pixel per, cells per
# output will be the full feature vector for kmeans
def extract_full_hog_features(X, o, p, c):
  # iterate over the dictionary
  for t, v in X.items():
    for f in v:
      # extract hog from the file


      # concatenate the features


  # Return the full vecetor



# Extract the hog information per class for the average histogram calculator.
# In this case the input will be dictionary of classes with a list of file locations for each.
# The output will be a full feature vector per class.



# A kmeans based BoVW classifier
# Create the BoVW class


  # Initialise with the number of clusters and store the member


  # the fit function to fit our kmeans to a feature vector of size (-1, dimensions)


    # create and fit the kmeans object


  # The predict function will return a histogram based on the kmeans algorithm and the number of clusters


    # you can use np.histogram to get the histogram just be careful of the output...
