"""
  This is a file that will do some basic lbp stuff in a library file.
  Import your libraries directly below this string.
"""
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern as lbp


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
