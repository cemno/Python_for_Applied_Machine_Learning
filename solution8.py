"""
  Today we will cover
  1. Multivariate Gaussian data creation
  2. Our own MVG class
  3. Our own diagonal MVG class
  4. Gaussian mixture models for colour classification
"""

"""
  ####### Import area
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as conf_mat
# our library
import libs.gaussian as pamlGaus

"""
  ####### Preamble
"""

ex01 = True
ex02 = True
ex03 = True
ex04 = False

"""
  ####### 1. Create data
"""
if ex01:
  # We have done this multiple times now so I won't comment very heavily
  # You'll need to import some stuff for this to work.
  # how many samples per distribution
  number_samples = 100
  # Create the two data arrays
  X0mu = np.array( [0.,0] )
  X0cv = np.array( [[2.,0],[0,2]] )
  X0 = np.random.multivariate_normal( X0mu, X0cv, number_samples )
  X1mu = np.array( [4.,1] )
  X1cv = np.array( [[1.,0.5],[0.5,1]] )
  X1 = np.random.multivariate_normal( X1mu, X1cv, number_samples )
  # Create the full data and labels based on these two different pieces of data
  X = np.vstack( (X0, X1) )
  labels = np.array( [0]*X0.shape[0] + [1]*X1.shape[0] )
  # plot everything so we can see what it looks like
  plt.figure()
  plt.scatter( X0[:,0], X0[:,1], c='cyan', marker='*', label='X0' )
  plt.scatter( X1[:,0], X1[:,1], c='yellow', marker=',', label='X1' )
  plt.title( 'X0 and X1' )
  plt.legend()
  plt.tight_layout()
  # plt.show() # what do you notice about this distribution of points?
  plt.close() # I close the figure so that it doesn't show up somewhere else

"""
  ####### 2. Multivariate Gaussian
"""
if ex02:
  # Make sure you have created the MVG class in gaussian.py and imported the library.
  # I imported my library as pamlGaus, you can call it something else if you'd like.
  # Once you have completed the MVG class:
  # Create a MVG object and train them individually on the X0, and X1 datasets
  mvg0 = pamlGaus.MultivariateGaussian()
  mvg0.train( X0 )
  mvg1 = pamlGaus.MultivariateGaussian()
  mvg1.train( X1 )
  # Now we will do something we haven't done before, and because our classes are simplistic
  # we have this option. We will look at self.mu and self.sigma from each of the mvg objects and
  # visually compare them to the original datasets. We do this by object.mu and object.sigma
  print( 'MVG' )
  print( 'Original X0 parameters\n', X0mu, '\n', X0cv )
  print( 'Calculated mvg0 parameters\n', mvg0.mu, '\n', mvg0.sigma )
  print( 'Original X1 parameters\n', X1mu, '\n', X1cv )
  print( 'Calculated mvg1 parameters\n', mvg1.mu, '\n', mvg1.sigma )
  # Now we will use these two distributions to classify the full data. Obviously in practice
  # we would have a training and evaluation set but in this case we are simply going to
  # classify the training set based on the models.
  # To do this we need both X0 and X1 concatenated together if you haven't already done that.
  # We will then use the log_likelihood function from both the MVGs to classify the data
  # resulting in a matrix of [m,2] (2 as we have 2 models).
  loglike = np.zeros( (X.shape[0], 2) )
  loglike[:,0] = mvg0.log_likelihood( X )
  loglike[:,1] = mvg1.log_likelihood( X )
  # Okay now we need to classify based on the maximum response in the matrix of log_likelihood.
  # In this case the closer a point is the the distribution the greater its log likelihood,
  # or the better it fits the distribution. Calculate that now
  classified = np.argmax( loglike, axis=1 )
  # Now we need a way to decide how good this classification is. For this we will use sklearns.metrcs
  # accuracy_score function. It compares the labels with the predictions and can give either
  # a normalised (0-1) score or a count of correctly classified points. You will need
  # the labels and the classified vector.
  acc = accuracy_score( labels, classified )
  print( 'Accuracy of the MVGs is:', acc )

"""
  ####### 3. Diagonal multivariate Gaussaian
"""
if ex03:
  # Make sure you have created the diagonal MVG class in gaussian.py and imported the library.
  # Basically you will do exactly the same thing you did in the previous example for standard
  # MVG.
  dmvg0 = pamlGaus.DiagonalMultivariateGaussian()
  dmvg0.train( X0 )
  dmvg1 = pamlGaus.DiagonalMultivariateGaussian()
  dmvg1.train( X1 )
  # It's a good idea to compare the diagonal MVG values to what you saw in the standard MVG
  # have a look at the difference between the two techniques (it should be limited for this data)
  print( 'Diagonal MVG' )
  print( 'Original X0 parameters\n', X0mu, '\n', X0cv )
  print( 'Calculated dmvg0 parameters\n', dmvg0.mu, '\n', dmvg0.sigma )
  print( 'Original X1 parameters\n', X1mu, '\n', X1cv )
  print( 'Calculated dmvg1 parameters\n', dmvg1.mu, '\n', dmvg1.sigma )
  # Calculate the log likelihood
  loglike = np.zeros( (X.shape[0], 2) )
  loglike[:,0] = dmvg0.log_likelihood( X )
  loglike[:,1] = dmvg1.log_likelihood( X )
  print( loglike[0,:] )
  print( loglike[100,:] )
  # Classify based on the log likelihood scores
  classified = np.argmax( loglike, axis=1 )
  # Calculate the accuracy of this technique.
  # Does it compare to full MVG?
  acc = accuracy_score( labels, classified )
  print( 'Accuracy of the diagonal MVGs is:', acc )
  # How accurate is the diagonal compared to the full version?

"""
  ####### 4. Gaussian mixture model classification
"""
if ex04:
  # Once again you need to complete the class in gaussian.py. Once you have done that
  # you can start on this section.
  # The first thing we need to do is read in the images from colour_snippets as a
  # dictionary of vectors. Here are the steps:
  # 1. Create an empty dictionary to store the vectors in
  # 2. sorted( os.listdir() ) all the directories in colour_snippets and iterate around them.
  # 3. sorted( os.listdir() ) all the files in the current directory iterations, you will
  #     also need os.path.join( root, <colour directory> )
  # 4. Read in the current image using imread and os.path.join( root, <colour directory>, <filename> )
  # 5. Vectorise the image using reshape()
  # 6. Load the vectorised image into the correct dictionary entry for that colour.
  #     d[colour] = vectorised image. You will also need to concatenate the new SAME
  #     colour images into the dictionary. I recommend using if statements with a flag that
  #     indicates that this is the first file.
  root = 'data/colour_snippets' # my root directory
  data = {} # empty dictionary
  for c in sorted( os.listdir( root ) ): # iterate around the root directory for the colour directories
    firstfile = True  # A flag for instantiating the dictionary
    for i in sorted( os.listdir( os.path.join( root, c ) ) ): # iterate around the files in the directory
      rgb = imread( os.path.join( root, c, i ) )  # load the image
      rgb = rgb.reshape( (-1,3) ) # convert to a feature vector rather than an image
      if firstfile: # that flag to say if we have already instantiated
        data[c] = rgb # Create the dictionary entry with the colour
        firstfile = False # make sure we don't do this again we need to concatenate after this.
      else:
        data[c] = np.vstack( (data[c], rgb) )
  # Next we need to create training and testing dictionaries.
  # We have used a manual technique based on random numbers previously but now
  # I will introduce a helpful function from sklearn.model_selection called train_test_split
  # You'll need to import that. The steps to follow:
  # 1. a) Create an empty dictionary for training
  #    b) A evaluation integer (starting at 0) to use both for a evaluation label set and
  #       for creating the evaluation data  points (like we have done before if it > 0 <do this> else)
  # 2. Specify the evaluation image size as 0.3
  # 3. For each dictionary entry create a training/evaluation split using train_test_split()
  # 4. Assign the appropriate training data to the appropriate dictionary key
  # 5. Append to a label set (0->6) and a data set (np.vstack()) for the evaluation
  Xt = {} # empty dictionary
  evalit = 0
  testsize = 0.3  # split size
  for k, v in data.items(): # for loop over the full data
    t, e = train_test_split( v, test_size=testsize )  # split the data
    Xt[k] = t # assign the training set
    if evalit == 0:
      Xe = e
      label = np.ones( (e.shape[0], 1) )*evalit
      evalit += 1
    else:
      Xe = np.vstack( (Xe, e) )
      label = np.vstack( (label, np.ones( (e.shape[0], 1) )*evalit) )
      evalit += 1
  # Now we will train the gmms based on the class you created.
  # 1. Create the base object with 3 n_mixtures
  # 2. Pass the training dictionary to the fit function.
  gmm = pamlGaus.MultiGMMs( 3 )
  gmm.fit( Xt )
  # let's make sure we have 6 gmms inside here
  print( gmm.gmms )
  # Okay now let's classifiy the image
  cls = gmm.predict( Xe )
  # Now we need to see how accurate we are, for this we will use a confusion matrix.
  # The confusion matrix is an N*N matrix where N is the number of gmms in the class.
  # You will need to import confusion_matrix from sklearn.metrics.
  # confusion_matrix( ground truth, predictions ) outputs the associated confusion matrix.
  # Give it a go, you'll want to normalize the confusion matrix too.
  cm = conf_mat( label, cls, normalize='true' )
  # Let's print the confusion matrix and discuss it a bit.
  # First you will need to turn this flag on: np.set_printoptions(suppress=True)
  # This stops numpy from outputting as scientific notation, which can make reading
  # the confusion matrix difficult.
  np.set_printoptions( suppress=True )
  print( cm )
  # So each row indicates the ground truth and the columns indicate the predictions.
  # And each element of the diagonal states the overall accuracy of the predictions.
  # i.e. how well is it blue and we say that it's blue, compared to how often do we
  # confuse blue with another colour.
  # Using these diagonals let's output the overall accuracy of the system...
  print( np.sum( cm.diagonal() )/float( cm.diagonal().shape[0] ) )
  # Can you think of a way that we might improve this? What colour space are we using?
