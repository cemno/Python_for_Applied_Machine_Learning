"""
  Today we will cover
  1. HOG based BoVW classification using MLPs
  2. LBP based classification using MLPs
  3. Combination of HOG-BoVW and LBP for classification using MLPs

  This practical will very closely follow last weeks SVM practical. So you can copy and
  paste a lot of the functions.
"""

"""
  ####### Import area
"""
import argparse
import numpy as np
import os
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from libs.features import BoVW, extract_full_hog_features, extract_hog_matrix, extract_lbp_feature
"""
  ####### Preamble
"""

parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--whatrun', action='store', required=True )
parser.add_argument( '--dataloc', action='store', required=True )
parser.add_argument( '--testperc', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', nargs='+', type=int, default=[8,8] )
parser.add_argument( '--cpb', nargs='+', type=int, default=[1,1] )
parser.add_argument( '--numclusters', action='store', type=int, default=8 )
parser.add_argument( '--hidenparams', nargs='+', type=int, default=[256])
parser.add_argument( '--epochs', action='store', type=int, default=10 )
parser.add_argument( '--radius', action='store', type=int, default=1 )
parser.add_argument( '--npoints', action='store', type=int, default=8 )
parser.add_argument( '--nbins', action='store', type=int, default=128 )
parser.add_argument( '--range_bins', nargs='+', type=int, default=[0,256] )
parser.add_argument( '--verbosity', action='store', default=True )
flags = parser.parse_args()

# convert the verbosity flag to boolean
flags.verbosity = True if flags.verbosity == 'True' else False

ex01 = False
ex02 = False
ex03 = False

if flags.whatrun == 'ex01':
  ex01 = True
if flags.whatrun == 'ex02':
  ex02 = True
if flags.whatrun == 'ex03':
  ex03 = True

# Load your data here as you will be using the same data for each exercise.
# Use last week as an example, it's exactly the same here.
root = flags.dataloc
Xt, Xe = {}, {}
eval_perc = flags.testperc
# iterate over the classes (textures)
for t in sorted( os.listdir( root ) ):
  imgpaths = []
  # iterate over the files
  for f in sorted( os.listdir( os.path.join( root, t ) ) ):
    # append the file name to a list
    imgpaths.append( os.path.join( root, t, f ) ) # storing the images locations rather than the images themselves
  # split the lists into train and test sets
  Xt[t], Xe[t] = train_test_split( imgpaths, test_size=eval_perc )
# plot to ensure the sizes of each
for k in Xt.keys():
  print( k, len( Xt[k] ), len( Xe[k] ) )

"""
  ####### 1. HOG-BoVW MLP
"""
if ex01:
  # Extract the features for the HOG-BoVW trainer using the normal HOG parameters
  orient = flags.orient
  ppc = flags.ppc
  cpb = flags.cpb
  classvec_train = extract_full_hog_features( Xt, orient, ppc, cpb )
  # train the bag of visual words with 64 clusters
  num_clusters = flags.numclusters
  bovw = BoVW( num_clusters )
  bovw.fit( classvec_train )
  # Create the training feature matrix based on the HOG-BoVW features.
  firstfile = True
  train_labels = []
  for i, (k, v) in enumerate( Xt.items() ):
    for f in v:
      train_labels.append( i )
      feat = extract_hog_matrix( f, orient, ppc, cpb )
      feat = bovw.predict( feat )
      feat = feat.reshape( (1,-1) ) # ensure it is a horizontal matrix
      if firstfile:
        Xtrain = feat
        firstfile = False
      else:
        Xtrain = np.vstack( (Xtrain, feat) )
  # Normalise the data
  mu = Xtrain.mean( axis=0 )
  st = Xtrain.std( axis=0 )
  Xnorm = (Xtrain-mu)/st
  # Extract the evaluation dataset in the same way as the train set.
  # Previously we extracted them individualy but this time we will have an evaluation matrix.
  # Don't forget the eval labels and to normalise the feature vector.
  firstfile = True
  eval_labels = []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      eval_labels.append( i )
      feat = extract_hog_matrix( f, orient, ppc, cpb )
      feat = bovw.predict( feat )
      feat = feat.reshape( (1,-1) ) # ensure it is a horizontal matrix
      feat = (feat-mu)/st
      if firstfile:
        Xeval = feat
        firstfile = False
      else:
        Xeval = np.vstack( (Xeval, feat) )
  # Now let's train the MLP.
  # You will need to import MLPClassifier from sklearn.
  # We will update the MLP classifier over a number of epochs so we need to use it
  # in a similar way to the example given in the pdf. My guess is that we should  use
  # 256 hidden layers (it's a number higher than our clusters or feature length). You
  # can experiment with different values but for now 256 should suffice.
  num_classes = len( Xt.keys() )
  hidden_layers = flags.hidenparams + [num_classes]
  clf = MLPClassifier( hidden_layer_sizes=hidden_layers, # 32 hidden 6 classes
                        activation='relu', # default activation function (non linear)
                        solver='adam', # default solver
                    random_state=1, max_iter=1, warm_start=True)
  # Now we are going to iteratively update the model using epochs.
  # We will use 500 but again you can experiment with this value.
  # For each epoch we will fit the normalised training data with the training labels.
  # We will also calculate the accuracy on the training dataset of our model using
  # predict and accuracy_score. We will only do this accuracy of the training set if
  # verbosity is set to True.
  for i in range( flags.epochs ):
    clf.fit( Xnorm, train_labels )
    if flags.verbosity:
      pred = clf.predict( Xnorm )
      acc = accuracy_score( train_labels, pred )
      print( 'Training accuracy of the MLP at epoch {} is: {:0.04f}'.format( i, acc ) )
  # Once we have finished our training proceedure we will use predict on the evaluation set
  # and produce the accuracy score and a confusion matrix.
  pred = clf.predict( Xeval )
  acc = accuracy_score( eval_labels, pred )
  print( 'Evaluation accuracy of the MLP using HOG-BoVW is: {:0.04f}'.format( acc ) )
  print( confusion_matrix( eval_labels, pred ) )
  # What do you notice about the training versus the evaluation accuracy scores?
  # Can you be guaranteed dat we are getting the best model for our evaluation data?


"""
  ####### 2. LBP-SVM
"""
if ex02:
  # Extract and normalise the training and evaluation data (similar to above) but for
  # local binary patterns. i.e. no need to train a BoVW classifier. The evaluation set
  # should be a matrix not just a single feature vector.
  train_labels = []
  firstfile = True
  for i, (k, v) in enumerate( Xt.items() ):
    for f in v:
      train_labels.append( i )
      feat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins  )
      feat = feat.reshape( (1,-1) )
      if firstfile:
        X = feat
        firstfile = False
      else:
        X = np.vstack( (X, feat ) )
  # print( X.shape )
  # Now let's normalise these values.
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # eval data
  firstfile = True
  eval_labels = []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      eval_labels.append( i )
      feat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins  )
      feat = feat.reshape( (1, -1) )
      feat = (feat-mu)/st
      if firstfile:
        Xeval = feat
        firstfile = False
      else:
        Xeval = np.vstack( (Xeval, feat) )

  # Train an MLP in exactly the same manner as the previous exercise.
  num_classes = len( Xt.keys() )
  hidden_layers = flags.hidenparams + [num_classes]
  clf = MLPClassifier( hidden_layer_sizes=hidden_layers, # 32 hidden 6 classes
                        activation='relu', # default activation function (non linear)
                        solver='adam', # default solver
                    random_state=1, max_iter=1, warm_start=True)

  for i in range( flags.epochs ):
    clf.fit( Xnorm, train_labels )
    if flags.verbosity:
      pred = clf.predict( Xnorm )
      acc = accuracy_score( train_labels, pred )
      print( 'Training accuracy of the MLP at epoch {} is: {:0.04f}'.format( i, acc ) )
  pred = clf.predict( Xeval )
  acc = accuracy_score( eval_labels, pred )
  print( 'Evaluation accuracy of the MLP using LBP is: {:0.04f}'.format( acc ) )
  print( confusion_matrix( eval_labels, pred ) )


"""
  ####### 3. LBP+HOG-BOvW-SVM
"""
if ex03:
  # Now we will combine both the lbp and the hog-bovw into a single feature vector for training
  # and evaluating. Follow the previous practical but once again we will have a single
  # evaluation set matrix, like the past two exercises.
  orient = flags.orient
  ppc = flags.ppc
  cpb = flags.cpb
  classvec_train = extract_full_hog_features( Xt, orient, ppc, cpb )
  # train the bag of visual words and fit it. My experiments show that 64 is best for this.
  num_clusters = flags.numclusters
  bovw = BoVW( num_clusters )
  bovw.fit( classvec_train )
  # Create the training feature vector.
  firstfile = True
  train_labels = []
  for i, (k, v) in enumerate( Xt.items() ):
    for f in v:
      train_labels.append( i )
      hogfeat = extract_hog_matrix( f, orient, ppc, cpb )
      hogfeat = bovw.predict( hogfeat )
      hogfeat = hogfeat.reshape( (1,-1) )
      lbpfeat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins  )
      lbpfeat = lbpfeat.reshape( (1,-1) )
      # print( hogfeat.shape, lbpfeat.shape )
      feat = np.hstack( (hogfeat, lbpfeat) )
      if firstfile:
        X = feat
        firstfile = False
      else:
        X = np.vstack( (X, feat) )
  # Normalise the data
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # Get the evaluation data
  eval_labels = []
  firstfile = True
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      eval_labels.append( i )
      # extract the feature
      hogfeat = extract_hog_matrix( f, orient, ppc, cpb )
      hogfeat = bovw.predict( hogfeat )
      hogfeat = hogfeat.reshape( (1,-1) )
      lbpfeat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins  )
      lbpfeat = lbpfeat.reshape( (1,-1) )
      feat = np.hstack( (hogfeat, lbpfeat) )
      feat = (feat-mu)/st
      if firstfile:
        Xeval = feat
        firstfile = False
      else:
        Xeval = np.vstack( (Xeval, feat) )
  # Now let's train the MLP same as the last two exercises.
  num_classes = len( Xt.keys() )
  hidden_layers = flags.hidenparams + [num_classes]
  clf = MLPClassifier( hidden_layer_sizes=hidden_layers, # 32 hidden 6 classes
                        activation='relu', # default activation function (non linear)
                        solver='adam', # default solver
                    random_state=1, max_iter=1, warm_start=True)

  for i in range( flags.epochs ):
    clf.fit( Xnorm, train_labels )
    if flags.verbosity:
      pred = clf.predict( Xnorm )
      acc = accuracy_score( train_labels, pred )
      print( 'Training accuracy of the MLP at epoch {} is: {:0.04f}'.format( i, acc ) )
  pred = clf.predict( Xeval )
  acc = accuracy_score( eval_labels, pred )
  print( 'Evaluation accuracy of the MLP using HOG-BoVW + LBP is: {:0.04f}'.format( acc ) )
  print( confusion_matrix( eval_labels, pred ) )
