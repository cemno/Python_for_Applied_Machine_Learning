"""
  Today we will cover
  1. HOG based BoVW classification using SVMs
  2. LBP based classification using SVMs
  3. Combination of HOG-BoVW and LBP for classification using SVMs (homework)
  4. Classifying an image from a loaded model (homework)
"""

"""
  ####### Import area
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve as prc

from libs.features_solution import BoVW, extract_full_hog_features, extract_hog_matrix, extract_lbp_feature
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
parser.add_argument( '--C', action='store', type=float, default=1.0 )
parser.add_argument( '--radius', action='store', type=int, default=1 )
parser.add_argument( '--npoints', action='store', type=int, default=8 )
parser.add_argument( '--nbins', action='store', type=int, default=128 )
parser.add_argument( '--range_bins', nargs='+', type=int, default=[0,256] )
parser.add_argument( '--image', action='store', required=False )
flags = parser.parse_args()

ex01 = False
ex02 = False
ex03 = False
ex04 = False
ex05 = False

if flags.whatrun == 'ex01':
  ex01 = True
if flags.whatrun == 'ex02':
  ex02 = True
if flags.whatrun == 'ex03':
  ex03 = True
if flags.whatrun == 'ex04':
  ex04 = True
if flags.whatrun == 'ex05':
  ex05 = True


# Load your data here as you will be using the same data for each exercise.
# Like last week I will create a dictionary (with classes as keys) of lists (where
# the elements in the list are full file locations).
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
  ####### HOG-BoVW-SVM
"""
if ex01:
  # Use last week as a guide and train a BoVW object based on the training information.
  # We will use the same orientation and other hog parameters too.
  orient = flags.orient
  ppc = flags.ppc
  cpb = flags.cpb
  classvec_train = extract_full_hog_features( Xt, orient, ppc, cpb )
  # train the bag of visual words and fit it. Let's start easy with 5 clusters again.
  num_clusters = flags.numclusters
  bovw = BoVW( num_clusters )
  bovw.fit( classvec_train )
  # Next we need to create a feature vector of the training images using the bovw object.
  # Each image will have their own histogram based entry.
  firstfile = True
  train_labels = []
  for i, (k, v) in enumerate( Xt.items() ):
    for f in v:
      train_labels.append( i )
      feat = extract_hog_matrix( f, orient, ppc, cpb )
      feat = bovw.predict( feat )
      feat = feat.reshape( (1,-1) ) # ensure it is a horizontal matrix
      if firstfile:
        X = feat
        firstfile = False
      else:
        X = np.vstack( (X, feat) )
  # Now we will train the SVMs - there is information in the pdf. You will need to import
  # SVC from sklearn.svm
  # first let's train a classifier with the linear kernel using the default values
  clf_linear = SVC( kernel='linear', C=flags.C ) # default C=1.0
  # fit the linear classifier
  clf_linear.fit( X, train_labels )
  # Next let's train a classifier with the rbf kernel
  clf_rbf = SVC( kernel='rbf', C=flags.C, gamma='scale' ) # default C=1.0 and gamma='scale'
  # fit the rbf kernel model
  clf_rbf.fit( X, train_labels )
  # now we will evaluate both classifiers at once.
  # for each image you will compute the bovw output and classify using the two svms.
  # Based on this output  you will store a prediction, one list for linear and one for rbf.
  # You will also need a label list
  pred_lin, pred_rbf, eval_labels = [], [], []
  # Now let's iterate through the evaluation set, assign the label, produce the feature vector,
  # classify the feature vector and store the score.
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      # assign the label
      eval_labels.append( i )
      # extract the feature vector
      feat = extract_hog_matrix( f, orient, ppc, cpb )
      feat = bovw.predict( feat )
      feat = feat.reshape( (1,-1) ) # ensure it is a horizontal matrix
      # classify the feature vector and store the output
      p = clf_linear.predict( feat )
      pred_lin.append( p )
      p = clf_rbf.predict( feat )
      pred_rbf.append( p )
  # Now let's calculate the accuracy and the confusion matrix fore each.
  acc_lin = accuracy_score( eval_labels, pred_lin )
  print( 'Accuracy of the linear SVM based BoVW is: {:0.04f}'.format( acc_lin ) )
  print( confusion_matrix( eval_labels, pred_lin ) )

  acc_rbf = accuracy_score( eval_labels, pred_rbf )
  print( 'Accuracy of the rbf SVM based BoVW is: {:0.04f}'.format( acc_rbf ) )
  print( confusion_matrix( eval_labels, pred_rbf ) )

  # Now one of the problems with machine learning in general is the data itself.
  # It can be significantly varied and cause problems when we use the raw data.
  # We covered this earlier in the semester but a trick to fix this is data normalisation.
  # In the case of this dataset we won't see great changes and the fact that it's a small
  # set can actually make it perform worse. Let's use the mean standard deviation from
  # earlier. You will do this on your training vector after bovw and for each of the samples
  # that you evaluate. You will then retrain your svms and evaluate on the normalised data.
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # Train the svms
  clf_linear = SVC( kernel='linear', C=1.0 ) # default C=1.0
  # fit the linear classifier
  clf_linear.fit( Xnorm, train_labels )
  # Next let's train a classifier with the rbf kernel
  clf_rbf = SVC( kernel='rbf', C=1.0, gamma='scale' ) # default C=1.0 and gamma='scale'
  # fit the rbf kernel model
  clf_rbf.fit( Xnorm, train_labels )
  # Now evaluate. Don't forget to reinitialise your lists.
  pred_lin, pred_rbf, eval_labels = [], [], []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      # assign the label
      eval_labels.append( i )
      # extract the feature vector
      feat = extract_hog_matrix( f, orient, ppc, cpb )
      feat = bovw.predict( feat )
      feat = feat.reshape( (1,-1) ) # ensure it is a horizontal matrix
      feat = (feat-mu)/st # just a single line of difference here.
      # classify the feature vector and store the output
      p = clf_linear.predict( feat )
      pred_lin.append( p )
      p = clf_rbf.predict( feat )
      pred_rbf.append( p )
  # Now let's calculate the accuracy and the confusion matrix fore each.
  # What type of evaluation should we use here? Can we use the f1-score?
  acc_lin = accuracy_score( eval_labels, pred_lin )
  print( 'Accuracy of the normalised linear SVM based BoVW is: {:0.04f}'.format( acc_lin ) )
  print( confusion_matrix( eval_labels, pred_lin ) )

  acc_rbf = accuracy_score( eval_labels, pred_rbf )
  print( 'Accuracy of the normalised rbf SVM based BoVW is: {:0.04f}'.format( acc_rbf ) )
  print( confusion_matrix( eval_labels, pred_rbf ) )


"""
  ####### LBP-SVM
"""
if ex02:
  # We don't have to train a bovw classifier here, we will just use our LBP extractor from
  # a previous practical (features.py)
  # First create the training set by extracting lbp features per image and concatenating them
  # into a feature matrix. (I'll just use the standard input values).
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
  print( X.shape )
  # Now let's normalise these values.
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # Train the linear and rbf svms (similar to ex01)
  clf_linear = SVC( kernel='linear', C=flags.C ) # default C=1.0
  clf_linear.fit( Xnorm, train_labels )
  clf_rbf = SVC( kernel='rbf', C=flags.C, gamma='scale' ) # default C=1.0 and gamma='scale'
  clf_rbf.fit( Xnorm, train_labels )
  # Now evaluate the performance of these lbp based svms
  eval_labels, pred_lin, pred_rbf = [], [], []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      eval_labels.append( i )
      feat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins )
      feat = feat.reshape( (1, -1) )
      feat = (feat-mu)/st

      p = clf_linear.predict( feat )
      pred_lin.append( p )

      p = clf_rbf.predict( feat )
      pred_rbf.append( p )

  # calculate the two accuracy scores. and confusion matrices
  acc_lin = accuracy_score( eval_labels, pred_lin )
  print( 'Accuracy of the linear SVM based BoVW is: {:0.04f}'.format( acc_lin ) )
  print( confusion_matrix( eval_labels, pred_lin ) )

  acc_rbf = accuracy_score( eval_labels, pred_rbf )
  print( 'Accuracy of the rbf SVM based BoVW is: {:0.04f}'.format( acc_rbf ) )
  print( confusion_matrix( eval_labels, pred_rbf ) )

"""
  ####### LBP+HOG-BOvW-SVM
"""
if ex03:
  # Now we will combine both the lbp and the hog-bovw into a single feature vector for training
  # and evaluating. Follow the two practicals above to complete this in your own time.
  # Train the bovw model.
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
  print( X.shape )
  # Normalise the data
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # Now let's train the different svms
  clf_linear = SVC( kernel='linear', C=flags.C) # default C=1.0
  clf_linear.fit( Xnorm, train_labels )
  clf_rbf = SVC( kernel='rbf', C=flags.C, gamma='scale' ) # default C=1.0 and gamma='scale'
  clf_rbf.fit( Xnorm, train_labels )
  # now let's predict both of these models
  label_eval, pred_lin, pred_rbf = [], [], []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      label_eval.append( i )
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

      p = clf_linear.predict( feat )
      pred_lin.append( p )

      p = clf_rbf.predict( feat )
      pred_rbf.append( p )

  # calculate the two accuracy scores. and confusion matrices
  acc_lin = accuracy_score( label_eval, pred_lin )
  print( 'Accuracy of the linear SVM based LBP+BoVW is: {:0.04f}'.format( acc_lin ) )
  print( confusion_matrix( label_eval, pred_lin ) )

  acc_rbf = accuracy_score( label_eval, pred_rbf )
  print( 'Accuracy of the rbf SVM based LBP+BoVW is: {:0.04f}'.format( acc_rbf ) )
  print( confusion_matrix( label_eval, pred_rbf ) )

  # Finally let's say we want to save this information so we can use it later.
  # Let's create a dictionary and store all the relevant information in it.
  outdict = {}
  # let's store the information we need again:
  # first what are the classification keys that we will need?
  outdict['labels'] = list( Xt.keys() )
  # hog stuffclf_rbf
  outdict['orient'] = orient
  outdict['ppc'] = ppc
  outdict['cpb'] = cpb
  # the bovw model
  outdict['bovw'] = bovw
  # lbp stuff
  outdict['radius'] = flags.radius
  outdict['npoints'] = flags.npoints
  outdict['nbins'] = flags.nbins
  outdict['range_bins'] = flags.range_bins
  # feature vector normalisation
  outdict['mu'] = mu
  outdict['std'] = st
  # now let's just store the rbf svm
  outdict['svm'] = clf_rbf
  # now we have all the information we need. Let's create a pickle
  with open( 'bovwhog+lbp+svmrbf.pkl', 'wb' ) as fid:
    pickle.dump( outdict, fid )

"""
  ####### LBP+HOG-BOvW-SVM from loaded model
"""
if ex04:
  # Another homework exercise!
  # In this example we will load an example image and classify it based on what we have
  # previously trained. THIS COULD BE VERY HELPFUL FOR YOUR ASSIGNMENT!
  # NOTE: You need to complete exercise 3 before this one!
  # Let's load the pickle
  with open( 'bovwhog+lbp+svmrbf.pkl', 'rb' ) as fid:
    info = pickle.load( fid )
  # Now let's load the image, you'll need to insert this on the command line. Something
  # like parser.add_argument( '--image', action='store', required=False )
  img = flags.image
  # Extract the feature of the image
  # extract the feature
  hogfeat = extract_hog_matrix( img, info['orient'],
                              info['ppc'], info['cpb'] )
  hogfeat = info['bovw'].predict( hogfeat )
  hogfeat = hogfeat.reshape( (1,-1) )
  lbpfeat, _ = extract_lbp_feature( img,
                                  radius=info['radius'], # the radius about which to look
                                  npoints=info['npoints'],  # the number of points around the radius.
                                  nbins=info['nbins'], # for plotting the histogram
                                  range_bins=info['range_bins']  )
  lbpfeat = lbpfeat.reshape( (1,-1) )
  feat = np.hstack( (hogfeat, lbpfeat) )
  feat = (feat-info['mu'])/info['std']
  # classify the image
  p = info['svm'].predict( feat )
  # output the result, this can be anything, a segmented image but in this case I'll
  # just print texturally what I think the class of texture is. Keep in mind that the output
  # the svm is a list.
  print( 'This texture image {} has been classified as a {}'.format( img, info['labels'][p[0]] ) )


"""
  Two class SVM for precision-recall curve
"""
if ex05:
  # This is just a little extra function for those that actually look through the solution.
  # In this example I will show you how to use the SVM class to output a value that can
  # be used as a score to a precision recall curve.
  # First because pr curves need only two classes we will extract the plain and spot class
  # only.
  l = ['plain', 'spots'] # this should really be in the argparse section...
  Tt, Te = {}, {}
  for k, v in Xt.items():
    if k in l:
      Tt[k] = v
      Te[k] = Xe[k]
  Xt = Tt
  Xe = Te
  # We will then extract lbp features like above.
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
  print( X.shape )
  # Now let's normalise these values.
  mu = X.mean( axis=0 )
  st = X.std( axis=0 )
  Xnorm = (X-mu)/st
  # Train the linear and rbf svms (similar to ex01)
  clf_linear = SVC( kernel='linear', C=flags.C ) # default C=1.0
  clf_linear.fit( Xnorm, train_labels )
  clf_rbf = SVC( kernel='rbf', C=flags.C, gamma='scale' ) # default C=1.0 and gamma='scale'
  clf_rbf.fit( Xnorm, train_labels )
  # Now evaluate the performance of these lbp based svms, we also need some score lists for each
  # of the svms.
  eval_labels, pred_lin, pred_rbf = [], [], []
  scr_lin, scr_rbf = [], []
  for i, (k, v) in enumerate( Xe.items() ):
    for f in v:
      eval_labels.append( i )
      feat, _ = extract_lbp_feature( f,
                                      radius=flags.radius, # the radius about which to look
                                      npoints=flags.npoints,  # the number of points around the radius.
                                      nbins=flags.nbins, # for plotting the histogram
                                      range_bins=flags.range_bins )
      feat = feat.reshape( (1, -1) )
      feat = (feat-mu)/st

      p = clf_linear.predict( feat )
      # usually we just use predict, but we wnat a linear value from the decision boundary
      # so we will emply decision_function.
      s = clf_linear.decision_function( feat )
      pred_lin.append( p )
      scr_lin.append( s )

      p = clf_rbf.predict( feat )
      s = clf_rbf.decision_function( feat )
      pred_rbf.append( p )
      scr_rbf.append( s )

  # calculate the two accuracy scores. and confusion matrices
  acc_lin = accuracy_score( eval_labels, pred_lin )
  print( 'Accuracy of the linear SVM based BoVW is: {:0.04f}'.format( acc_lin ) )
  print( confusion_matrix( eval_labels, pred_lin ) )

  acc_rbf = accuracy_score( eval_labels, pred_rbf )
  print( 'Accuracy of the rbf SVM based BoVW is: {:0.04f}'.format( acc_rbf ) )
  print( confusion_matrix( eval_labels, pred_rbf ) )

  # now the f1score stuff.
  p, r, t = prc( eval_labels, scr_lin )
  # print( 't', len( t ) )
  f1 = 2*p*r/(p+r+0.0000001)
  am = np.argmax( f1 )
  plt.figure()
  plt.plot()
  plt.plot( r, p )
  plt.plot( r[am], p[am], 'r*' )
  plt.title( 'Linear Precision Recall: F1-score of {}'.format( f1[am] ) )
  plt.show()

  p, r, t = prc( eval_labels, scr_rbf )
  # print( 't', len( t ) )
  f1 = 2*p*r/(p+r+0.0000001)
  am = np.argmax( f1 )
  plt.figure()
  plt.plot()
  plt.plot( r, p )
  plt.plot( r[am], p[am], 'r*' )
  plt.title( 'RBF Precision Recall: F1-score of {}'.format( f1[am] ) )
  plt.show()
