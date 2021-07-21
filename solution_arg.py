"""
  Today we will cover
  1. Texture based BoVW classification using KL Divergence
"""

"""
  ####### Import area
"""
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve as prc

from libs.features_solution import extract_hog_matrix, extract_full_hog_features, extract_class_hog_features, BoVW
from libs.classifiers_solution import templatematch
# The following is a guide only. Read through what I have written in the pdf and see if you can
# come up with your own solution. Mine isn't overly efficient, it's just one solution.

# first let's parse the arguments based on the pdf information.
parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--dataloc', action='store', required=True )
parser.add_argument( '--textures', nargs="+", default=[] )
parser.add_argument( '--testperc', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', action='store', type=int, default=8 )
parser.add_argument( '--cpb', action='store', type=int, default=1 )
parser.add_argument( '--numclusters', action='store', type=int, default=16 )
flags = parser.parse_args()

#########################################################
## Below is my solution to this problem                ##
#########################################################
# The first step is to load the images names into a list per class (so a dictionary of lists).
# You will need to split these lists into two dictionaries, one for training and one for evaluation.
root = flags.dataloc
testperc = flags.testperc
ftrain, feval = {}, {} # dictionaries for
for t in sorted( os.listdir( root ) ):
  templist = []
  for f in sorted( os.listdir( os.path.join( root, t ) ) ):
    templist.append( os.path.join( root, t, f ) )
  # split into train and test
  ftrain[t], feval[t] = train_test_split( templist, test_size=testperc )
# Just a little print to look at the sizes
for k in ftrain.keys():
  print( k, len( ftrain[k] ), len( feval[k] ) )
# next let's extract the keys we want.
if len( flags.textures ) > 1:
  ttemp, etemp = {}, {}
  for k, v in ftrain.items():
    if k in flags.textures:
      ttemp[k] = ftrain[k]
      etemp[k] = feval[k]
  ftrain = ttemp
  feval = etemp
print( ftrain.keys(), feval.keys() )


# Now we need to train the bovw classifier based on the kmeans algorithm.
# This is an unsupervised clustering technique so we don't need lables for the training set.
# what we do need is a single feature matrix for all the classes.
# First let's specify some of the parameters for the HOG descriptor:
# orientation:8, pixels per cell:8, cells per block:1, visualize:False, feature vector: False
orient = flags.orient
ppc = (flags.ppc, flags.ppc)
cpb = (flags.cpb, flags.cpb)
# I am going to hardcode the visualize and feature vector parameters but you could pass them
# if you wanted. Let's go the feature.py function (you'll need to add the information in "add_to_features.py")
# to this file. First we will extract the hog information individually from a file name then
# we will extract it into a single feature vector (for training only)
# Once you have done that come back and extract the training feature vector for kmeans
kmeanstrain = extract_full_hog_features( ftrain, orient, ppc, cpb )
print( kmeanstrain.shape ) # what is our feature size?
# Next we are going to create our bovw classifier. Finish this in the features.py file.
# Now we can train our BoVW classsifer.
num_clusters = flags.numclusters
bovw = BoVW( num_clusters )
bovw.fit( kmeanstrain )
# Next we need to create the template matcher, which will be based on the bovw histogram output.
# The key here is that we have a template for EACH CLASS. So you need to create training data
# per class... Maybe you can edit the full feature extraction function (I actually recommend that,
# for efficiency) to output both the full feature and a dictionary per class?
# I'll create a new function for this in features.py extract_class_hog_features
histtrain = extract_class_hog_features( ftrain, orient, ppc, cpb )
# Now we need to code up the kl divergence class. This is in the classifier.py
tmatch = templatematch( bovw )
tmatch.fit( histtrain )
# Now we want to predict per image in the evaluation set. Keep in mind that we will need
# a label here too!
label, pred = [], []
start = True
for i, (k, v) in enumerate( feval.items() ):
  for f in v:
    label.append( i )
    feat = extract_hog_matrix( f, orient, ppc, cpb )
    p, s = tmatch.predict( feat )
    pred.append( p )
    if start:
      scores = s
      start = False
    else:
      scores = np.vstack( (scores, s) )
# accuracy
acc = accuracy_score( label, pred )
print( 'Accuracy of the KL divergence based BoVW is:', acc )
# So we get about 3 times guess, not a great classifier but okay. What happens when you play
# with the HOG and number of cluster parameters? Can you get it better?

# if we are only using two textures we can do f1-score! This is important for your assignment.
# You can really only do p-r curves for 2 classes. If you do more than that you need to
# consider other metrics.
if len( feval.keys() ) == 2:
  p, r, t = prc( np.array( label ), scores[:,0] )
  # Now you need to calculate the f1-score in the same way as above.
  f1 = 2*p*r/(p+r+0.0000001)
  # plot the precision recall and the point where F1-score is at it's maximum.
  am = np.argmax( f1 )
  plt.figure()
  plt.plot()
  plt.plot( r, p )
  plt.plot( r[am], p[am], 'r*' )
  plt.title( 'Precision recall curve - Precision Recall: F1-score of {}'.format( f1[am] ) )
  plt.show()
# We are getting weirdd results here, why?
