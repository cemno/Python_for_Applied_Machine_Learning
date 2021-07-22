"""
assignment
"""
import skimage.color
from skimage.io import imsave, imread, imshow, show, imshow_collection
import pickle
import argparse
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve as prc

"""
Classes
"""
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern as lbp
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



parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--whatrun', action='store', required=True )
parser.add_argument( '--dataloc', action='store', required=False)
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


mlp = False
pca = False

if flags.whatrun == "mlp":
    mlp = True
if flags.whatrun == "pca":
    pca = True

#data for number two:
print("BG")

with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)

    #print(picture_BG)


print("SP")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)

# look at the data of number two and visualise it
    #print(picture_SP.keys())

#dict_keys(['train', 'validation', 'evaluation'])
#The sets in the dictionary are: dict_keys(['train', 'validation', 'evaluation'])
#The size of the data matrix X for each set is: (10000, 3) (5000, 3) (5000, 3): Many pictures especially in
#training data set, but also each 5000 in the validation and evaluation data sets
#he size of each entry is: (64, 64, 3) (64, 64, 3) (64, 64, 3): These are very big pictures, as visible

#imshow(picture_SP["train"] [1])
#show()
for fname in ['data/PAML_data/Q2_BG_dict.pkl',  'data/PAML_data/Q2_SP_dict.pkl']:
	print("data/PAML_data/Q2_BG_dict.pkl", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The number of entries for each set is:", len(data['train']), len(data['validation']), len(data['evaluation']))
		print("The size of each entry is:", data['train'][0].shape, data['validation'][0].shape, data['evaluation'][0].shape)
"""
  ####### 2. LBP-MLP
"""
if mlp:
  # Extract and normalise the training and evaluation data (similar to above) but for
  # local binary patterns. i.e. no need to train a BoVW classifier. The evaluation set
  # should be a matrix not just a single feature vector.
  Xt = {"BG": picture_BG["train"], "SP": picture_SP["train"]}
  train_labels = []
  firstfile = True
  for i, (k,v) in enumerate( Xt.items() ):
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
  Xe = {"BG": picture_BG["validation"], "SP": picture_SP["validation"]}
  firstfile = True
  eval_labels = []
  for i, (k,v) in enumerate( Xe.items() ):
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
