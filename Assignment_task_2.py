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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve as prc
from libs.features_assigment import extract_lbp_feature, extract_full_hog_features, extract_hog_matrix, BoVW


"""
Classes
"""
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern as lbp



parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--whatrun', action='store', required=True )
parser.add_argument( '--dataloc', action='store', required=False)
parser.add_argument( '--testperc', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', nargs='+', type=int, default=[8,8] )
parser.add_argument( '--cpb', nargs='+', type=int, default=[1,1] )
parser.add_argument( '--numclusters', action='store', type=int, default=8 )
parser.add_argument( '--C', action='store', type=float, default=1.0 )
parser.add_argument( '--hidenparams', nargs='+', type=int, default=[256])
parser.add_argument( '--epochs', action='store', type=int, default=10 )
parser.add_argument( '--radius', action='store', type=int, default=1 )
parser.add_argument( '--npoints', action='store', type=int, default=8 )
parser.add_argument( '--nbins', action='store', type=int, default=128 )
parser.add_argument( '--range_bins', nargs='+', type=int, default=[0,256] )
parser.add_argument( '--verbosity', action='store', default=True )
flags = parser.parse_args()

svm = False
mlp = False
pca = False

if flags.whatrun == "svm":
    svm = True
if flags.whatrun == "mlp":
    mlp = True
if flags.whatrun == "pca":
    pca = True

#data for number two:
with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)
print("BG imported")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)
print("SP imported")

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

# Create data:
Xt = {"BG": picture_BG["train"], "SP": picture_SP["train"]}
Xe = {"BG": picture_BG["validation"], "SP": picture_SP["validation"]}


if svm:
    # Use last week as a guide and train a BoVW object based on the training information.
    # We will use the same orientation and other hog parameters too.
    orient = flags.orient
    ppc = flags.ppc
    cpb = flags.cpb
    classvec_train = extract_full_hog_features(Xt, orient, ppc, cpb)
    # train the bag of visual words and fit it. Let's start easy with 5 clusters again.
    num_clusters = flags.numclusters
    bovw = BoVW(num_clusters)
    bovw.fit(classvec_train)
    # Next we need to create a feature vector of the training images using the bovw object.
    # Each image will have their own histogram based entry.
    firstfile = True
    train_labels = []
    for i, (k, v) in enumerate(Xt.items()):
        for f in v:
            train_labels.append(i)
            feat = extract_hog_matrix(f, orient, ppc, cpb)
            feat = bovw.predict(feat)
            feat = feat.reshape((1, -1))  # ensure it is a horizontal matrix
            if firstfile:
                X = feat
                firstfile = False
            else:
                X = np.vstack((X, feat))
    print(X)
    # Now we will train the SVMs - there is information in the pdf. You will need to import
    # SVC from sklearn.svm
    # first let's train a classifier with the linear kernel using the default values
    clf_linear = SVC(kernel='linear', C=flags.C)  # default C=1.0
    # fit the linear classifier
    clf_linear.fit(X, train_labels)
    # Next let's train a classifier with the rbf kernel
    clf_rbf = SVC(kernel='rbf', C=flags.C, gamma='scale')  # default C=1.0 and gamma='scale'
    # fit the rbf kernel model
    clf_rbf.fit(X, train_labels)
    # now we will evaluate both classifiers at once.
    # for each image you will compute the bovw output and classify using the two svms.
    # Based on this output  you will store a prediction, one list for linear and one for rbf.
    # You will also need a label list
    pred_lin, pred_rbf, eval_labels = [], [], []
    # Now let's iterate through the evaluation set, assign the label, produce the feature vector,
    # classify the feature vector and store the score.
    for i, (k, v) in enumerate(Xe.items()):
        for f in v:
            # assign the label
            eval_labels.append(i)
            # extract the feature vector
            feat = extract_hog_matrix(f, orient, ppc, cpb)
            feat = bovw.predict(feat)
            feat = feat.reshape((1, -1))  # ensure it is a horizontal matrix
            # classify the feature vector and store the output
            p = clf_linear.predict(feat)
            pred_lin.append(p)
            p = clf_rbf.predict(feat)
            pred_rbf.append(p)
    # Now let's calculate the accuracy and the confusion matrix fore each.
    acc_lin = accuracy_score(eval_labels, pred_lin)
    print('Accuracy of the linear SVM based BoVW is: {:0.04f}'.format(acc_lin))
    print(confusion_matrix(eval_labels, pred_lin))

    acc_rbf = accuracy_score(eval_labels, pred_rbf)
    print('Accuracy of the rbf SVM based BoVW is: {:0.04f}'.format(acc_rbf))
    print(confusion_matrix(eval_labels, pred_rbf))

    # Now one of the problems with machine learning in general is the data itself.
    # It can be significantly varied and cause problems when we use the raw data.
    # We covered this earlier in the semester but a trick to fix this is data normalisation.
    # In the case of this dataset we won't see great changes and the fact that it's a small
    # set can actually make it perform worse. Let's use the mean standard deviation from
    # earlier. You will do this on your training vector after bovw and for each of the samples
    # that you evaluate. You will then retrain your svms and evaluate on the normalised data.
    mu = X.mean(axis=0)
    st = X.std(axis=0)
    Xnorm = (X - mu) / st
    # Train the svms
    clf_linear = SVC(kernel='linear', C=1.0)  # default C=1.0
    # fit the linear classifier
    clf_linear.fit(Xnorm, train_labels)
    # Next let's train a classifier with the rbf kernel
    clf_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')  # default C=1.0 and gamma='scale'
    # fit the rbf kernel model
    clf_rbf.fit(Xnorm, train_labels)
    # Now evaluate. Don't forget to reinitialise your lists.
    pred_lin, pred_rbf, eval_labels = [], [], []
    for i, (k, v) in enumerate(Xe.items()):
        for f in v:
            # assign the label
            eval_labels.append(i)
            # extract the feature vector
            feat = extract_hog_matrix(f, orient, ppc, cpb)
            feat = bovw.predict(feat)
            feat = feat.reshape((1, -1))  # ensure it is a horizontal matrix
            feat = (feat - mu) / st  # just a single line of difference here.
            # classify the feature vector and store the output
            p = clf_linear.predict(feat)
            pred_lin.append(p)
            p = clf_rbf.predict(feat)
            pred_rbf.append(p)
    # Now let's calculate the accuracy and the confusion matrix fore each.
    # What type of evaluation should we use here? Can we use the f1-score?
    acc_lin = accuracy_score(eval_labels, pred_lin)
    print('Accuracy of the normalised linear SVM based BoVW is: {:0.04f}'.format(acc_lin))
    print(confusion_matrix(eval_labels, pred_lin))

    acc_rbf = accuracy_score(eval_labels, pred_rbf)
    print('Accuracy of the normalised rbf SVM based BoVW is: {:0.04f}'.format(acc_rbf))
    print(confusion_matrix(eval_labels, pred_rbf))

"""
  ####### 2. LBP-MLP
"""
if mlp:
  # Extract and normalise the training and evaluation data (similar to above) but for
  # local binary patterns. i.e. no need to train a BoVW classifier. The evaluation set
  # should be a matrix not just a single feature vector.
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
  # now the f1score stuff
  p, r, t = prc(eval_labels, pred)
  # print( 't', len( t ) )
  f1 = 2 * p * r / (p + r + 0.0000001)
  am = np.argmax(f1)
  plt.figure()
  plt.plot()
  plt.plot(r, p)
  plt.plot(r[am], p[am], 'r*')
  plt.title('Background and red data Precision Recall: F1-score of {}'.format(f1[am]))
  plt.show()

