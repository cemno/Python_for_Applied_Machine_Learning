"""
assignment
"""
import pickle
import argparse
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve as prc
from libs_assignment.features import extract_lbp_feature, extract_class_hog_features, extract_full_hog_features, extract_hog_matrix, BoVW
from libs_assignment.classifiers import templatematch

parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--whatrun', action='store', required=True )
parser.add_argument( '--dataloc', action='store', required=False)
parser.add_argument( '--testperc', action='store', type=float, default=0.3 )
parser.add_argument( '--orient', action='store', type=int, default=8 )
parser.add_argument( '--ppc', nargs='+', type=int, default=[8,8] )
parser.add_argument( '--cpb', nargs='+', type=int, default=[1,1] )
parser.add_argument( '--numclusters', action='store', type=int, default=8 )
parser.add_argument( '--gamma', action='store', type=float, default=1.0 )
parser.add_argument( '--C', action='store', type=float, default=1.0 )
parser.add_argument( '--hidenparams', nargs='+', type=int, default=[256])
parser.add_argument( '--epochs', action='store', type=int, default=200 )
parser.add_argument( '--radius', action='store', type=int, default=1 )
parser.add_argument( '--npoints', action='store', type=int, default=8 )
parser.add_argument( '--nbins', action='store', type=int, default=128 )
parser.add_argument( '--range_bins', nargs='+', type=int, default=[0,256] )
parser.add_argument( '--verbosity', action='store', default=True )
flags = parser.parse_args()

hog_bovw_mlp = False
svm = False
mlp = False
template = False


if flags.whatrun == "hog_bovw_mlp":
    hog_bovw_mlp = True
if flags.whatrun == "svm":
    svm = True
if flags.whatrun == "mlp":
    mlp = True
if flags.whatrun == "template":
    template = True
#data for number two:
with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)
print("BG imported")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)
print("SP imported")

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
    mu = classvec_train.mean(axis=0)
    st = classvec_train.std(axis=0)
    classvec_train = (classvec_train - mu) / st
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
    clf_linear = SVC(kernel='linear', C=flags.C)
    clf_linear.fit(X, train_labels)
    # classifier with the rbf kernel
    clf_rbf = SVC(kernel='rbf', C=flags.C, gamma = flags.gamma)
    # fit the rbf kernel model
    clf_rbf.fit(X, train_labels)

    pred_lin, pred_rbf, eval_labels = [], [], []
    for i, (k, v) in enumerate(Xe.items()):
        for f in v:
            # assign the label
            eval_labels.append(i)
            # extract the feature vector
            feat = extract_hog_matrix(f, orient, ppc, cpb)
            feat = bovw.predict(feat)
            feat = feat.reshape((1, -1))
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

    # now the f1score stuff
    p, r, t = prc(eval_labels, pred_rbf)
    # print( 't', len( t ) )
    f1 = 2 * p * r / (p + r + 0.0000001)
    am = np.argmax(f1)
    plt.figure()
    plt.plot()
    plt.plot(r, p)
    plt.plot(r[am], p[am], 'r*')
    plt.title('RBF Precision Recall: F1-score of {}'.format(f1[am]))
    plt.show()

"""
  ####### 2. LBP-MLP
"""
if mlp:
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
  plt.title('MLP Precision Recall: F1-score of {}'.format(f1[am]))
  plt.show()

if template:
    orient = flags.orient
    ppc = flags.ppc
    cpb = flags.cpb
    kmeanstrain = extract_full_hog_features(Xt, orient, ppc, cpb)
    num_clusters = flags.numclusters
    bovw = BoVW(num_clusters)
    bovw.fit(kmeanstrain)
    histtrain = extract_class_hog_features(Xt, orient, ppc, cpb)
    tmatch = templatematch(bovw)
    tmatch.fit(histtrain)
    label, pred = [], []
    start = True
    for i, (k, v) in enumerate(Xe.items()):
        for f in v:
            label.append(i)
            feat = extract_hog_matrix(f, orient, ppc, cpb)
            p, s = tmatch.predict(feat)
            pred.append(p)
            if start:
                scores = s
                start = False
            else:
                scores = np.vstack((scores, s))
    # accuracy
    acc = accuracy_score(label, pred)
    print('Accuracy of the KL divergence based BoVW is:', acc)
    if len(Xe.keys()) == 2:
        p, r, t = prc(np.array(label), scores[:, 0])
        f1 = 2 * p * r / (p + r + 0.0000001)
        am = np.argmax(f1)
        plt.figure()
        plt.plot()
        plt.plot(r, p)
        plt.plot(r[am], p[am], 'r*')
        plt.title('Precision recall curve - Precision Recall: F1-score of {}'.format(f1[am]))
        plt.show()