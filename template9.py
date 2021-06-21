"""
  Today we will cover
  1. Outlier detection and removal
  2. Precision, Recall, and F1 Score
  3. Precision, Recall, and F1 score for GMMs (homework)
"""

"""
  ####### Import area
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import libs.outlierdetection as od
import libs.metrics as met
"""
  ####### Preamble
"""
#parser = argparse.ArgumentParser(description= 'Parsing the command line', add_help=True)
#parser.add_argument('--whatrun', action = 'store', required=True)
#flags = parser.parse_args()

ex01 = False
ex02 = True
ex03 = False
"""
if flags.whatrun == 'ex01':
    ex01 = True
elif flags.whatrun == 'ex02':
    ex02 = True
elif flags.whatrun == 'ex03':
    ex03 = True
else:
    print("Not a variable that can be set.")
"""
"""
  ####### 1. Outliers
"""
if ex01:
  # First we will load the picke called week08_mvg.pkl
  # It contains a single variable in the load of size (n,2).
  fid = open('data/week09/week08_mvg.pkl', 'rb')
  X = pickle.load(fid)
  fid.close()

  # confirm the size
  print('original size', type(X), np.shape(X))

  # In the pdf I gave you the mean and covariance of what the data should be.
  # Let's calculate these two values to see what it actually is. In practice you won't
  # always know these values.
  print(X.mean(axis=0))
  print(np.cov(X.T))

  # Okay, so the first problem you should see is that there are nan values in there.
  # Let's remove them.
  nloc = np.where(np.isnan(X))
  X = np.delete(X, nloc[0], axis = 0)
  print('After nan', np.shape(X))


  # let's calculate the mean and covariance again.
  print(X.mean(axis = 0))
  print(np.cov(X.T))

  # So there is still an issue, let's delete the infinite values
  nloc = np.where(np.isinf(X))
  X = np.delete(X, nloc[0], axis=0)
  print('After inf', np.shape(X))

  # calculate the mean and covariance again.
  print(X.mean(axis = 0))
  print(np.cov(X.T))

  # so we have values close to the descrived mu and sigma but not exactly...
  # for 1000 points you would hope that we could get closer. The next step is to
  # plot this data. In your own time I recommend you also plot the data at each of
  # previous steps: Import, NaN removal, Inf removal, what do you notice?
  plt.figure()
  plt.scatter(X[:,0], X[:,1])
  plt.title('After nan and inf')
  # plt.show()
  plt.close()

  # What do you see here, how many outliers are there visually?
  # Let's now move to our outlierdetection.py file to create two different methods
  # for removing outliers. Print the mean and covariance for each one, does it improve?
  # If you set up the function as described in the pdf you will have to do this for
  # each dimension. Obviously this isn't optimal but it's really just to show you
  # how to use numpy.logical_and as we will use this later.
  bz0 = od.z_score(X[:, 0])
  bz1 = od.z_score(X[:, 1])
  Xz = X[np.logical_and(bz0, bz1)]
  print('Z-score')
  print(Xz.mean(axis = 0))
  print(np.cov(Xz.T))
  bzm0 = od.z_mod_score(X[:, 0])
  bzm1 = od.z_mod_score(X[:, 1])
  Xm = X[np.logical_and(bzm0, bzm1)]
  print('Mod-score')
  print(Xm.mean(axis=0))
  print(np.cov(Xm.T))

  # Now let's plot them all on one plot, use subplots and set sharex=True and sharey=True
  # Don't forget to put a title on each subplot...
  fig, ax = plt.subplots(1,3,sharex=True, sharey=True)
  ax[0].scatter(X[:,0], X[:,1])
  ax[0].set_title('Original')
  ax[1].scatter(Xz[:,0], Xz[:,1])
  ax[1].set_title('Z-sore')
  ax[2].scatter(Xm[:, 0], Xm[:, 1])
  ax[2].set_title('Mod-sore')
  plt.show()
  # Play around with the thresholds and see if you can improve this.
  # What do you notice about the two different techniques?

"""
  ####### 2. Precision, Recall, and F1 Score
"""
if ex02:
  # Make sure you read the pdf before here. Then you will need to code up the F1 score
  # class in the metrics.py file.
  # So to continue here you have to have created the first 3 steps in the metric.py file.
  # As a working example we are going to create univariate data to classify.
  # You will need np.random.normal( mean, std, size=(number of samples, 1) )
  # Let's create two univariate (mean, std) - X0~(0.7, 0.3) and X1~(0.3, 0.1) and start with 10
  # samples for each. You will need to create associated boolean labels where the first
  # distribution is True and the second is False. You will finally need to concatenate
  # the labels and data into single matrices.
  # Also just for simplicity let's clip both X0 and X1 to between 0 and 1 using np.clip()
  # https://numpy.org/doc/stable/reference/generated/numpy.clip.html
  samples = 1000
  X0 = np.random.normal( 0.7, 0.3, size = (samples, 1))
  X0 = np.clip(X0, 0., 1.)
  L0 = np.ones((samples, 1))
  X1 = np.random.normal(0.3, 0.1, size = (samples, 1))
  X1 = np.clip(X1, 0., 1.)
  L1 = np.zeros((samples, 1))
  X = np.vstack((X0, X1))
  L = np.vstack((L0,L1))

  # Now plot this as a histogram to see what it looks like.
  # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
  plt.figure()
  plt.hist(X0, bins = 50, range = (0., 1.))
  plt.hist(X1, bins = 50, range = (0., 1.))
  plt.show()
  plt.close()
  # Now to use our statistics function we need to create a predictions vector.
  # Let's assume a threshold of 0.4 (0.7 - 0.3) to start with.
  # Create the boolean predictions.
  f1s = met.f1score()
  p, r, f1 = f1s.calculate_statistics(L, X >= 0.4)
  print('p {:0.03f} r {:0.03f}, f1 {:0.03f}'.format(p, r, f1))

  # Now let's see how good our prediction is.
  # Create a f1score object and classify with your statistics method.


  # Play with that threshold a bit and see what you get.

  # Okay so playing with this threshold is somewhat guess work.
  # We usually need a way of automatically selecting the threshold, this is
  # where step 4 of the f1score comes into its own. Let's go back and complete that.

  # In my example I have used the __call__ function, but you could have just as easily
  # given it its own name. Let's use the same f1score object that we created previously
  # but use this threshold calculating method to see what we get.
  p, r, f1, th = f1s(L, X, 20, True)

  plt.figure()
  h, _, _ = plt.hist(X0, bins = 50, range = (0., 1.), color = 'blue')
  plt.hist(X1, bins = 50, range = (0., 1.), color = 'red')
  plt.plot([th, th], [0,h.max()], color = 'green')
  plt.show()
  # Now increase the number of samples and see what happens.

"""
  ####### 3. Precision, Recall, and F1 Score for GMMs
"""
# if ex03:
  # load in the colour snippets like we did in week 7 (you can copy and paste but wouldn't
  # it be convenient if we had put this in a library file).
  # But this week we are going to convert the images to the Lab space. Rember to import.
  # And as we are doing colour we are only going to use the chrominance channels (a and b)
  # So you will have each image, convert it to Lab, then index only the 1st and 2nd channels.
  # In your own time I recommend using all three channels and comparing them to RGB to see the results.


  # So now we have our colour data dictionaries but to start with we really only want
  # the black and white data. You can play with these two colours later, I recommend
  # having a look at blue and grey (they are closely related). For the two colours
  # split into two dictionaries, one for train and one for evaluation using train_test_split
  # don't forget to import.


  # Next we will train a GMM using our MultiGMM class. You'll need to import this.
  # The output of this gmm class for the prediction should also now include the scores...


  # From the evaluation dictionary create a single matrix of datapoints.
  # Don't forget the labels, but in this case make black True and white False


  # Now let's predict this vector and evaluate its accuracy using sklearn.metrics accuracy score
  # Keep in mind that we have inverted the labels (first = True, second = False) so we will
  # need to invert the classification (0->True, 1->False) np.logical_not  might come in handy.


  # In this case, like last week we were able to train two GMMs (or multiple GMMs).
  # But what if we only have 1 gmm, we can't compute the argmax and return a classification.
  # This is where the scores and the f1score is helpful. Let's assume we only have the black
  # training data (you could retrain the gmm but it's not needed) in this case that's the 0th
  # column in the scores output.
  # Let's get the data and reshape it to -1, 1


  # Now that we are familiar with how precision-recall curves work
  # we will switch over to inbuilt versions from sklearn.metrics. This is not due to errors in
  # the class above but due to the inbuilt efficiency of the functions (making them quicker).
  # For your assignment I recommend you use one of the following two versions.
  # 1. classification_report.
  # You will need import classification_report from sklearn.metrics.
  # This method is somewhat similar to the class, you will create thresholds using np.linspace
  # based on the minimum and maximum values of the scores. Then we will create empty arrays for
  # precision, recall, and f1-score. Once you have done that you can iterate over the
  # thresholds.
  # Inside the for loop you will use classification_report( labels, scores>=current threshold,
  # output_dict=True, zero_division=0 ).
  # Now this is where it gets a little tricky, the output will be a dictionary based on the labels.
  # if you have 1 and 0 the dictionary will have keys '1.0' and '0.0' for you to access the data,
  # in our case we have a boolean array so it will be "True" and "False". Now we only care
  # about the positive case so we would index into this dictionary in the following manner:
  # <output from classification_report>['True']['f1-score']
  # <output from classification_report>['True']['precision']
  # <output from classification_report>['True']['recall']
  # Each of the values will be stored in the arrays you created earlier.


  # Next we will plot the precision(y-axis) and recall (x-axis) based on your arrays.
  # And we will display the best F1-score (using np.argmax).


  # 2. precision_recall_curve - recommended for your assignment
  # This is a much simpler version and the one I would recommend you use from now on.
  # first import precision_recall_curve from sklearn.metrics, i have imported it as prc.
  # this looks like precision, recall, thresholds = precision_recall_curve( labels, scores )


  # Now you need to calculate the f1-score in the same way as above.


  # plot the precision recall and the point where F1-score is at it's maximum.
