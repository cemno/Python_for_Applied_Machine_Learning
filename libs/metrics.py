"""
  This file will contain any metrics we build.
"""
import matplotlib.pyplot as plt
import numpy as np

# The precision recall curve and the f1 metric are a widely used method of displaying
# how accurate a technique is at classification. The pdf and last weeks lecture should
# give you an overview of the maths behind it. Here we will code up a class for
# calculating these values, plus introduce a new inbuilt python class function.
# 1. create a class called f1score
# 2. create the init function that has epsilon as a default input with a value of 0.000001
#     this should be stored as a class member.
# 3. create a method that calculates the precison recall and f1score from two boolean inputs
#     labels and predictions.
#     You will most likely need np.logical_and and np.logical_not to complete this method.
#     You should also return the three key statistics
# !!!!!!!!!!! Once you have done these 3 steps go back to the template and start working there
#             until you get sent back here for step 4.
# 4. Create a function that dynamically calculates the thresholds for predicting scores
#     and selects the best precision and recall based on the highest f1score.
#     Hopefully from the exercise in template.py you see the importance of a good threshold.
#     You can use np.linspace to calculate a range of thresholds to iterate over.
#     The inputs to this should be: labels, scores (float), number of thresholds (int),
#     verbosity (boolean)
#     Where the verbosity flag let's you plot the output of precision recall and f1score
#     and print the best scores (including the threshold)
#     At the very least it should output the vectors of P, R, and F1.
import numpy as np


class f1score():
  # 2.
  def __init__(self, epsilon = 0.000001):
    self.epsilon = epsilon


  # 3. I'll call mine calculate_statistics
  def calculate_statistics(self, labels, predictions):
    TP = np.sum(np.logical_and(labels, predictions))
    FP = np.sum(np.logical_and(np.logical_not(labels), predictions))
    MD = np.sum(np.logical_and(labels, np.logical_not(predictions)))

    P = TP / (TP + FP + self.epsilon)
    R = TP / (TP + MD + self.epsilon)
    F1 = 2 * P * R / (P + R + self.epsilon)
    return P, R, F1

  # 4. Create a threshold f1score calculator I will use __call__()
  def __call__(self, labels, scores, nthresh = 20, verbosity = True):
    # create the thresholds (min and max of scores)
    thresholds = np.linspace(scores.min(), scores.max(), nthresh)

    # create the empty precision recall and f1score vectors
    P = np.zeros((nthresh,))
    R = np.zeros((nthresh,))
    F1 = np.zeros((nthresh,))
    # iterate around the thresholds
    for i, th in enumerate(thresholds):
      s = scores >= th
      # boolean of the scores
      # print( len( np.where( s )[0] ) )
      # calculate the statistics
      P[i], R[i], F1[i] = self.calculate_statistics(labels, s)

    # calculate the best based on the highest f1 score
    bestth = np.argmax(F1)
    plt.figure()
    plt.plot(R, P)
    plt.plot(R[bestth], P[bestth], 'r*')
    plt.xlabel( 'Recall' )
    plt.ylabel( 'Precision' )
    plt.title( 'Precision Recall Plot - F1 Score {:0.03f}'.format(F1[bestth]) )
    # let's plot these values if verbosity is true

    # return the best values.
    return P[bestth], R[bestth], F1[bestth], thresholds[bestth]