"""
  Today we will cover
  1. creating some more functions.
  2. what to think about when selecting features
  3. linear regression using sklearn
"""

"""
  ####### Import area
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
"""
  ####### Preamble
"""
runex1 = False
runex2 = False
runex3 = False

# Helper function for you to use!
# function for extracting information from the housing.csv file
# notice that I put this function at the top and not at the bottom.
def extract_data( location ):
  data = pd.read_csv( location )
  print( 'the shape of the data is:', data.shape )
  str = ''
  out = {}
  for k, val in data.items(): # explain enumerate
    str += '|{} '.format( k )
    out[k] = np.array( val )
  print( str )
  return out, data.shape[0]

"""
  ####### 1. Create metrics as functions where the inputs are 2D vectors.
"""
if runex1:
  # g in this case is the labels, or expected values.
  # p in this case is the predictions of our models.
  # both of these need to be exactly the same dimensions. Let's assume that they are
  # always of shape (1,N)
  # Create a function for the following three expressisons, you can find the functions
  # in the PDF.
  # Mean squared error or L2-Distance
  #       mse = \frac{1}{N}\sum_{i=0}^{N-1}(g - p)^2
  def mse( g, p ):


  # Mean absolute error or L1-Distance (|x| indicates the absolute value of x)
  #       mae = \frac{1}{N}\sum_{i=0}^{N-1}|g - p|
  def mae( g, p ):


  # Root mean squared
  #       rms = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(g - p)^2}
  def rms( g, p ):


  # now let's use them all.
  # first let's load the pickle week04_gp.pkl
  # This file is a dictionary with the prediction and ground truth in there.
  # Do you remember how to check what keys are in the pickle?
  # If we are going to use pickle what do we need to do in the import area?


  # Now that we know the keys let's use the three functions we created...


"""
  ##### 2. Feature selection
"""
# if runex2:
  # Please refer to the pdf for how to install pandas in our environment.
  # Once you have installed it make sure you add it to the import area "import pandas as pd".
  # For this exercise we will try and see if we can select a feature for training a linear regression model.
  # It is important to have your goal fixed in your mind, so in this case we want
  # our independent variables to have some form of linear relationship with our dependent
  # variable.
  # First let's use extract_data to extract the information from the csv file "week04_housing.csv"
  # and investigate the returning dictionary.


  # What do you think the depenedent value is here?


  # now let's plot the depenedent versus the others to see if we can spot a good
  # independent value HINT use the scatter plot function, don't forget to import.


  # so from that visual inspection, keeping in mind we want a linear relationship,
  # which would you select?
  # Run it again if you need to. Once you have selected one, comment that for loop out.
  # Now save this variable versus the depenedent variable using savefig...


  # So that's how we might go about selecting variables using a visual inspection.
  # This isn't always possible especially when you consider quadratic relationships or even noisey data.
  # But it's ALWAYS a good place to start, having a look at your data. If you don't know
  # the properties of your data it's much more difficult to build a good model.

"""
  ###### 3. Linear regression
"""
# if runex3:
  # So in the previous exercise we selected a dependent and independent variable.
  # Let's see if we can train a linear regression model that fits this information together.
  # And then finally we will use the metrics from exercise 1 to evaluate the performance
  # our our systems.
  # 1. Let's load the data again and get the median income as x and the median house value as y


  # Now we need to split these into a training and testing set.
  # There are a number of ways to do this, I will step you through one variation, and
  # not a very efficient solution.
  # 1. import random in your import area.
  # 2. Decide on a training/testing split. Let's go with 50% for now (0.5) of the number of rows


  # 3. Create a "traininglist" of randomly assigned indexes that will select data from
  # both x and y. To do this we will use random.sample( A, B ), where A is the full list of
  # numbers we can select from: range(0, 20640) in this case. And B is the number of
  # values you are going to select out of A. Try it now


  # Now we need to create the evaluation list.
  # Basically you will just go through range( numrows ) and if an integer is not in
  # trainlist you will put it in evallist. Hopefully you see that this is not very
  # efficient..


  # So now we have a list of indexs that corrospond to the training and evaluation
  # samples. Let's use these indexes on x and y to create the subsets.


  # Now we have the data to train a model and the data to evaluate how good our model is.
  # Let's plot these two sub sets individually.


  # Okay now we'll do the linear regression using sklearn. First you need to import
  # the linear regression function... from sklearn.linear_model import LinearRegression
  # First instantiate the class (there are some hints in the pdf).


  # Try and fit using your current data. What error comes up?


  # So we have to fix the data so that it's the correct shape. The notification actually
  # tells you how to go about this. There are a number of ways to go about this, but try
  # the recommended version, however make the size (-1, 1) not (1, -1) like suggested.


  # you could have also done the following


  # don't forget to do it for the text set too.


  # now let's fit the model with the appropriately shapped data.


  # So inside the class we now have a trained model. Let's predict the test set based on
  # this model!


  # now let's plot ypred as a line and y_test as a scatter versus x_test


  # now let's use our metrics to see how accurately we were able to predict things?
  # Remember we had these set up as (1,N) matrix...
