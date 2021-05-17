"""
  Today we will cover
  1. creating some more functions.
  2. what to think about when selecting features
  3. linear regression using sklearn
"""

"""
  ####### Import area
"""
import pickle
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
random.seed(1212)
"""
  ####### Preamble
"""
runex1 = False
runex2 = False
runex3 = True


# Helper function for you to use!
# function for extracting information from the housing.csv file
# notice that I put this function at the top and not at the bottom.
def extract_data(location):
    data = pd.read_csv(location)
    print('the shape of the data is:', data.shape)
    str = ''
    out = {}
    for k, val in data.items():  # explain enumerate
        str += '|{} '.format(k)
        out[k] = np.array(val)
    print(str)
    return out, data.shape[0]


'''
    def mse(g, p):
        if len(g) == len(p):
            sum = float()
            for i in range(len(g) - 1):
                sum += (g[i] - p[i]) ** 2
            mse = sum / len(g)
            return mse
        else:
            raise Exception("Input for mse() has not the same length.")

        # Mean absolute error or L1-Distance (|x| indicates the absolute value of x)


    #       mae = \frac{1}{N}\sum_{i=0}^{N-1}|g - p|
    def mae(g, p):
        if len(g) == len(p):
            sum = float()
            for i in range(len(g) - 1):
                sum += np.abs(g[i] - p[i])
            mse = sum / len(g)
            return mse
        else:
            raise Exception("Input for mse() has not the same length.")


    # Root mean squared
    #       rms = \sqrt{\frac{1}{N}\sum_{i=0}^{N-1}(g - p)^2}
    def rms(g, p):
        if len(g) == len(p):
            sum = float()
            for i in range(len(g) - 1):
                sum += (g[i] - p[i]) ** 2
            rms = np.sqrt(sum / len(g))
            return rms
        else:
            raise Exception("Input for rms() has not the same length.")
    '''


def mse(g, p):
    diff = g - p
    sqrt = diff ** 2
    return np.mean(sqrt)


def mae(g, p):
    diff = g - p
    abs = np.abs(diff)
    return np.mean(abs)


def rms(g, p):
    return np.sqrt(mse(g, p))


"""
  ####### 1. Create metrics as functions where the inputs are 2D vectors.
"""
if runex1:
    # g in this case is the labels, or expected values.
    # p in this case is the predictions of our models.
    # both of these need to be exactly the same dimensions. Let's assume that they are
    # always of shape (1,N)
    # Create a function for the following three expressions, you can find the functions
    # in the PDF.
    # Mean squared error or L2-Distance
    #       mse = \frac{1}{N}\sum_{i=0}^{N-1}(g - p)^2
    g = [1, 2, 5, 4, 5]
    p = [5, 5, 3, 4, 5]

    # print("MSE: " + repr(mse(g, p)) + "\nMAE: " + repr(mae(g, p)) + "\nRMS: " + repr(round(rms(g, p), 2)))

    # now let's use them all.
    # first let's load the pickle week04_gp.pkl
    # This file is a dictionary with the prediction and ground truth in there.
    # Do you remember how to check what keys are in the pickle?
    # If we are going to use pickle what do we need to do in the import area?
    with open("data/week05/Practical5/week04_gp.pkl", 'rb') as file:
        dump = pickle.load(file)
        # print(dump.keys())
        gt = dump["gt"]
        # print(gt)
        gt = gt[0, :]
        pred = dump["pred"]
        pred = pred[0, :]

    # Now that we know the keys let's use the three functions we created...
    print("MSE: " + repr(round(mse(gt, pred), 2)) + "\nMAE: " + repr(round(mae(gt, pred), 2)) + "\nRMS: " + repr(
        round(rms(gt, pred), 2)))
"""
  ##### 2. Feature selection
"""
if runex2:
    # Please refer to the pdf for how to install pandas in our environment.
    # Once you have installed it make sure you add it to the import area "import pandas as pd".
    # For this exercise we will try and see if we can select a feature for training a linear regression model.
    # It is important to have your goal fixed in your mind, so in this case we want
    # our independent variables to have some form of linear relationship with our dependent
    # variable.
    # First let's use extract_data to extract the information from the csv file "week04_housing.csv"
    # and investigate the returning dictionary.
    housing = extract_data("data/week05/Practical5/week04_housing.csv")
    print(housing)
    print(housing[0]["latitude"])
    # What do you think the dependent value is here?
    # median_house_value
    # i would guess the median_house_value is dependent on the median_income of an individual.
    '''
    Y = housing[0]["median_house_value"]
    for k, x in housing[0].items():
        if k == 'median_house_value':
            continue
        #
        plt.figure()
        plt.scatter(x, Y, alpha=0.2)
        plt.xlabel(k)
        plt.ylabel( 'median_house_value')
        plt.tight_layout()
        plt.show()
    '''
    # now let's plot the dependent versus the others to see if we can spot a good
    plt.figure()
    plt.xlabel("Median house value")
    plt.ylabel("Median income")
    plt.title("Property dependent on income")
    plt.scatter(housing[0]["median_house_value"], housing[0]["median_income"], c="red", alpha=0.2)
    plt.savefig("data/week05/Practical5/income_house-value.png")
    # independent value HINT use the scatter plot function, don't forget to import.

    # so from that visual inspection, keeping in mind we want a linear relationship,
    # which would you select?
    # Run it again if you need to. Once you have selected one, comment that for loop out.
    # Now save this variable versus the dependent variable using savefig...

    # So that's how we might go about selecting variables using a visual inspection.
    # This isn't always possible especially when you consider quadratic relationships or even noisey data.
    # But it's ALWAYS a good place to start, having a look at your data. If you don't know
    # the properties of your data it's much more difficult to build a good model.
    pass
"""
  ###### 3. Linear regression
"""
if runex3:
    # So in the previous exercise we selected a dependent and independent variable.
    # Let's see if we can train a linear regression model that fits this information together.
    # And then finally we will use the metrics from exercise 1 to evaluate the performance
    # our our systems.
    # 1. Let's load the data again and get the median income as x and the median house value as y
    housing = extract_data("data/week05/Practical5/week04_housing.csv")
    # Now we need to split these into a training and testing set.
    # There are a number of ways to do this, I will step you through one variation, and
    # not a very efficient solution.
    # 1. import random in your import area.
    # 2. Decide on a training/testing split. Let's go with 50% for now (0.5) of the number of rows

    # 3. Create a "traininglist" of randomly assigned indexes that will select data from
    # both x and y. To do this we will use random.sample( A, B ), where A is the full list of
    # numbers we can select from: range(0, 20640) in this case. And B is the number of
    # values you are going to select out of A. Try it now
    data_length = housing[1]
    trainList = random.sample(range(0, data_length), int(data_length / 2))
    # print(len(trainList))

    # Now we need to create the evaluation list.
    # Basically you will just go through range( numrows ) and if an integer is not in
    # trainlist you will put it in evallist. Hopefully you see that this is not very
    # efficient..
    evalList = []
    for i in range(data_length):
        if i not in trainList:
            evalList.append(i)
    # alternative (more pythonic)
    evalList = [i for i in range(data_length) if i not in trainList]

    # print(len(evalList))

    # So now we have a list of index's that correspond to the training and evaluation
    # samples. Let's use these indexes on x and y to create the subsets.

    x_train = [housing[0]["median_income"][i] for i in trainList]
    # or (because its a vector) we can do: x_train = housing[0]["median_income"][trainList]
    x_eval = [housing[0]["median_income"][i] for i in evalList]

    y_train = [housing[0]["median_house_value"][i] for i in trainList]
    y_eval = [housing[0]["median_house_value"][i] for i in evalList]

    # Now we have the data to train a model and the data to evaluate how good our model is.
    # Let's plot these two sub sets individually.
    plt.figure()
    plt.subplot(211)
    plt.scatter(y_train, x_train, c="red", alpha=0.2)
    plt.subplot(212)
    plt.scatter(y_eval, x_eval, c="green", alpha=0.2)
    # plt.show()
    plt.savefig("data/week05/Practical5/train_vs_eval.png")

    # Okay now we'll do the linear regression using sklearn. First you need to import
    # the linear regression function... from sklearn.linear_model import LinearRegression
    # First instantiate the class (there are some hints in the pdf).
    linreg = LinearRegression()
    # Try and fit using your current data. What error comes up?
    #    linreg.fit(x_train, y_train)
    # So we have to fix the data so that it's the correct shape. The notification actually
    # tells you how to go about this. There are a number of ways to go about this, but try
    # the recommended version, however make the size (-1, 1) not (1, -1) like suggested.
    x_train = np.asarray(x_train)
    x_eval = np.asarray(x_eval)
    y_train = np.asarray(y_train)
    y_eval = np.asarray(y_eval)
    x_train = x_train.reshape(-1, 1)
    x_eval = x_eval.reshape(-1, 1)
    # you could have also done the following

    # don't forget to do it for the text set too.

    # now let's fit the model with the appropriately shaped data.
    linreg.fit(x_train, y_train)
    print("R²: " + repr(round(linreg.score(x_train, y_train), 2)))
    # So inside the class we now have a trained model. Let's predict the test set based on
    # this model!
    y_pred = linreg.predict(x_eval)
    # now let's plot ypred as a line and y_test as a scatter versus x_test
    plt.figure()
    plt.scatter(x_eval, y_eval, c="red", alpha=0.1)
    plt.plot(x_eval, y_pred, c="black")
    plt.show()
    # now let's use our metrics to see how accurately we were able to predict things?
    # Remember we had these set up as (1,N) matrix...
    print("mse {}".format(mse(y_train, y_pred)))
    print("mae {}".format(mae(y_train, y_pred)))
    print("rms {}".format(rms(y_train, y_pred)))
    pass
