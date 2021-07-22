"""
assignment
"""
import libs as libs
import skimage.color
from skimage.io import imsave, imread, imshow, show, imshow_collection
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve as prc

# KMeans
solution_1 = True
pie_chart = False
plot_cluster = False
# Multivariate Gaussian
solution_2 = False
ex03 = True
ex04 = True

# first the data will be loaded.
#data for number one:
print("background")
with open("data/PAML_data/Q1_BG_dict.pkl", "rb") as sf:
    picture_background = pickle.load(sf)
    #print (picture_background)


with open("data/PAML_data/Q1_Red_dict.pkl", "rb") as sf:
    picture_red = pickle.load(sf)
    #print (picture_red)

print("yellow")

with open("data/PAML_data/Q1_Yellow_dict.pkl", "rb") as sf:
    picture_yellow = pickle.load(sf)
    #print(picture_yellow)



#data for number two:
print("BG")

with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)
    #print(picture_BG)


print("SP")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)
    #print(picture_SP.keys())

#dict_keys(['train', 'validation', 'evaluation'])
#The sets in the dictionary are: dict_keys(['train', 'validation', 'evaluation'])
#The size of the data matrix X for each set is: (10000, 3) (5000, 3) (5000, 3): Many pictures especially in
#training data set, but also each 5000 in the validation and evaluation data sets
#he size of each entry is: (64, 64, 3) (64, 64, 3) (64, 64, 3): These are very big pictures, as visible

#imshow(picture_SP["train"] [1])
#show()
#imshow_collection(picture_SP["train"])
#show()
#imshow_collection(picture_SP["validation"])
#show()
#imshow(picture_SP["validation"] [1])
#show()
#imshow_collection(picture_SP["evaluation"])
#show()
for fname in ['data/PAML_data/Q1_BG_dict.pkl',  'data/PAML_data/Q1_Red_dict.pkl',  'data/PAML_data/Q1_Yellow_dict.pkl']:
	print("PAML_data/Q1_BG_dict.pkl'", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The size of the data matrix X for each set is:", data['train'].shape, data['validation'].shape, data['evaluation'].shape)


for fname in ['data/PAML_data/Q2_BG_dict.pkl',  'data/PAML_data/Q2_SP_dict.pkl']:
	print("data/PAML_data/Q2_BG_dict.pkl", fname)
	with open(fname,'rb') as fp:
		data = pickle.load(fp)
		print("The sets in the dictionary are:", data.keys())
		print("The number of entries for each set is:", len(data['train']), len(data['validation']), len(data['evaluation']))
		print("The size of each entry is:", data['train'][0].shape, data['validation'][0].shape, data['evaluation'][0].shape)


# Arranging data # Addition: no need to stack, doing a kmeans with one cluster is good enough, sorting data to the correct labels later on
#train_data = np.vstack([picture_background["train"], picture_red["train"]])
#print(train_data.shape)
# Creating k_means classes
if solution_1:
    train_data_combined = np.vstack([picture_background["train"], picture_red["train"], picture_yellow["train"]])

    kmeans_all = KMeans(500) # Whats the optimal cluster number?! 6 Clusters first time yellow/orange gets separated from red

    ## Changing colour space
    colourspace = "xyz"
    train_data_combined = skimage.color.convert_colorspace(train_data_combined, fromspace= "rgb", tospace=colourspace)


    ## Fitting data
    # kmeans_background.fit(picture_background["train"])
    # print(kmeans_background.cluster_centers_)
    # kmeans_red.fit(picture_red["train"])
    # print(kmeans_red.cluster_centers_)
    #kmeans_yellow.fit(picture_yellow["train"])
    #print(kmeans_yellow.cluster_centers_)

    # Fit data to satisfy _n_threads in kmeans object, which handles errors if no fitting was done
    kmeans_all.fit(train_data_combined)
    #print(kmeans_all.cluster_centers_)
    # Change cluster_centroids to the ones calculated by each KMeans object for each Set
    #kmeans_all.cluster_centers_ = np.vstack([kmeans_background.cluster_centers_, kmeans_red.cluster_centers_])


    ## Classifiy data based on its distance to the clusters using predict
    #Y_background = kmeans_background.predict(picture_background["validation"])
    #Y_red = kmeans_red.predict(picture_background["validation"])

    #Y_all = kmeans_all.predict(validation_data_combined)
    validation_data_red = skimage.color.convert_colorspace(picture_red["validation"], fromspace= "rgb", tospace=colourspace)
    red_validation = kmeans_all.predict(validation_data_red)
    validation_data_yellow = skimage.color.convert_colorspace(picture_yellow["validation"], fromspace= "rgb", tospace=colourspace)
    yellow_validation = kmeans_all.predict(validation_data_yellow)

    def cluster_occurrences(classification_array):
      """ Returns value for cluster with maximum red occurrences and a list all clusters where red was matched at least ones"""
      occurrences = {}
      for num in np.unique(classification_array):
        # Fill dictionary with occurrences per class
        occ = classification_array.tolist().count(num)
        occurrences.update({num: occ})
      print(occurrences)
      # Class with most occurrences - This is debatable but probably best for low cluster numbers.
      max_occ =  max(zip(occurrences.values(), occurrences.keys()))[1]
      occurrences = np.asarray(list(occurrences.keys()), dtype = int)
      return max_occ.astype(int), occurrences

    # Choosing cluster with the most red occurrences
    red_cluster, all_red_clusters =  cluster_occurrences(red_validation)
    yellow_cluster, all_yellow_clusters = cluster_occurrences(yellow_validation)
    print("Red cluster: {}\nYellow cluster: {}".format(red_cluster, yellow_cluster))
    #print(Y_all)

    #print(np.where(Y_all == 0)[0].size, np.where(Y_all == 1)[0].size)

    if pie_chart:
      labels_all = kmeans_all.labels_
      print(labels_all)
      labels=list(labels_all)
      centroid = kmeans_all.cluster_centers_
      print("centroids")
      print(centroid)
      centroid = skimage.color.convert_colorspace(centroid, fromspace=colourspace, tospace="rgb")
      centroid = centroid * 255
      print(centroid)

      percent=[]
      for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)
      print(percent)

      plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
      plt.show()

    # now use our data on this
    # Let's use this class to cluster some data (D0 from exercise 1) with 4 clusters to start with
    # and an imax of zero (we only randomly assign centers).
    # Create the object
    #kmeans = KMeans(2, imax=100)
    #X = picture_background  # I did this so I could easily change later.
    #kmeans.fit(X)
    #Y = kmeans.predict(X)
    # let's plot what this looks like
    # but first we want to know the  unique values in D0a so we aren't constantly having
    # to change the label values, you'll need np.unique
    if plot_cluster:
      Yu = np.unique(Y_all)
      print(Yu.shape)
      # now scatter plt based on the predictions
      plt.figure()
      for i in Yu:
        plt.scatter(validation_data_combined[Y_all == i, 0], validation_data_combined[Y_all == i, 1], label='{}'.format(i))
      plt.legend()
      plt.show()
      plt.close()

    #1280,720,3 -> 921600,3
    # Read in image and transfrom to mask
    #img_input = imread("data/week04/week03_rgb.png")

    for img_input in picture_SP["train"]:
       img_transformed = img_input.reshape(-1, img_input[0, 0, :].size)
       img_transformed = skimage.color.convert_colorspace(img_transformed, fromspace="rgb", tospace=colourspace)
       print(img_transformed.shape)
       index_classes = kmeans_all.predict(img_transformed)
       msk_zero = np.zeros(img_transformed.shape, dtype=int)
       yellow_pixel, red_pixel = 0, 0
       for i in range(msk_zero.shape[0]):
         if index_classes[i] in all_yellow_clusters: # instead of "in all_..." write "== yellow_cluster"
           msk_zero[i] = np.array([255, 255, 0], dtype=int)
           yellow_pixel += 1
         elif index_classes[i] in all_red_clusters: # instead of "in all_..." write "== red_cluster"
           msk_zero[i] = np.array([255, 0, 0], dtype=int)
           red_pixel += 1
         else:
           msk_zero[i] = np.array([0, 255, 0], dtype=int)
       msk = msk_zero.reshape(img_input.shape)
       print("Red pixel: {:0.01f}%, Yellow pixel: {:0.01f}%, Background pixel: {:0.01f}%".format(red_pixel * 100 / msk_zero.shape[0], yellow_pixel * 100 / msk_zero.shape[0], (msk_zero.shape[0] - yellow_pixel - red_pixel) * 100 / msk_zero.shape[0]))
       imshow(msk, vmin=0, vmax=255)
       plt.figure()
       plt.imshow(msk, vmin=0, vmax=255)
       plt.imshow(img_input)
       plt.show()

    # Create evaluation labels for background and red data
    evaluation_data_bg_red = np.vstack([picture_background["evaluation"], picture_red["evaluation"]])
    evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace = "rgb", tospace = colourspace)
    len_bg = len(picture_background["evaluation"])
    len_red = len(picture_red["evaluation"])
    eval_labels_bg_red = np.concatenate([np.zeros(len_bg, dtype = int), np.ones(len_red, dtype = int)])

    # Extend evaluation labels with labels for yellow dataset
    evaluation_data_all = np.vstack([picture_background["evaluation"], picture_red["evaluation"], picture_yellow["evaluation"]])
    evaluation_data_all = skimage.color.convert_colorspace(evaluation_data_all, fromspace = "rgb", tospace = colourspace)
    len_yellow = len(picture_yellow["evaluation"])
    eval_labels_all = np.concatenate([eval_labels_bg_red, np.full(len_yellow, 2)])

    # Create labels based on prediction on the evaluation dataset for background and red
    pred_eval_bg_red = kmeans_all.predict(evaluation_data_bg_red)
    tmp = np.zeros(pred_eval_bg_red.shape, dtype=int)
    for i, cluster in enumerate(pred_eval_bg_red):
      if cluster in all_red_clusters: # instead of "in all_..." write "== red_cluster"
        tmp[i] = 1
    pred_eval_bg_red = tmp

    pred_eval_all = kmeans_all.predict(evaluation_data_all)
    tmp = np.zeros(pred_eval_all.shape, dtype=int)
    for i, cluster in enumerate(pred_eval_all):
      if cluster in all_red_clusters: # instead of "in all_..." write "== red_cluster"
        tmp[i] = 1
      elif cluster in all_yellow_clusters: # instead of "in all_..." write "== yellow_cluster"
        tmp[i] = 2
    pred_eval_all = tmp

    # now the f1score stuff.
    p, r, t = prc(eval_labels_bg_red, pred_eval_bg_red)
    # print( 't', len( t ) )
    f1 = 2*p*r/(p+r+0.0000001)
    am = np.argmax( f1 )
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( 'Background and red data Precision Recall: F1-score of {:0.04f}'.format( f1[am] ) )
    plt.show()

    # calculate the two accuracy scores. and confusion matrices
    acc_lin = accuracy_score( eval_labels_bg_red, pred_eval_bg_red )
    print( 'Accuracy of the bg and red data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( eval_labels_bg_red, pred_eval_bg_red ) )

    # calculate the two accuracy scores. and confusion matrices
    acc_lin = accuracy_score( eval_labels_all, pred_eval_all )
    print( 'Accuracy of the bg, red and yellow data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( eval_labels_all, pred_eval_all ) )

if solution_2:
    class MultivariateGaussian:
        # Create the __init__ function, you will also need to initialise the base class: super().__init__()
        # This class will also take two inputs: mu and sigma which default to an empty list each.
        # If both of these members are not empty you should run the _precalculations method
        # which we will code up next.
        def __init__(self, mu=[], sigma=[]):
            super().__init__()
            self.mu = mu
            self.sigma = sigma
            if (not (self.sigma == []) and (not (self.mu == []))):
                self._precalculations()

        # When we perform the log likelihood calculation we need to calculate some values
        # including the Sigma^-1 and |Sigma| as you can see in the pdf. Along with these
        # values we will also precompute the constant values from the pdf.
        # Create a method called _precalculations with no inputs.
        def _precalculations(self):
            # How many dimensions do we have?
            n = self.mu.shape[1]

            # Calculate the inverse matrix using np.linalg.inv and store as a member
            self.inv_sigma = np.linalg.inv(self.sigma)

            # calculate the two constant values from the pdf.
            # the log determinant can be calculated by np.linalg.slogdeg()
            log_two_pi = -n / 2. * np.log(2 * np.pi)
            log_det = -0.5 * np.linalg.slogdet(self.sigma)[1]

            # now sum these two constants together and store them as a member.
            self.constant = log_two_pi + log_det

        # Next we will overwrite the log_likelihood method from the base class.
        def log_likelihood(self, X):
            # get the shape of the data (m samples, n dimensions)
            m, n = X.shape

            # create an empty log likelihood output to the shape of m
            llike = np.zeros((m,))

            # calculate the residuals X - mu
            resids = X - self.mu

            # iterate over the number of data points (m) in residuals and calculate the log likelihood for each.
            # equation in the pdf, using the members created in _precalculations.
            # Hopefully, you see the benefit of precalculating the constants and inverse.
            for i in range(m):
                llike[i] = self.constant - resids[i, :] @ self.inv_sigma @ resids[i, :].T

            # return the log likelihood values
            return llike

        # Now we will overwrite the train function.
        def train(self, X):
            # get the shape of the data
            m, n = X.shape

            # step 1 estimate the mean values. X is of size (m,n) and take the sum over m samples.
            # then divide by the total number of samples.
            mu = np.sum(X, axis=0) / float(m)
            mu = np.reshape(mu, (1, n))

            # Step 2 calculate the covariance matrix
            # residuals
            norm_X = X - mu

            # covariance n,n = (n,m @ m,n) / float( m )
            sigma = (norm_X.T @ norm_X) / float(m)

            # Assign class values and compute internals
            self.mu = mu
            self.sigma = sigma

            # step 3 precalcuate the internals for log likelihood
            self._precalculations()


    class MultivariateGaussian:
        # Create the __init__ function, you will also need to initialise the base class: super().__init__()
        # This class will also take two inputs: mu and sigma which default to an empty list each.
        # If both of these members are not empty you should run the _precalculations method
        # which we will code up next.
        def __init__(self, mu=[], sigma=[]):
            super().__init__()
            self.mu = mu
            self.sigma = sigma
            if (not (self.sigma == []) and (not (self.mu == []))):
                self._precalculations()

        # When we perform the log likelihood calculation we need to calculate some values
        # including the Sigma^-1 and |Sigma| as you can see in the pdf. Along with these
        # values we will also precompute the constant values from the pdf.
        # Create a method called _precalculations with no inputs.
        def _precalculations(self):
            # How many dimensions do we have?
            n = self.mu.shape[1]

            # Calculate the inverse matrix using np.linalg.inv and store as a member
            self.inv_sigma = np.linalg.inv(self.sigma)

            # calculate the two constant values from the pdf.
            # the log determinant can be calculated by np.linalg.slogdeg()
            log_two_pi = -n / 2. * np.log(2 * np.pi)
            log_det = -0.5 * np.linalg.slogdet(self.sigma)[1]

            # now sum these two constants together and store them as a member.
            self.constant = log_two_pi + log_det

        # Next we will overwrite the log_likelihood method from the base class.
        def log_likelihood(self, X):
            # get the shape of the data (m samples, n dimensions)
            m, n = X.shape

            # create an empty log likelihood output to the shape of m
            llike = np.zeros((m,))

            # calculate the residuals X - mu
            resids = X - self.mu

            # iterate over the number of data points (m) in residuals and calculate the log likelihood for each.
            # equation in the pdf, using the members created in _precalculations.
            # Hopefully, you see the benefit of precalculating the constants and inverse.
            for i in range(m):
                llike[i] = self.constant - resids[i, :] @ self.inv_sigma @ resids[i, :].T

            # return the log likelihood values
            return llike

        # Now we will overwrite the train function.
        def train(self, X):
            # get the shape of the data
            m, n = X.shape

            # step 1 estimate the mean values. X is of size (m,n) and take the sum over m samples.
            # then divide by the total number of samples.
            mu = np.sum(X, axis=0) / float(m)
            mu = np.reshape(mu, (1, n))

            # Step 2 calculate the covariance matrix
            # residuals
            norm_X = X - mu

            # covariance n,n = (n,m @ m,n) / float( m )
            sigma = (norm_X.T @ norm_X) / float(m)

            # Assign class values and compute internals
            self.mu = mu
            self.sigma = sigma

            # step 3 precalcuate the internals for log likelihood
            self._precalculations()


    colourspace = "hsv"
    train_data_bg = skimage.color.convert_colorspace(picture_background["train"], fromspace="rgb", tospace=colourspace)
    train_data_red = skimage.color.convert_colorspace(picture_red["train"], fromspace="rgb", tospace=colourspace)
    validation_data_bg_red = skimage.color.convert_colorspace(np.vstack([picture_background["validation"], picture_red["validation"]]), fromspace="rgb", tospace=colourspace)
    # Make sure you have created the MVG class in gaussian.py and imported the library.
    # I imported my library as pamlGaus, you can call it something else if you'd like.
    # Once you have completed the MVG class:
    # Create a MVG object and train them individually on the X0, and X1 datasets
    mvg_bg = MultivariateGaussian()
    mvg_bg.train(train_data_bg)
    mvg_red = MultivariateGaussian()
    mvg_red.train(train_data_red)
    # Now we will do something we haven't done before, and because our classes are simplistic
    # we have this option. We will look at self.mu and self.sigma from each of the mvg objects and
    # visually compare them to the original datasets. We do this by object.mu and object.sigma
    # print('MVG')
    # print('Original X0 parameters\n', X0mu, '\n', X0cv)
    # print('Calculated mvg0 parameters\n', mvg0.mu, '\n', mvg0.sigma)
    # print('Original X1 parameters\n', X1mu, '\n', X1cv)
    # print('Calculated mvg1 parameters\n', mvg1.mu, '\n', mvg1.sigma)
    # Now we will use these two distributions to classify the full data. Obviously in practice
    # we would have a training and evaluation set but in this case we are simply going to
    # classify the training set based on the models.
    # To do this we need both X0 and X1 concatenated together if you haven't already done that.
    # We will then use the log_likelihood function from both the MVGs to classify the data
    # resulting in a matrix of [m,2] (2 as we have 2 models).
    loglike = np.zeros((validation_data_bg_red.shape[0], 2))
    loglike[:, 0] = mvg_bg.log_likelihood(validation_data_bg_red)
    loglike[:, 1] = mvg_red.log_likelihood(validation_data_bg_red)
    # Okay now we need to classify based on the maximum response in the matrix of log_likelihood.
    # In this case the closer a point is the the distribution the greater its log likelihood,
    # or the better it fits the distribution. Calculate that now
    classified = np.argmax(loglike, axis=1)
    # Now we need a way to decide how good this classification is. For this we will use sklearns.metrcs
    # accuracy_score function. It compares the labels with the predictions and can give either
    # a normalised (0-1) score or a count of correctly classified points. You will need
    # the labels and the classified vector.
    evaluation_data_bg_red = np.vstack([picture_background["validation"], picture_red["validation"]])
    evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace="rgb",
                                                              tospace=colourspace)
    len_bg = len(picture_background["validation"])
    len_red = len(picture_red["validation"])
    eval_labels_bg_red = np.concatenate([np.zeros(len_bg, dtype=int), np.ones(len_red, dtype=int)])

    acc = accuracy_score(eval_labels_bg_red, classified)
    print('Accuracy of the MVGs is:', acc)
    print( confusion_matrix( eval_labels_bg_red, classified ) )

    # now the f1score stuff.
    p, r, t = prc(eval_labels_bg_red, classified)
    # print( 't', len( t ) )
    f1 = 2*p*r/(p+r+0.0000001)
    am = np.argmax( f1 )
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( 'Background and red data Precision Recall: F1-score of {}'.format( f1[am] ) )
    plt.show()


