"""
assignment
"""
import skimage.color
from skimage.io import imsave, imread, imshow, show, imshow_collection
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve as prc


pca = True
ex02 = True
ex03 = True
ex04 = True

#data for number two:
print("BG")

with open("data/PAML_data/Q2_BG_dict.pkl", "rb") as sf:
    picture_BG = pickle.load(sf)

    print(picture_BG)


print("SP")

with open("data/PAML_data/Q2_SP_dict.pkl", "rb") as sf:
    picture_SP = pickle.load(sf)

# look at the data of number two and visualise it
    print(picture_SP.keys())

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

if pca:
  # The final exercise is based on priciple component analysis (PCA) which you will
  # try in your own time. I will release the soluton but now that we have used
  # a number of new pythonic tools you should be able to complete the exercise.
  # If you have questions, as always, I am available via email or during my consultation time.
  # In the lecture you were shown a number of uses for PCA, but in this practical you
  # will use it for dimension reduction. If we consider the housing data from a previous
  # practical we saw that there was a lot of independent variables that could be used to
  # predict the dependent variables. Sometimes these independent variables can be in the
  # 100's or even 1000's, which are usually untenable. PCA is a way of reducing these
  # dimensions down to something we can use for machine learning applications. Let's copy the
  # extract data function from week 04 to the top of this file. Then load the week04_housing.csv
  # file.
  picture_SP
  data, numrows = extract_data( 'picture_BG' )

  # Create a varaible with the dependent variable
  Y = data['picture_BG']
  # Now we need to create a matrix of our independent variables. Let's select:
  # longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population,
  # and median_house_value and put them in an (N*7) matrix.
  kys = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'median_income']
  X = []
  for k, v in data.items():
    if k in kys:
      # print( k, np.isnan( np.sum( v ) ) ) # Uncomment this line to see where the nan values are
      X.append( v )
  X = np.array( X ).T # This needs to be transposed back to (N*7)
  print( X.shape )
  # But we have a problem! There are nan values in this matrix. PCA can't handle nan values
  # so we will remove them. We need two functions here, np.isnan and np.where. isnan locates
  # the nan values in a vector or matrix and where returns the exact location in matrix co-ordinates.
  # let's print np.where( np.isnan( X ) ) can you work out how to remove values for X based on these
  # co-ordinates? We really only need the rows. (You could also just remove the offending column but,
  # that kind of defeats the purpose here). You can use np.delete to delete the rows:
  # https://numpy.org/doc/stable/reference/generated/numpy.delete.html
  wnan = np.where( np.isnan( X ) )
  print( 'before delete', wnan )
  X = np.delete( X, wnan[0], 0 ) # don't forget to tell it the axis it is deleting.
  print( 'after delete', np.where( np.isnan( X ) ) )
  # Now we have our data we can play with PCA. As always we need to import something.
  # "from sklearn.decomposition import PCA" So we are now using sklearn!
  # Now we need to create the PCA object (PCA is a class), let's select PCA components of 2,
  # keeping in mind that the componenets need to be <= 7, but 7 doesn't really make
  # much sense here, we are trying to reduce the dimension. 2 Is just a random number, we'll
  # play with different values later.
  # obj = PCA( n_components=N ) # N = 2 for now
  N = 2
  obj = PCA( n_components=N )
  # Now we need to fit the object to the X data we created.
  obj.fit( X )
  # now let's print the variance
  print( obj.explained_variance_ratio_ )
  # So the first dimension contains 95% of the variance! That's a significant amount.
  # It means that first dimension contains the majority of the information, so this
  # COULD be a good representation. Go back and change N to see what happens.

  ####### Spectral Clustering with sklearn

  if ex04:
    # This exercise is just a simple homework exercise to get you familiar with reading the
    # sklearn documentation. We are after Spectral clustering (urls are in the pdf). This
    # is just another method of clustering and actually can use KMeans. We will use this method
    # of clustering with the "discretize" variable. You'll find that in the documentation.
    # You'll also need to import the appropriate library, I'll let you look it up.
    # Create the object
    spcl = sklearnsc( n_clusters=3, assign_labels="discretize", )
    # Fit and predict the data (D0 or D1 whichever)
    Y = spcl.fit_predict( D1 )
    # Plot the result
    u = np.unique( Y )
    plt.figure()
    for i in u:
      plt.scatter( D1[Y==i,0], D1[Y==i,1], label='{}'.format( i ) )
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Calculate the metric.
    acc = skcs( L1.reshape( (-1,) ), Y )
    print( acc )

  #alex:use clustering and bag of visual words (lecture 10) or neural networks then
  #alex:or template matching with histograms, but no good classifier
  #alex: ass. 11: Hog instead of lbp. Together with bag of visual words:
  # alex: get more samples from image usaully with hog. Lbp gives me histogram,
  # alex: hog feature vector with orientations per location; lbp single value per location!


  ################
  for evaluation data set: precision recall curve f1
  # 2. precision_recall_curve - recommended for your assignment
  # This is a much simpler version and the one I would recommed you use from now on.
  # first import precision_recall_curve from sklearn.metrics, i have imported it as prc.
  # this looks like precision, recall, thresholds = precision_recall_curve( labels, scores )
  p, r, t = prc(labels, sc)
  p, r, t = prc(labels,dist(col, 0)
  # Now you need to calculate the f1-score in the same way as above.
  f1 = 2 * p * r / (p + r + 0.0000001)
  # plot the precision recall and the point where F1-score is at it's maximum.
  am = np.argmax(f1)
  plt.figure()
  plt.plot()
  plt.plot(r, p)
  plt.plot(r[am], p[am], 'r*')
  plt.title('Precision recall curve - Precision Recall: F1-score of {}'.format(f1[am]))
  plt.show()

