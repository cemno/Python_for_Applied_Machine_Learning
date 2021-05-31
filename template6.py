"""
  Today we will cover
  1. Skimage and colour spaces. Are there better features?
  2. Texture features - local binary patterns, histogram of oriented gradients
  3. Image normalisation
  4. Dimensionality reduction - principle component analysis
"""

"""
  ####### Import area
"""

import numpy as np

"""
  ####### Preamble
"""
ex01 = False
ex02 = False
ex03 = False
ex04 = True


"""
  ####### 1. Colour spaces
"""
if ex01:
  # Colour spaces are a very important part of computer vision and machine learning.
  # Generally when we look at images they are in the RGB (red, green, and blue).
  # Can anyone tell me what's wrong with the RGB colour space?
  # To start with let's load an image (we did this in week 03), what do you have to import
  # to load an image?
  # Once you have done this read in the week05_sp_00.png


  # Now we need the colour space conversions from skimage.
  # put "import skimage.color as skcol" in your import area. We did this in week 03.
  # Once you have done this let's create a gray scale version and a lab version and any
  # other versions you want to. https://scikit-image.org/docs/dev/api/skimage.color.html


  # Now like we have done previously, let's add 20 to the lab image and subtract 20 from
  # another (remember you only want the luminance channel). You will need to copy the lab
  # image (lab.copy()) into two new images. This is due to the mutable properties of numpy arrays...


  # Now let's convert the greyscale image and the two lab images to rgb and save them.
  # You'll need to add something to your import section if you didn't do it earlier.


  # Depending on how you do this you will get some warnings. You can usually ignore them.
  # Just check the images you save.
  # This was just a refresher on how to deal with images. In the next few exercises
  # we will use some of these principles.

"""
  ####### 2. Local binary patterns and histograms of oriented gradients.
"""
if ex02:
  # Another key component of machine learning is texture information.
  # These play a key role in deep learning where features are learned dynamically.
  # But in traditional machine learning we often had to create them ourselves.
  # There are a number of texture information providers, LBP, HOG, Gabor, SIFT, SURF,
  # and Radon are just a few.
  # In this exercise we will use two very well known descriptors and anayse the
  # resulting feature vectors and what they tell us.
  # Let's start by loading the hor_stripe.jpg


  # let's convert it to greyscale


  # Let's start with the histogram of oriented gradients. You will need to import
  # hog from the skimage.features library...
  # This is another tutorial: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
  # The hog feature splits the image into blocks and within those blocks calculates
  # the orientation of pixels (gradients) these values are then aggregated into a
  # histogram that shows the general orientation of that block.
  # Each block is the treated independently.
  # For this we need a number of hyperparameters: orientations, pixels_per_cell,
  # cells_per_block, and we will set visualize to True for this exercise.
  # Our first pass
  #       orientations = 4
  #       pixels_per_cell=(8,8)
  #       cells_per_block=(1,1)
  #       visualize = True
  # Create variables with these parameters


  # The total call would be:
  #   features, map = hog( gry, orientations=, pixels_per_cell=, cells_per_block=, visualize=True, feature_vector=False )
  # Now using the above rgb image, the variables you created, and the call I set create
  # a hog representation of the image.


  # Now we will visualise both the image and the hog map.
  # We will use the subplots function we used in the visualisation practical.
  # What do you need to import?
  # Let's create a subplot with (1,2) windows (rgb, map)


  # let's plot the image on ax[0] using imshow( rgb )


  # now let's plot the map (ax[1])


  # plt.show()
  # Go back and play with the parameters.
  # Read in some of the other texture snippets and see what it looks like.
  # Depending on the parameters you can get a perfect representation of the input
  # image, can anyone say what the problem with this is?
  # Now how do we use the feature vector output by the function.
  # We currently have feature_vector set to False, what shape does that give us?


  # So if we wanted a single feature vector for the whole image, i.e. a representation
  # of the entire image in blocks we can set the feature_vector to True. Try this.
  # What do you think a pitfall of this is? Can you come up witha better representation?

  # first let's reset the parameters to the default values but keep feature vector as True.
  # Essetially we have rgb.shape[0]//ppc[0] boxes on the row axis.
  # Then we boxes[rows]*boxes[cols]*orientations (we are negating the cells per block but you
  # can see how they are represented with feature_vector set to False)
  # So our feature shape should be:


  # So to use it we need to slice this feature vector up based on these parameters.
  # Which parameter are we breaking this vector up with?
  # Once you have worked this out, let's split the feature vector up based on that parameter.
  # As feat is a numpy array, we can use the np.array_split( x, number of splits )
  # how many splits do we have?


  # this outputs a list of numpy arrays, but we want a numpy matrix?


  # Okay so now we want a histogram of all the bins! This is why we needed a matrix.
  # For this we can use the array.sum() function, but we will specify which axis we
  # want to sum along. In this case we want to sum all the values in the rows so that
  # we get a vector of shape (4,). x.sum( axis=0 )


  # But we want a distribution of each of the bins. So let's divide our histogram by
  # the number of samples...


  # So this is our feature vector, what happens if you play with the parameters or
  # use different images? Do this in your own time.
  # Does a more coarse representation give a bitter feature vector?
  # Keep in  mind that you could have done this histogram representation using feature_vector set to False.
  # Try that in your own time.


  # Now let's look into LBPs
  # Local binary patterns are another way to do something similar. However, for LBPs
  # it's difficult to get the map that we got from the HOG variant.
  # I will now introduce a new concept to you: creating our own libraries!
  # In this working directory create a directory called "libs".
  # Put the file called "features.py" in that directory then open it and follow the
  # first part of the instructions (up until it tells you to come back here).

  # Now you should have the basic lbp feature extractor that prints the features.
  # So now we have to import this library? Try it...
  # Once  you have imported the library let's call the function using the hor_stripe.jpg
  # and the default parameters.


  # Once you have run it go back to features.py

  # You should now be finished the feature extractor. Let's create variables for the
  # main parameters, just use the defaults for now.


  # Now let's call the feature extractor again using these parameters.


  # Now let's plot the histogram of these values.


  # play with the input values and see what happens, what about different images?
  # Do this in your own time to familiarise yourself with everything.

"""
  ####### 3. Image normalisation
"""
if ex03:
  # In this exercise we will introduce a couple of new concepts.
  # In the lecture you have already seen how to normalise a dataset based on different
  # techniques, we will concentrate on min max norm and mu std norm. But, we will
  # do it over a batch of images.
  # The first thing we will do is load the images. As always there are a number of
  # ways to do this, and I will show you just one. First what is the root directory
  # of the sweet pepper images that you downloaded?


  # Once we have that we need to "import os", os is a powerful library for operating on
  # file systems. There are some great tools in this library like path creations, making directories
  # reading directory files and many more. For now I would like you to list the contents
  # of your root directory and print them to screen: print( os.listdir( <your directory location> ) )


  # From this you should see a list of of your images. But they probably won't appear in
  # alphanumeric order. Let's use the inbuilt python function "sorted" on the output list
  # and print that to screen


  # For what we are doing here it's not a big deal, but sometimes this is very important!
  # Okay so we can extract a list of the images in that directory.
  # Now we need to go through that list (for loop) load the images individually, and
  # get the following statistics of each channel of each image max, min, mean, std.
  # You will then calculate the global value of each. To load each image you will have to
  # concatenate root with the individual file name in the list. Here are two ways:
  # file = root + '/' + <file iteration> # where they are all strings
  # file = os.path.join( root, <file iteration> ) # can you work out what this is doing?
  # You will need to:
  #   iterate over the sorted list of file names
  #   create the filename (root and iteration)
  #   load the image
  #   put each channel of the image into another array or list (append)
  #   possibly convert to a numpy array
  #   calculate the statistics


  # Okay now we have a parameters. View the pdf to see the equations you need to implement.
  # Start with min max normalisation then move to the mu std normalisation.
  # For now we will just normalise a single image for your root directory. I will use
  # week05_sp_00.png


  # I will also turn the the individual values into arrays


  # Min Max normalisation = \frac{x-min}{max-min}


  # Now let's make sure this is actually normalised between 0 and 1 per channel.
  # Hint we want this per channel so you need to specify the axis, and you can specify
  # two axis at once as a tuple i.e. axis=(0,1) to calculate along the first and second
  # axis.


  # Mu std normalisation. \frac{X-mu}{std}


  # let's check the normalisation values...


  # You should notice that the normalisation range is different here. We have -x to y
  # which should be centered approximately around zero with a standard deviation of 1.
  # let's check that for just one of the channels.


  # So it's very close. As we get more data this will potentially converge better
  # to those default values.

"""
  ####### 4. PCA
"""
if ex04:
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


  # Create a varaible with the dependent variable


  # Now we need to create a matrix of our independent variables. Let's select:
  # longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population,
  # and median_house_value and put them in an (N*7) matrix.


  # But we have a problem! There are nan values in this matrix. PCA can't handle nan values
  # so we will remove them. We need two functions here, np.isnan and np.where. isnan locates
  # the nan values in a vector or matrix and where returns the exact location in matrix co-ordinates.
  # let's print np.where( np.isnan( X ) ) can you work out how to remove values for X based on these
  # co-ordinates? We really only need the rows. (You could also just remove the offending column but,
  # that kind of defeats the purpose here). You can use np.delete to delete the rows:
  # https://numpy.org/doc/stable/reference/generated/numpy.delete.html


  # Now we have our data we can play with PCA. As always we need to import something.
  # "from sklearn.decomposition import PCA" So we are now using sklearn!
  # Now we need to create the PCA object (PCA is a class), let's select PCA components of 2,
  # keeping in mind that the componenets need to be <= 7, but 7 doesn't really make
  # much sense here, we are trying to reduce the dimension. 2 Is just a random number, we'll
  # play with different values later.
  # obj = PCA( n_components=N ) # N = 2 for now


  # Now we need to fit the object to the X data we created.


  # now let's print the variance "explained_variance_ratio_"


  # So the first dimension contains 95% of the variance! That's a significant amount.
  # It means that first dimension contains the majority of the information, so this
  # COULD be a good representation. Go back and change N to see what happens.
