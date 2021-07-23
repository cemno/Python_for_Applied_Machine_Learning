"""
assignment
"""
import os
from time import time
import skimage.color
from skimage.io import imsave, imread
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve as prc
from libs_assignment.data_loading import data_task_1
from libs_assignment.pixel_segmentation import MultivariateGaussian, create_mask, KMeans
import regex as re
import pickle

# Various parameters to change for working
#general
data_loc = "data/PAML_data"
img_path = "data/data_assignment/test_images" # Path for input image to create mask from
# if outputpath is None: #create folder mask, else use supplied outputpath (do nothing)
output_path = os.path.join(img_path, "mask") # Optional, can be same or different
verbose = False
colourspace = "hsv"
# KMeans
solution_1 = True
only_maximum_cluster = True
# Multivariate Gaussian
solution_2 = False

# first the data will be loaded.
#data for number one:
picture_background, picture_red, picture_yellow = data_task_1(data_loc, verbose= verbose)


if solution_1:
    start = time()
    # Create training data
    train_data_combined = np.vstack([picture_background["train"], picture_red["train"], picture_yellow["train"]])
    train_data_bg_red = np.vstack([picture_background["train"], picture_red["train"]])
    # Crate validation data
    validation_data_red = picture_red["validation"]
    validation_data_yellow = picture_yellow["validation"]

    # Create evaluation data set for background and red data
    evaluation_data_bg_red = np.vstack([picture_background["evaluation"], picture_red["evaluation"]])
    evaluation_data_all = np.vstack([picture_background["evaluation"], picture_red["evaluation"],
                                     picture_yellow["evaluation"]])

    # Extend evaluation labels with labels for yellow dataset
    eval_labels_bg_red = np.concatenate([np.zeros(len(picture_background["evaluation"]), dtype=int),
                                         np.ones(len(picture_red["evaluation"]), dtype=int)])
    eval_labels_all = np.concatenate([eval_labels_bg_red, np.full(len(picture_yellow["evaluation"]), 2)])

    ## Changing colour space if necessary
    if not colourspace == "rgb":
        train_data_combined = skimage.color.convert_colorspace(train_data_combined, fromspace="rgb",
                                                               tospace=colourspace)
        train_data_bg_red = skimage.color.convert_colorspace(train_data_bg_red, fromspace="rgb",
                                                               tospace=colourspace)
        validation_data_red = skimage.color.convert_colorspace(validation_data_red, fromspace="rgb",
                                                               tospace=colourspace)
        validation_data_yellow = skimage.color.convert_colorspace(validation_data_yellow, fromspace="rgb",
                                                                  tospace=colourspace)
        evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace="rgb",
                                                                  tospace=colourspace)
        evaluation_data_all = skimage.color.convert_colorspace(evaluation_data_all, fromspace="rgb",
                                                               tospace=colourspace)

    # Fit data
    kmeans_bg_red = KMeans(8)
    kmeans_bg_red.fit(train_data_bg_red)
    kmeans_all = KMeans(8) # 6 to 9 is ok, 8 is best
    kmeans_all.fit(train_data_combined)

    # Predict red and yellow validation data set and choosing cluster with the most red occurrences:
    kmeans_bg_red.validate_cluster(validation_data_red, "red", only_max=only_maximum_cluster)
    kmeans_all.validate_cluster(validation_data_red, "red", only_max=only_maximum_cluster)
    kmeans_all.validate_cluster(validation_data_yellow, "yellow", only_max=only_maximum_cluster)
    if verbose:
        print("Red cluster: {}\nYellow cluster: {}".format(kmeans_all.red, kmeans_all.yellow))

    # Visualize colours of KMeans to get a better understanding of the data
    if verbose:
        kmeans_bg_red.visualize_cluster(colourspace)
        kmeans_all.visualize_cluster(colourspace)


    # Read in image and transfrom to mask
    for img in [item for item in os.listdir(img_path) if re.search('\.png$', item)]:
        input_img = imread(os.path.join(img_path, img))
        mask, _ = create_mask(input_img, kmeans_all, red_cluster=kmeans_all.red, colourspace=colourspace, yellow_cluster=kmeans_all.yellow, verbose=verbose)
        mask_name = "msk_" + img
        os.makedirs(output_path, exist_ok=True)
        imsave(os.path.join(output_path, mask_name), mask)

    # Create labels based on prediction on the evaluation dataset for background and red
    pred_eval_bg_red = kmeans_bg_red.prediction_labels(evaluation_data_bg_red)

    pred_eval_all = kmeans_all.prediction_labels(evaluation_data_all)

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
    #plt.show()

    # calculate the two accuracy scores. and confusion matrices
    acc_lin = accuracy_score( eval_labels_bg_red, pred_eval_bg_red )
    print( 'Accuracy of the bg and red data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( eval_labels_bg_red, pred_eval_bg_red ) )

    # calculate the two accuracy scores. and confusion matrices
    acc_lin = accuracy_score( eval_labels_all, pred_eval_all )
    print( 'Accuracy of the bg, red and yellow data is: {:0.04f}'.format( acc_lin ) )
    print( confusion_matrix( eval_labels_all, pred_eval_all ) )

    end = time()
    dtime = end - start
    print("Durchlauf dauert: {:0.02f}s".format(dtime))
    # Write KMeans objects to hard drive
    # with open(os.path.join(data_loc, "kmeans_all.pkl"), "wb") as file:
    #     pickle.dump(kmeans_all, file = file)
    # with open(os.path.join(data_loc, "kmeans_bg_red.pkl"), "wb") as file:
    #     pickle.dump(kmeans_bg_red, file = file)


if solution_2:
    start = time()
    # Create train data
    train_data_bg = picture_background["train"]
    train_data_red =picture_red["train"]
    # Create validation data
    validation_data_bg_red = np.vstack([picture_background["validation"], picture_red["validation"]])
    # Create evaluation
    evaluation_data_bg_red = np.vstack([picture_background["evaluation"], picture_red["evaluation"]])
    # Create labels
    len_bg = len(picture_background["validation"])
    len_red = len(picture_red["validation"])
    valid_labels_bg_red = np.concatenate([np.zeros(len_bg, dtype=int), np.ones(len_red, dtype=int)])
    if not colourspace == "rgb":
        train_data_bg = skimage.color.convert_colorspace(train_data_bg, fromspace="rgb", tospace=colourspace)
        train_data_red = skimage.color.convert_colorspace(train_data_red, fromspace="rgb", tospace=colourspace)
        validation_data_bg_red = skimage.color.convert_colorspace(validation_data_bg_red, fromspace="rgb",
                                                                  tospace=colourspace)
        evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace="rgb",
                                                                  tospace=colourspace)

    mvg_bg = MultivariateGaussian()
    mvg_bg.train(train_data_bg)
    mvg_red = MultivariateGaussian()
    mvg_red.train(train_data_red)


    loglike = np.zeros((validation_data_bg_red.shape[0], 2))
    loglike[:, 0] = mvg_bg.log_likelihood(validation_data_bg_red)
    loglike[:, 1] = mvg_red.log_likelihood(validation_data_bg_red)

    classified = np.argmax(loglike, axis=1)

    mvg = [mvg_bg, mvg_red]
    # Read in image and transfrom to mask
    for img in [item for item in os.listdir(img_path) if re.search('\.png$', item)]:
        input_img = imread(os.path.join(img_path, img))
        mask, _ = create_mask(input_img, mvg, red_cluster=np.full(1,1, dtype = int), colourspace=colourspace,
                              verbose=verbose)
        mask_name = "msk_" + img
        os.makedirs(output_path, exist_ok=True)
        imsave(os.path.join(output_path, mask_name), mask)

    acc = accuracy_score(valid_labels_bg_red, classified)
    print('Accuracy of the MVGs is:', acc)
    print( confusion_matrix( valid_labels_bg_red, classified ) )

    # now the f1score stuff.
    p, r, t = prc(valid_labels_bg_red, classified)
    # print( 't', len( t ) )
    f1 = 2*p*r/(p+r+0.0000001)
    am = np.argmax( f1 )
    print("Validation data set:\nPrecision: {:0.04f}, Recall: {:0.04f}\n".format(p[am], r[am]))
    plt.figure()
    plt.plot()
    plt.plot( r, p )
    plt.plot( r[am], p[am], 'r*' )
    plt.title( 'Background and red data Precision Recall: F1-score of {}'.format( f1[am] ) )
    plt.show()

    # Test threshold with evaluation data set
    loglike = np.zeros((evaluation_data_bg_red.shape[0], 2))

    loglike[:, 0] = mvg_bg.log_likelihood(evaluation_data_bg_red)
    loglike[:, 1] = mvg_red.log_likelihood(evaluation_data_bg_red)

    classified = np.argmax(loglike, axis=1)

    len_bg = len(picture_background["evaluation"])
    len_red = len(picture_red["evaluation"])
    eval_labels_bg_red = np.concatenate([np.zeros(len_bg, dtype=int), np.ones(len_red, dtype=int)])

    p, r, t = prc(eval_labels_bg_red, classified)
    print("Evaluation data set:\nPrecision: {:0.04f}, Recall: {:0.04f}\n".format(p[am], r[am]))
    end = time()
    dtime = end - start
    print("Durchlauf dauert: {:0.02f}s".format(dtime))
    # with open(os.path.join(data_loc, "mvg.pkl"), "wb") as file:
    #      dict = {"mvg_bg": mvg_bg, "mvg_red": mvg_red}
    #      pickle.dump(dict, file = file)