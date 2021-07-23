"""
assignment
"""
import argparse
import os
import skimage.color
from skimage.io import imsave, imread
import numpy as np
from libs_assignment.data_loading import data_task_1
from libs_assignment.pixel_segmentation import create_mask
import regex as re
import pickle

parser = argparse.ArgumentParser( description='Extracting command line arguments', add_help=True )
parser.add_argument( '--whatrun', action='store', required=True )
parser.add_argument( '--dataloc', action='store', required=True)
parser.add_argument( '--input_img_path', action='store', required=True)
parser.add_argument( '--output_path', action='store', required=False, default = None)
parser.add_argument( '--verbosity', action='store', default=False )
flags = parser.parse_args()

kmeans_bg_red = False
kmeans_all = False
mvg = False


if flags.whatrun == "kmeans_bg_red":
    kmeans_bg_red = True
if flags.whatrun == "kmeans_all":
    kmeans_all = True
if flags.whatrun == "mvg":
    mvg = True

verbose = flags.verbosity
only_maximum_cluster = True
data_loc = flags.dataloc
colourspace = "hsv" #has to be changed with the kmeans object
img_path = flags.input_img_path
if flags.output_path is None:
    output_path = os.path.join(img_path, "mask")
else:
    output_path = flags.output_path

# first the data will be loaded.
picture_background, picture_red, picture_yellow = data_task_1(data_loc, verbose= verbose)


if kmeans_bg_red:
    # Create training data
    train_data_bg_red = np.vstack([picture_background["train"], picture_red["train"]])

    # Crate validation data
    validation_data_red = picture_red["validation"]

    # Create evaluation data set for background and red data
    evaluation_data_bg_red = np.vstack([picture_background["evaluation"], picture_red["evaluation"]])

    # Extend evaluation labels with labels for yellow dataset
    eval_labels_bg_red = np.concatenate([np.zeros(len(picture_background["evaluation"]), dtype=int),
                                         np.ones(len(picture_red["evaluation"]), dtype=int)])

    ## Changing colour space if necessary
    if not colourspace == "rgb":
        train_data_bg_red = skimage.color.convert_colorspace(train_data_bg_red, fromspace="rgb",
                                                               tospace=colourspace)
        validation_data_red = skimage.color.convert_colorspace(validation_data_red, fromspace="rgb",
                                                               tospace=colourspace)
        evaluation_data_bg_red = skimage.color.convert_colorspace(evaluation_data_bg_red, fromspace="rgb",
                                                                  tospace=colourspace)

    # Load KMeans object
    with open(os.path.join(data_loc, "kmeans_bg_red.pkl"), "rb") as file:
        kmeans_bg_red = pickle.load(file = file)

    # Visualize colours of KMeans to get a better understanding of the data
    if verbose:
        kmeans_bg_red.visualize_cluster(colourspace)

    # Read in image and transfrom to mask
    for img in [item for item in os.listdir(img_path) if re.search('\.png$', item)]:
        input_img = imread(os.path.join(img_path, img))
        mask, _ = create_mask(input_img, kmeans_bg_red, red_cluster=kmeans_bg_red.red, colourspace=colourspace, verbose=verbose) # underscore gives dictionary with pixel count for each type
        mask_name = "msk_" + img
        os.makedirs(output_path, exist_ok=True)
        imsave(os.path.join(output_path, mask_name), mask)

if kmeans_all:
    # Create training data
    train_data_combined = np.vstack([picture_background["train"], picture_red["train"], picture_yellow["train"]])

    # Crate validation data
    validation_data_red = picture_red["validation"]
    validation_data_yellow = picture_yellow["validation"]

    # Create evaluation data set for background and red data
    evaluation_data_all = np.vstack([picture_background["evaluation"], picture_red["evaluation"],
                                     picture_yellow["evaluation"]])

    # Extend evaluation labels with labels for yellow dataset
    eval_labels_bg_red = np.concatenate([np.zeros(len(picture_background["evaluation"]), dtype=int),
                                         np.ones(len(picture_red["evaluation"]), dtype=int),
                                        np.full(len(picture_yellow["evaluation"]), 2, dtype=int)])
    ## Changing colour space if necessary
    if not colourspace == "rgb":
        train_data_combined = skimage.color.convert_colorspace(train_data_combined, fromspace="rgb",
                                                               tospace=colourspace)
        validation_data_red = skimage.color.convert_colorspace(validation_data_red, fromspace="rgb",
                                                               tospace=colourspace)
        validation_data_yellow = skimage.color.convert_colorspace(validation_data_yellow, fromspace="rgb",
                                                                  tospace=colourspace)
        evaluation_data_all = skimage.color.convert_colorspace(evaluation_data_all, fromspace="rgb",
                                                               tospace=colourspace)

    # Load KMeans object
    with open(os.path.join(data_loc, "kmeans_all.pkl"), "rb") as file:
        kmeans_all = pickle.load(file=file)

    # Visualize colours of KMeans to get a better understanding of the data
    if verbose:
        kmeans_all.visualize_cluster(colourspace)

    # Read in image and transfrom to mask
    for img in [item for item in os.listdir(img_path) if re.search('\.png$', item)]:
        input_img = imread(os.path.join(img_path, img))
        mask, _ = create_mask(input_img, kmeans_all, red_cluster=kmeans_all.red, colourspace=colourspace, # underscore gives dictionary with pixel count for each type
                              yellow_cluster=kmeans_all.yellow, verbose=verbose)
        mask_name = "msk_" + img
        os.makedirs(output_path, exist_ok=True)
        imsave(os.path.join(output_path, mask_name), mask)

if mvg:
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
    # Import precalculated MVG Objects
    with open(os.path.join(data_loc, "mvg.pkl"), "rb") as file:
        mvg = pickle.load(file = file)
    mvg_bg = mvg["mvg_bg"]
    mvg_red = mvg["mvg_red"]
    mvg = [mvg_bg, mvg_red]
    # Read in image and transfrom to mask
    for img in [item for item in os.listdir(img_path) if re.search('\.png$', item)]:
        input_img = imread(os.path.join(img_path, img))
        mask, _ = create_mask(input_img, mvg, red_cluster=np.full(1,1, dtype = int), colourspace=colourspace, verbose=verbose) # underscore gives dictionary with pixel count for each type
        mask_name = "msk_" + img
        os.makedirs(output_path, exist_ok=True)
        imsave(os.path.join(output_path, mask_name), mask)