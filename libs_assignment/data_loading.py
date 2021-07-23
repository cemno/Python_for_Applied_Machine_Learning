import pickle
import os.path

def data_task_1(path, verbose = False):
    """
    :param path: Path to folder containing the .pkl data for Task 1.
    :param verbose: Print out the Structure of the data set.
    :return: Returns the dictionaries for the background, red and yellow data set, respectively.
    """
    if not verbose:
        with open(os.path.join(path, "Q1_BG_dict.pkl"), "rb") as sf:
            picture_background = pickle.load(sf)

        with open(os.path.join(path, "Q1_Red_dict.pkl"), "rb") as sf:
            picture_red = pickle.load(sf)

        with open(os.path.join(path, "Q1_Yellow_dict.pkl"), "rb") as sf:
            picture_yellow = pickle.load(sf)
    else:
        with open(os.path.join(path, "Q1_BG_dict.pkl"), "rb") as sf:
            picture_background = pickle.load(sf)
        print("Background data loaded.")

        with open(os.path.join(path, "Q1_Red_dict.pkl"), "rb") as sf:
            picture_red = pickle.load(sf)
        print("Red data loaded.")

        with open(os.path.join(path, "Q1_Yellow_dict.pkl"), "rb") as sf:
            picture_yellow = pickle.load(sf)
        print("Yellow data loaded.")

        for fname in [os.path.join(path, "Q1_BG_dict.pkl"),  os.path.join(path, "Q1_Red_dict.pkl"), os.path.join(path, "Q1_Yellow_dict.pkl")]:
            print("Schema of data set: ", fname)
            with open(fname,'rb') as fp:
                data = pickle.load(fp)
                print("The sets in the dictionary are:", data.keys())
                print("The size of the data matrix X for each set is:", data['train'].shape, data['validation'].shape, data['evaluation'].shape)

    return picture_background, picture_red, picture_yellow

