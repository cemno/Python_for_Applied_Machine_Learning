import numpy as np
from skimage.color import convert_colorspace
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def cluster_occurrences(classification_array, only_max=True):
    """ Returns a np.array with all clusters where red has occurred at least ones. With only_max = True the np.array only contains one number for the cluster with the most occurrences."""
    occurrences = {}
    for num in np.unique(classification_array):
        # Fill dictionary with occurrences per class
        occ = classification_array.tolist().count(num)
        occurrences.update({num: occ})
    # Class with most occurrences - This is debatable but probably best for low cluster numbers.
    if only_max:
        max_occ = max(zip(occurrences.values(), occurrences.keys()))[1]
        return np.asarray(max_occ, dtype=int)
    else:
        occurrences = np.asarray(list(occurrences.keys()), dtype=int)
        return occurrences

def visualizing_kmeans_cluster(kmeans,  colourspace = "rgb"):
    """
    Print a pie chart with the area as percentage of the training data matching to each cluster. Pie chart colorized by rgb color.
    :param kmeans:
    :param colourspace:
    :return:
    """
    labels_all = kmeans.labels_
    labels=list(labels_all)
    centroid = kmeans.cluster_centers_
    if not colourspace == "rgb":
        centroid = convert_colorspace(centroid, fromspace=colourspace, tospace="rgb")
        centroid = centroid * 255
    percent=[]
    for i in range(len(centroid)):
        j=labels.count(i)
        j=j/(len(labels))
        percent.append(j)

    plt.pie(percent,colors=np.array(centroid/255),labels=np.arange(len(centroid)))
    plt.show()



def create_mask(img, cluster_methode, red_cluster, yellow_cluster=None, colourspace="rgb", verbose=False):
    """
    Creates a mask of a given image based on the KMeans clusters
    :param img: Input image with shape of (w, h, 3).
    :param cluster_methode: Trained cluster object(s), use list for multiple objects.
    :param red_cluster: numpy.array containing all classes of the cluster methode that shall be converted to red
    :param yellow_cluster: numpy.array containing all classes of the cluster methode that shall be converted to yellow
    :param colourspace: Change colour space to match training data.
    :param verbose: Prints percentage of Pixels Vegetation/red(/yellow) and outputs input image and mask
    :return: returns the msk (same format as input) and a dictionary with each pixel count. keys: "vegetation", "red", "yellow"
    """

    yellow = np.array([255, 255, 0], dtype=int)
    red = np.array([255, 0, 0], dtype=int)
    green = np.array([0, 255, 0], dtype=int)
    # Check if yellow cluster is supplied, if not set to same as red cluster (gets overwritten later on)
    no_yellow = False
    if yellow_cluster is None:
        no_yellow = True
        yellow_cluster = red_cluster
        yellow = red

    # Transform image to 2D Array (instead of 3D) and convert colourspace if necessary
    img_transformed = img.reshape(-1, img[0, 0, :].size)
    if not colourspace == "rgb":
        img_transformed = convert_colorspace(img_transformed, fromspace="rgb", tospace=colourspace)
    if isinstance(cluster_methode, KMeans):
        index_classes = cluster_methode.predict(img_transformed)
    elif all(isinstance(obj, MultivariateGaussian) for obj in cluster_methode):
        loglike = np.zeros((img_transformed.shape[0], len(cluster_methode)))
        for i, mvg in enumerate(cluster_methode):
            loglike[:, i] = mvg.log_likelihood(img_transformed)
        index_classes = np.argmax(loglike, axis=1)
    msk_zero = np.zeros(img_transformed.shape, dtype=int)
    yellow_pixel, red_pixel = 0, 0
    for i in range(msk_zero.shape[0]):
        if index_classes[i] in yellow_cluster:
            msk_zero[i] = yellow
            yellow_pixel += 1
        elif index_classes[i] in red_cluster:
            msk_zero[i] = red
            red_pixel += 1
        else:
            msk_zero[i] = green
    msk = msk_zero.reshape(img.shape)
    if verbose:
        print("Red pixel: {:0.01f}%, Yellow pixel: {:0.01f}%, Background pixel: {:0.01f}%".format(
            red_pixel * 100 / msk_zero.shape[0], yellow_pixel * 100 / msk_zero.shape[0],
            (msk_zero.shape[0] - yellow_pixel - red_pixel) * 100 / msk_zero.shape[0]))
        plt.figure()
        plt.subplot(121)
        plt.imshow(msk, vmin=0, vmax=255)
        plt.subplot(122)
        plt.imshow(img)
        plt.show()
    if no_yellow:
        vegetation = msk_zero.shape[0] - yellow_pixel
        pixel_count = {"vegetation": vegetation, "red": yellow_pixel}
    else:
        vegetation = msk_zero.shape[0] - red_pixel - yellow_pixel
        pixel_count = {"vegetation": vegetation, "red": red_pixel, "yellow": yellow_pixel}
    return msk, pixel_count

class KMeans(KMeans):
    def validate_cluster(self, validation_data, colour: str, only_max = True):
        """
        :param validation_data: supply as numpy.array
        :param colour: The colour you want to identify as String. Can be access via KMeans_obj.your_colour, use "red" or "yellow"
        :param only_max: With only_max = True the np.array only contains one number for the cluster with the most occurrences.
        """
        pred_cluster = self.predict(validation_data)
        setattr(self, colour, cluster_occurrences(pred_cluster, only_max))

    def prediction_labels(self, evaluation_data_set):
        pred_eval = self.predict(evaluation_data_set)
        tmp = np.zeros(pred_eval.shape, dtype=int)
        if hasattr(self, "yellow"):
            for i, cluster in enumerate(pred_eval):
                if cluster in self.red:
                    tmp[i] = 1
                elif cluster in self.yellow:
                    tmp[i] = 2
            return tmp
        else:
            for i, cluster in enumerate(pred_eval):
                if cluster in self.red:
                    tmp[i] = 1
            return tmp
    def visualize_cluster(self, colourspace = "rgb"):
        visualizing_kmeans_cluster(self, colourspace=colourspace)


class MultivariateGaussian:
    def __init__(self, mu=[], sigma=[]):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        if (not (self.sigma == []) and (not (self.mu == []))):
            self._precalculations()

    def _precalculations(self):
        n = self.mu.shape[1]
        self.inv_sigma = np.linalg.inv(self.sigma)
        log_two_pi = -n / 2. * np.log(2 * np.pi)
        log_det = -0.5 * np.linalg.slogdet(self.sigma)[1]
        self.constant = log_two_pi + log_det

    def log_likelihood(self, X):
        m, n = X.shape
        llike = np.zeros((m,))
        resids = X - self.mu
        for i in range(m):
            llike[i] = self.constant - resids[i, :] @ self.inv_sigma @ resids[i, :].T
        return llike

    def train(self, X):
        m, n = X.shape
        mu = np.sum(X, axis=0) / float(m)
        mu = np.reshape(mu, (1, n))
        norm_X = X - mu
        sigma = (norm_X.T @ norm_X) / float(m)
        self.mu = mu
        self.sigma = sigma
        self._precalculations()