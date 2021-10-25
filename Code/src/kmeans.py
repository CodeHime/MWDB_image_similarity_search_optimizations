from distance_calculator import *
from sklearn.cluster import KMeans
from dimensionality_reducer_fns import *
import numpy as np
import pandas as pd
import os


class Kmeans:
    """
    Represents Kmeans dimension technique
    ...
    Attributes:
        k: int
            Number of reduced features

        X: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

    Methods:
        transform(X)
            Transforms and returns X in the latent semantic space and the latent semantics
    """

    def __init__(self, *args):
        """
        :param data_matrix: input matrix of shape (num_objects, num_features)
                Data matrix to be reduced
        :param k: Number of reduced features

        OR
        :param folder: folder containing LDA latent features

        """
        if (len(args)) == 2:
            # normal object instantiation with data_matrix and k
            imgs_slc = args[1]
            k = args[0]
            num_imgs = len(imgs_slc)
            arr_shp = imgs_slc[0][1].shape[0]
            imgs = np.zeros((num_imgs, arr_shp))
            for i in range(0, num_imgs):
                imgs[i] = imgs_slc[i][1]
            imgs_flat = imgs.reshape(num_imgs, arr_shp)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(imgs_flat)
            self.centers = kmeans.cluster_centers_
            self.new_object_map = np.zeros((num_imgs, k))
            self.weight = np.zeros((num_imgs))
            for i in range(0, num_imgs):
                for j in range(0, k):
                    self.new_object_map[i][j] = manhattan_fn(imgs_flat[i], self.centers[j])
                self.weight[i] = np.sum(self.new_object_map[i][:])
            # Since good latent semantics give high discrimination power
            # DESIGN_DECISION: variance or distance maximized? Distance as we have a center not a line or curve
            temp, self.sub_wt_pairs = get_sorted_matrix_on_weights(self.new_object_map, np.sum(self.weight, axis=0), return_order=True)
        elif (len(args)) == 1:
            self.centers = pd.read_csv(os.path.join(args[0], "centers.csv")).to_numpy()
            self.new_object_map = pd.read_csv(os.path.join(args[0], "new_object_map.csv")).to_numpy()
            self.weight = pd.read_csv(os.path.join(args[0], "weight.csv")).to_numpy()
            self.sub_wt_pairs = pd.read_csv(os.path.join(args[0], "sub_wt_pairs.csv")).to_numpy()
        else:
            raise Exception("Invalid object instantiation: parameters must be either <data_matrix:2D numpy>,<k:int> "
                            "or <folder:string>.")

    def get_decomposition(self):
        """
        Parameters:
            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced

        Returns:
            Transforms and returns X in the latent semantic space and the latent semantics
        """
        return self.centers

    def get_latent_features(self):
        """
        :return: centers
        """
        return self.centers

    def transform(self, data_matrix):
        """
        :param data_matrix: matrix to transform (query_objects, num_features)
        :return: pca transformed matrix
        """
        new_map = np.zeros((len(data_matrix), len(self.centers)))
        for j in range(len(self.centers)):
            new_map[j] = manhattan_fn(data_matrix[:len(self.centers)][j], self.centers[j])
        return new_map

    def get_obj_weight_pairs(self):
        """
        :return: objects
        :return: weights
        """
        return self.sub_wt_pairs

    def save(self, folder):
        """
        Save Kmeans to given folder
        """
        pd.DataFrame(self.centers).to_csv(os.path.join(folder, "centers.csv"), index=False)
        pd.DataFrame(self.new_object_map).to_csv(os.path.join(folder, "new_object_map.csv"), index=False)
        pd.DataFrame(self.weight).to_csv(os.path.join(folder, "weight.csv"), index=False)
        pd.DataFrame(self.sub_wt_pairs).to_csv(os.path.join(folder, "sub_wt_pairs.csv"), index=False)

    def get_top_k_matches(self, k, xq):
        return manhattan(self.new_object_map, k, self.transform([xq]))

