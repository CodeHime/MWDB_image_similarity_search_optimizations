from distance_calculator import *
from dimensionality_reducer_fns import *
import numpy as np
import pandas as pd
import os


class Svd:
    """
    Represents SVD dimension technique
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
            self.U, self.S, self.VT, self.sub_wt_pairs = svd_cust(args[0], k_num=args[1],return_order=True)
        elif (len(args)) == 1:
            self.U = pd.read_csv(os.path.join(args[0], "U.csv")).to_numpy()
            self.S = pd.read_csv(os.path.join(args[0], "S.csv")).to_numpy()
            self.VT = pd.read_csv(os.path.join(args[0], "VT.csv")).to_numpy()
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
        return self.U, self.S, self.VT

    def get_latent_features(self):
        """
        :return: U and S
        """
        return self.U, self.S

    def transform(self, data_matrix):
        """
        :param data_matrix: matrix to transform (query_objects, num_features)
        :return: pca transformed matrix
        """
        return np.dot(data_matrix, np.transpose(self.VT))

    def get_obj_weight_pairs(self):
        """
        :return: objects
        :return: weights
        """
        return self.sub_wt_pairs

    def save(self, folder):
        """
        Save SVD to given folder
        """
        pd.DataFrame(self.U).to_csv(os.path.join(folder, "U.csv"), index=False)
        pd.DataFrame(self.S).to_csv(os.path.join(folder, "S.csv"), index=False)
        pd.DataFrame(self.VT).to_csv(os.path.join(folder, "VT.csv"), index=False)
        pd.DataFrame(self.sub_wt_pairs).to_csv(os.path.join(folder, "sub_wt_pairs.csv"), index=False)

    def get_top_k_matches(self, k, xq):
        return euclidean(np.dot(self.U, self.S), k, self.transform([xq]))

