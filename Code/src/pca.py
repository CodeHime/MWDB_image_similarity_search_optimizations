from distance_calculator import *
from dimensionality_reducer_fns import *
import pandas as pd
import os


class Pca:
    """
    Represents PCA dimension technique
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
        Parameters:
            k: int
                Number of reduced features

            X: ndarray of shape (num_objects, num_features)
                Data matrix to be reduced
        """
        if (len(args)) == 2:
            X = args[1]
            k = args[0]
            self.x_ = np.array(X, dtype=np.float32)
            self.features_ = self.x_.shape[1]

            self.x_covariance_ = np.cov(self.x_)
            self.eigen_values_, self.eigen_vectors_ = eig(self.x_covariance_)

            temp, self.sub_wt_pairs = get_sorted_matrix_on_weights(self.eigen_values_,
                                                                   self.eigen_vectors_.astype(float),
                                                                   return_order = True)
            self.eigen_vectors_ = self.eigen_vectors_.transpose()[::-1]
            self.eigen_values_ = self.eigen_values_[::-1]

            self.u_, self.s_, self.u_transpose_ = self.eigen_vectors_[:k].astype(float).transpose(), \
                                                  np.diag(self.eigen_values_[:k].astype(np.float)), \
                                                  self.eigen_vectors_[:k].astype(float)
        elif (len(args)) == 1:
            self.u_ = pd.read_csv(os.path.join(args[0], "U.csv")).to_numpy()
            self.u_transpose_ = self.u_.transpose()
            self.s_ = pd.read_csv(os.path.join(args[0], "S.csv")).to_numpy()
            self.x_ = pd.read_csv(os.path.join(args[0], "X.csv")).to_numpy()
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
        return self.u_, self.s_, self.u_transpose_

    def get_latent_features(self):
        """
        :return: components of PCA
        """
        return self.u_, self.s_

    def get_vector_space(self):
        """
        :return: vector space of PCA
        """
        return np.dot(self.u_, self.s_)

    def transform(self, data_matrix):
        """
        :param data_matrix: matrix to transform (query_objects, num_features)
        :return: pca transformed matrix
        """
        Q = np.concatenate((self.x_, data_matrix), axis=0)
        return np.dot(np.cov(Q)[-1][:-1], self.u_)

    def get_obj_weight_pairs(self):
        """
        :return: objects
        :return: weights
        """
        return self.sub_wt_pairs

    def save(self, folder):
        """
        Save PCA to given folder
        """
        pd.DataFrame(self.u_).to_csv(os.path.join(folder, "U.csv"), index=False)
        pd.DataFrame(self.x_).to_csv(os.path.join(folder, "X.csv"), index=False)
        pd.DataFrame(self.s_).to_csv(os.path.join(folder, "S.csv"), index=False)
        pd.DataFrame(self.sub_wt_pairs).to_csv(os.path.join(folder, "sub_wt_pairs.csv"), index=False)

    def get_top_k_matches(self, k, xq):
        return euclidean(np.dot(self.u_, self.s_), k, self.transform([xq]))
