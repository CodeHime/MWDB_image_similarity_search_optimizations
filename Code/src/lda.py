from sklearn.decomposition import LatentDirichletAllocation as LDA
from dimensionality_reducer_fns import *
from distance_calculator import *


class Lda:
    """
    Represents LDA feature reduction class
    ...
    Attributes:

        Data_matrix: ndarray of shape (num_objects, num_features)
            Data matrix to be reduced

        k: int
            Number of reduced features

    Methods:

        compute_lda(X):
            Returns a Matrix of K latent features * N objects

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
            self.data_matrix = args[1]
            self.k = args[0]
            self.lda_ = LDA(n_components=self.k).fit(self.data_matrix)
            self.new_object_map = self.transform(self.data_matrix)
            # Take average as sum of all probabilities will always be 1
            print(np.average(self.lda_.components_, axis=0))
            self.sub_wt_pairs = []
            for i in range(len(np.average(self.lda_.components_, axis=0))):
                self.sub_wt_pairs.append([i, sorted(np.average(self.lda_.components_, axis=0))])
            # temp, self.sub_wt_pairs = get_sorted_matrix_on_weights(np.average(self.lda_.components_, axis=0), self.lda_.components_, return_order=True)
        elif (len(args)) == 1:
            # load object from folder
            self.lda_ = LDA()
            self.lda_.components_ = pd.read_csv(os.path.join(args[0], "components.csv")).to_numpy()
            self.lda_.exp_dirichlet_component_ = pd.read_csv(
                os.path.join(args[0], "exp_dirichlet_components.csv")).to_numpy()
            self.new_object_map = pd.read_csv(os.path.join(args[0], "new_object_map.csv")).to_numpy()
            self.lda_.set_params(n_components=self.lda_.components_.shape[0])
            self.lda_.doc_topic_prior_ = 1 / self.lda_.components_.shape[0]
            self.sub_wt_pairs = pd.read_csv(os.path.join(args[0], "sub_wt_pairs.csv")).to_numpy()
        else:
            raise Exception("Invalid object instantiation: parameters must be either <data_matrix:2D numpy>,<k:int> "
                            "or <folder:string>.")

    def transform(self, data_matrix):
        """
        :param data_matrix: matrix to transform (query_objects, num_features)
        :return: lda transformed matrix
        """
        return self.lda_.transform(data_matrix)

    def get_latent_features(self):
        """
        :return: components of lda(normalized)
        """
        return self.lda_.components_

    def get_vector_space(self):
        """
        :return: vector space of PCA
        """
        return self.new_object_map

    def get_obj_weight_pairs(self):
        """
        :param data_matrix: matrix to transform (query_objects, num_features)
        :return: objects - lda transformed matrix
        :return: weights - components of lda(normalized)
        """
        # TODO: HOW?
        return self.sub_wt_pairs

    def save(self, folder):
        """
        Save LDA to given folder
        """
        pd.DataFrame(self.lda_.components_).to_csv(os.path.join(folder, "components.csv"), index=False)
        pd.DataFrame(self.lda_.exp_dirichlet_component_).to_csv(os.path.join(folder, "exp_dirichlet_components.csv"), index=False)
        pd.DataFrame(self.new_object_map).to_csv(os.path.join(folder, "new_object_map.csv"), index=False)
        pd.DataFrame(self.sub_wt_pairs).to_csv(os.path.join(folder, "sub_wt_pairs.csv"), index=False)

    def get_top_k_matches(self, k, xq):
        # DESIGN_DECISIONS: KL divergence as it is a probability distribution
        return kl_divergence(self.new_object_map, k, self.transform([xq]))
