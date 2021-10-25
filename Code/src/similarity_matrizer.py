import pandas as pd

from distance_calculator import *
from latent_features_extractor import *


def get_similarity_matrix(xb, k, base_dir, technique="pca", sim_type="subject", sim_method="euclidean"):
    """
    Get similarity matrix of features

    """
    similarity_matrix = []
    # DESIGN_DECISION: why type as features and not image type? Because we are already associating image type in later stages.
    if sim_type != "subject":
        xb = xb.transpose()

    # DESIGN_DECISION: why euclidean,etc.
    for xq in xb:
        similarity_matrix.append(euclidean_fn(xb, xq))
    # # normalize array
    # similarity_matrix = np.array(similarity_matrix)
    # norm = np.linalg.norm(similarity_matrix)
    # similarity_matrix = np.ones(similarity_matrix.shape) - similarity_matrix / norm

    if not os.path.isdir(os.path.join(base_dir, config['Phase2']['similarity_dir'])):
        os.makedirs(os.path.join(base_dir, config['Phase2']['similarity_dir']))

    pd.DataFrame(similarity_matrix)\
        .to_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], sim_type + ".csv"))

    if technique == "":
        return np.array(similarity_matrix)

    # Perform technique and return top k latent semantics
    sub_wt_pairs = perform_dimensionality_reductions(similarity_matrix, k, technique, base_dir)

    return np.array(similarity_matrix)
