import pandas as pd

from distance_calculator import *
from latent_features_extractor import *


def get_similarity_matrix(xb, k, base_dir, features_dir, technique="pca", sim_type="subject", sim_method="euclidean"):
    """
    Get similarity matrix of features

    """
    similarity_matrix = []
    # DESIGN_DECISION: why type as features and not image type? Because we are already associating image type in later stages.
    if sim_type != "subject":
        xb = xb.transpose()
        xy_id_dict = get_type_from_ids(features_dir, range(xb.shape[0]))
    else:
        xy_id_dict = get_subjects_from_ids(features_dir, range(xb.shape[0]))

    # DESIGN_DECISION: why euclidean,etc.
    for xq in xb:
        similarity_matrix.append(euclidean_fn(xb, xq))

    xy_similarity_matrix = []
    xy_id_mapping = {}
    index = 0
    for id, indexes in xy_id_dict.items():
        i_matrix = []
        for i in indexes:
            i_matrix.append(similarity_matrix[i])
        xy_similarity_matrix.append(np.sum(i_matrix, axis=0))
        xy_id_mapping.update({index: id})
        index+=1

    similarity_matrix = xy_similarity_matrix
    # # normalize array
    # similarity_matrix = np.array(similarity_matrix)
    # norm = np.linalg.norm(similarity_matrix)
    # similarity_matrix = np.ones(similarity_matrix.shape) - similarity_matrix / norm

    if not os.path.isdir(os.path.join(base_dir, config['Phase2']['similarity_dir'])):
        os.makedirs(os.path.join(base_dir, config['Phase2']['similarity_dir']))

    pd.DataFrame(similarity_matrix) \
        .to_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], sim_type + ".csv"))
    pd.DataFrame(xy_id_mapping) \
        .to_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], sim_type + "id_map.csv"))

    if technique == "":
        return np.array(similarity_matrix)

    # Perform technique and return top k latent semantics
    sub_wt_pairs = perform_dimensionality_reductions(similarity_matrix, k, technique, base_dir)

    return np.array(similarity_matrix)
