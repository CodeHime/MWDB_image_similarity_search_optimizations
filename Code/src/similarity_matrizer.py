import pandas as pd

from distance_calculator import *
from latent_features_extractor import *


def most_frequent(List):
    counter = 0
    freq_dict = {}
    for i in List:
        freq_dict.update({i: freq_dict.get(i,0)+1})

    nums = []
    freq = sorted(freq_dict.values())[-1]
    for k, v in freq_dict.items():
        if v == freq:
            nums.append(k)
    return nums


def get_similarity_matrix(xb, k, base_dir, features_dir, technique="pca", sim_type="subject", sim_method="euclidean"):
    """
    Get similarity matrix of features

    """
    similarity_matrix = []
<<<<<<< HEAD
    # DESIGN_DECISION: Type is noise type as we are interested in the similarities between images of the same type
    if sim_type != "subject":
        # xb = xb.transpose()
=======
    # DESIGN_DECISION: why type as features and not image type? Because we are already associating image type in later stages.
    if sim_type != "subject":
        xb = xb.transpose()
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
        xy_id_dict = get_type_from_ids(features_dir, range(xb.shape[0]))
    else:
        xy_id_dict = get_subjects_from_ids(features_dir, range(xb.shape[0]))

    # DESIGN_DECISION: Average over type/similarity
    xy_similarity_matrix = []
    xy_id_mapping = {}
    index = 0
    for id, indexes in xy_id_dict.items():
        i_matrix = []
        for i in indexes:
            i_matrix.append(xb[i])
        xy_similarity_matrix.append(np.average(i_matrix, axis=0))
        xy_id_mapping.update({index: id})
        index += 1

<<<<<<< HEAD
    # DESIGN_DECISION: Dot product to get similarity matrix
    similarity_matrix = np.dot(xy_similarity_matrix, np.array(xy_similarity_matrix).transpose())
    # # normalize array to rescale to a proper range in 0 to 1
=======
    similarity_matrix = np.dot(xy_similarity_matrix, np.array(xy_similarity_matrix).transpose())
    # DESIGN_DECISION: why euclidean,etc.
    # for xq in xy_similarity_matrix:
    #     similarity_matrix.append(euclidean_fn(xb, xq))
    # similarity_matrix = xy_similarity_matrix
    # # normalize array
    # similarity_matrix = np.array(similarity_matrix)
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
    # norm = np.linalg.norm(similarity_matrix)
    # similarity_matrix = np.ones(similarity_matrix.shape) - similarity_matrix / norm

    if not os.path.isdir(os.path.join(base_dir, config['Phase2']['similarity_dir'])):
        os.makedirs(os.path.join(base_dir, config['Phase2']['similarity_dir']))

    pd.DataFrame(similarity_matrix) \
        .to_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], sim_type + ".csv"), header=False, index=False)
    pd.DataFrame(np.array(list(xy_id_mapping.items()))) \
        .to_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], sim_type + "id_map.csv"), header=False, index=False)

    if technique == "":
        return np.array(similarity_matrix)

    # Perform technique and return top k latent semantics
    sub_wt_pairs = perform_dimensionality_reductions(similarity_matrix, k, technique, base_dir)

    return np.array(similarity_matrix)
