from distance_calculator import *

def perform_dimensionality_reductions(matrix, k, technique):
  if technique=="pca":
    obj = Pca(k,matrix)
  elif technique=="svd":
    obj = Svd(k,matrix)
  else:
    obj = Lda(k,matrix)
  obj.save("")
  return obj.get_obj_weight_pairs()

def get_similarity_matrix(xb, technique="pca", method="euclidean"):
    """
    Get similarity matrix of features

    """
    similarity_matrix = []
    for xq in xb:
        similarity_matrix.append(euclidean_fn(xb, xb.shape[0], xq))
    # Perform technique and return top k latent semantics

    return np.array(similarity_matrix)
