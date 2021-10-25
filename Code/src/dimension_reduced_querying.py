from pca import *
from svd import *
from lda import *
from kmeans import *
from config import *


def get_top_k_matches_latent_space(query_matrix, k, technique, base_dir):
  if technique == "pca":
    obj = Pca(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "svd":
    obj = Svd(k, os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "lda":
    obj = Lda(k, os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "kmeans":
    obj = Kmeans(k, os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  else:
      raise ValueError("No such technique exists.")
  return obj.get_top_k_matches(k, query_matrix)


