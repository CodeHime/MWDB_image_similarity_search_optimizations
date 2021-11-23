from pca import *
from svd import *
from lda import *
from kmeans import *
from config import *


<<<<<<< HEAD
def get_saved_latent_object(technique, base_dir):
=======
def get_top_k_matches_latent_space(query_matrix, k, technique, base_dir):
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
  if technique == "pca":
    obj = Pca(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "svd":
    obj = Svd(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "lda":
    obj = Lda(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  elif technique == "kmeans":
    obj = Kmeans(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  else:
      raise ValueError("No such technique exists.")
<<<<<<< HEAD
  return obj


def get_top_k_matches_latent_space(query_matrix, k, technique, base_dir):
  obj = get_saved_latent_object(technique, base_dir)
=======
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
  return obj.get_top_k_matches(k, query_matrix)


