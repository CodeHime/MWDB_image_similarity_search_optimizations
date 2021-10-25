from pca import *
from svd import *
from lda import *
from kmeans import *
from config import *


def perform_dimensionality_reductions(matrix, k, technique, base_dir):
  if technique == "pca":
    obj = Pca(k, matrix)
  elif technique == "svd":
    obj = Svd(k, matrix)
  elif technique == "lda":
    obj = Lda(k, matrix)
  elif technique == "kmeans":
    obj = Kmeans(k, matrix)
  else:
      raise ValueError("No such technique exists.")

  if not os.path.isdir(os.path.join(base_dir, config['Phase2'][technique + '_dir'])):
    os.makedirs(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  obj.save(os.path.join(base_dir, config['Phase2'][technique + '_dir']))
  return obj.get_obj_weight_pairs()

