from pca import *
from svd import *
from lda import *
from kmeans import *
from config import *


def get_subjects_from_ids(features_dir, indx):
  sub_id_dict = {}
  image_files = get_image_file(features_dir, indx)
  for i in range(len(image_files)):
    sub_id = re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png").split("-")[-2]
    sub_id_dict.update({sub_id: sub_id_dict.get(sub_id, [i])})
  return sub_id_dict


def get_images_from_ids(features_dir, indx):
  img_id_dict = {}
  image_files = get_image_file(features_dir, indx)
  for i in range(len(image_files)):
    img_id_dict.update({indx[i]: image_files[indx[i]]})
  return img_id_dict


def get_type_from_ids(features_dir, indx):
  type_id_dict = {}
  image_files = get_image_file(features_dir, indx)
  for i in range(len(image_files)):
    type_id = re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png").split("-")[-3]
    type_id_dict.update({type_id: type_id_dict.get(type_id, [i])})
  return type_id_dict


def get_sample_from_ids(features_dir, indx):
  type_id_dict = {}
  image_files = get_image_file(features_dir, indx)
  for i in range(len(image_files)):
    type_id = re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png").split("-")[-1]
    type_id_dict.update({type_id: type_id_dict.get(type_id, [i])})
  return type_id_dict


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

