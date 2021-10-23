# Top K image search using original images
# Here I am using a library named faiss for the similarity search
# Eucleadean and cosine:: TODO :: why this and not other distances
import random
import os
import pandas as pd
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial import distance


def euclidean_fn(xb, xq):
  """
  Calculate the euclidean distance and return distance matrix
  :param xb: Data matrix to find similarity in
  :param xq:  Query matrix to find similarity for
  """
  eu=np.sqrt(np.sum(np.square(xb-xq), axis=1))
  return eu


def cosine_fn(xb, xq):
  """
  Calculate the cosine distance and return distance matrix
  :param xb: Data matrix to find similarity in
  :param xq:  Query matrix to find similarity for
  """
  cos = np.array([distance.cosine(i, xq) for i in xb])
  return cos


def manhattan_fn(xb, xq):
  """
  Calculate the manhattan distance and return distance matrix
  :param xb: Data matrix to find similarity in
  :param xq:  Query matrix to find similarity for
  """
  man = np.sum(np.absolute(xb-xq), axis=1)
  return man


def kl_divergence_fn(xb, xq):
  """
  Calculate the kl divergence and return divergence matrix
  :param xb: Data matrix to find similarity in
  :param xq:  Query matrix to find similarity for
  """
  return np.sum(np.where(xb != 0, xb * np.log(xb / xq), 0), axis=1)


def euclidean(xb, k, xq):
  """
  Calculate the euclidean distance and return the top k values
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """                       
  eu=euclidean_fn(xb, xq)
  idx = np.argpartition(eu, k)[:k]
  return eu[idx], idx

def kl_divergence(xb, k, xq):
  """
  Calculate the kl divergence and return divergence matrix
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """
  kl = kl_divergence_fn(xb, xq)
  idx = np.argpartition(kl, k)[:k]
  return kl[idx], idx


def cosine(xb, k, xq):
  """
  Calculate the cosine distance and return the top k values
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """
  cos = cosine_fn(xb, xq)
  idx = np.argpartition(em, k)[:k]
  return cos[idx], idx


def manhattan(xb, k, xq):
  """
  Calculate the manhattan distance and return the top k values
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """
  man = manhattan_fn(xb, xq)
  idx = np.argpartition(man, k)[:k]
  return man[idx], idx


def earth_movers(xb, k, xq):
  """
  Calculate the earth movers distance and return the top k values
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """
  # em = np.array([wasserstein_distance(i,xq[0]) for i in xb])
  em = np.array([wasserstein_distance(np.histogram(i)[1], np.histogram(xq)[1]) for i in xb])
  idx = np.argpartition(em, k)[:k]
  return em[idx], idx


def top_k_match(xb, k, xq, method="euclidean"):
  """
  General function to call distance functions
  :param xb: Data matrix to find similarity in
  :param k: Number of top objects to return
  :param xq:  Query matrix to find similarity for
  """
  if method == "euclidean":
    return euclidean(xb, k, xq)
  elif method == "cosine":
    return cosine(xb, k, xq)
  elif method == "manhattan":
    return manhattan(xb, k, xq)
  elif method == "earth_movers":
    return earth_movers(xb, k, xq)


def get_image_file(features_dir, image_ids):
  """
  Get image-image id mapping file
  :param features_dir: features directory where mapping is stored
  :param image_ids: image ids to fetch
  """
  df = pd.read_csv(os.path.join(features_dir, "image_ids.csv"))#.iloc[image_ids]
  return df["image_idx"].to_list()