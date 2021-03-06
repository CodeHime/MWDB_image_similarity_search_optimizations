import os
import sys
import numpy as np
import random
from distance_calculator import *
# http://mlwiki.org/index.php/Euclidean_LSH
import json


class LSH_node:
  def __init__(self, w, vectors, b, plane_norms=np.array([])):
    if plane_norms.size == 0:
      # self.plane_norms = np.random.normal(np.mean(vectors, axis=0),
      #                                     np.std(vectors, axis=0), size=(vectors.shape[1], vectors.shape[1]))
      self.plane_norms = np.random.rand(vectors.shape[1], vectors.shape[1]) - .5
    else:
      self.plane_norms = plane_norms

    self.w = w
    self.b = b
    self.vectors = vectors

    self.buckets = {}

    for i in range(len(vectors)):
      # convert from array to string
      hash_str = self.transform(vectors[i], shift=True)
      # create bucket if it doesn't exist
      if hash_str not in self.buckets.keys():
        self.buckets[hash_str] = []
      # add vector position to bucket
      self.buckets[hash_str] = self.buckets[hash_str] + [i]

  def transform(self, xq, shift=True):
    # DESIGN_DECISION: map vector to the random projection plane
    # Formula used: ((np.dot(vectors - 0.5, lsh.plane_norms)*lsh.lsh_family[0].w*vectors.shape[0] + lsh.lsh_family[0].b) / lsh.lsh_family[0].w > 0).astype(int)
    # Explanation:
    # Shift vectors left by 0.5 and calculate dot products of all vectors
    if shift:
      v_dot = np.dot(xq - 0.5, self.plane_norms)
    else:
      v_dot = np.dot(xq, self.plane_norms)
    # if (v_dot > 0).astype(int).sum() > 0 and (v_dot < 0).astype(int).sum() > 0:
    #   print((v_dot > 0).astype(int).sum(), (v_dot < 0).astype(int).sum()

    # a random b would eliminate errors/borderline cases)
    v_dot = v_dot * self.w * self.vectors.shape[0] + self.b
    v_dot = v_dot / self.w * self.vectors.shape[0]
    # Convert dot product to binary
    v_dot = v_dot > 0
    # Convert boolean to int for bucketing
    v_dot = v_dot.astype(int)
    # print(v_dot)
    return ''.join(v_dot.astype(str))

  def get_bucket(self, xq, k=1):
    d, i = hamming(self.buckets.keys(), k, self.transform(xq))
    return d, np.array(list(self.buckets.keys()))[np.array(i)]

  def save(self, lsh_family_dir, bucket_index):
    with open(os.path.join(lsh_family_dir, bucket_index + '.json'), 'w', encoding='utf-8') as f:
      json.dump(self.buckets, f, ensure_ascii=False, indent=4)

  def get_size(self):
    return sys.getsizeof(self.buckets)


class LSH_family:
  def __init__(self, K, num_obj, vectors):
    self.w = num_obj
    self.b = random.uniform(0, self.w)
    self.K = K
    # self.plane_norms = np.random.rand(*vectors.shape)
    # Calculate random projection plane using gaussian distribution
    # self.plane_norms = np.random.normal(np.mean(vectors, axis=0),
    #                                     np.std(vectors, axis=0), size=(vectors.shape[1], vectors.shape[1]))
    self.plane_norms = np.random.rand(vectors.shape[1], vectors.shape[1]) - .5

    self.lsh_family = []
    for j in range(self.K):
      # DESIGN_DECISION: set range of b here!!!
      # self.b = random.uniform(0, self.w)
      cur_node = LSH_node(self.w, vectors, self.b, plane_norms=self.plane_norms)
      self.lsh_family.append(cur_node)

  def get_bucket(self, xq, k=1):
    set_list = {}
    for hash in self.lsh_family:
      d, indices = hash.get_bucket(xq, k)
      indices = indices.flatten()
      # res = {indices[i]: d[i] for i in range(len(indices))}
      for i in range(len(indices)):
        set_list[indices[i]] = set_list.get(indices[i], []) + [d[i]]
    return set_list

  def get_all_candidates(self, xq, k=1):
    set_list = self.get_bucket(xq, k=k)
    candidate_list = set()
    q_buckets_dict = {}

    total_buckets_searched = 0
    total_non_unique_candidate_images = 0

    for bucket, v in set_list.items():
      new_dict = {}
      for i in list(set(v)):
        v_freq = v.count(i)
        new_dict.update({i: v_freq})
        # APPLYING CONJUNCTION TO ALL LSH FAMILY NODES
        if v_freq == self.K:
          total_buckets_searched+=1
          for hash in self.lsh_family:
            total_non_unique_candidate_images+=len(hash.buckets[bucket])
            candidate_list.update(hash.buckets[bucket])
      q_buckets_dict.update({bucket: new_dict})
    return q_buckets_dict, candidate_list, total_buckets_searched, total_non_unique_candidate_images

  def get_size(self):
    family_size = 0
    for hash in self.lsh_family:
      family_size += hash.get_size()
    return family_size

  def save(self, lsh_fam_dir):
    for b_i in range(len(self.lsh_family)):
      self.lsh_family[b_i].save(lsh_fam_dir, str(b_i))


class LSH:
  def __init__(self, L, K, vectors, num_obj=5):
    self.L = L
    self.K = K
    self.vectors = vectors
    self.num_obj = num_obj

    self.lsh_families = []
    for i in range(self.L):
      self.lsh_families.append(LSH_family(self.K, self.num_obj, self.vectors))

  def get_bucket(self, xq, k=1):
    set_list = {}
    for fam_i in range(len(self.lsh_families)):
      set_list[fam_i] = self.lsh_families[fam_i].get_bucket(xq, k)
    return set_list

  def get_all_candidates(self, xq, k=1):
    set_list = {}
    candidate_list = set()
    total_buckets_searched = 0
    total_non_unique_candidate_images = 0
    for fam_i in range(len(self.lsh_families)):
      set_list[fam_i], fam_candidate_list, fam_buckets_searched, fam_non_unique_candidate_images = self.lsh_families[fam_i].get_all_candidates(xq, k=k)
      total_buckets_searched += fam_buckets_searched
      total_non_unique_candidate_images += fam_non_unique_candidate_images
      candidate_list.update(fam_candidate_list)
    return set_list, list(candidate_list), fam_buckets_searched, fam_non_unique_candidate_images

  def get_size(self):
    total_lsh_size = 0
    for fam in self.lsh_families:
      total_lsh_size+=fam.get_size()
    return total_lsh_size

  def save(self, lsh_dir):
    if not os.path.isdir(lsh_dir):
      os.makedirs(lsh_dir)
    for fam_i in range(len(self.lsh_families)):
      if not os.path.isdir(os.path.join(lsh_dir, str(fam_i))):
        os.makedirs(os.path.join(lsh_dir, str(fam_i)))
      self.lsh_families[fam_i].save(os.path.join(lsh_dir, str(fam_i)))
