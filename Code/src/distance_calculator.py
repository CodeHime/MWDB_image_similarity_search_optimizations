# Top K image search using original images
# Here I am using a library named faiss for the similarity search
# Eucleadean and cosine:: TODO :: why this and not other distances
import faiss                   # make faiss available
import random
from scipy.stats import wasserstein_distance

def euclidean(xb, k, xq):
  """
  Calculate the euclidean distance and return the top k values
  """
  dimension = xb.shape[1]    # dimensions of each vector                         

  index = faiss.IndexFlatL2(dimension)   # build the index, d=size of vectors 
  # here we assume xb contains a n-by-d numpy matrix of type float32
  index.add(xb)                  # add vectors to the index
  # print(f"Total comparisions: {index.ntotal}")
  return index.search(xq, k)

def cosine(xb, k, xq):
  """
  Calculate the cosine distance and return the top k values
  """
  em = np.array([distance.cosine(i,xq) for i in xb])
  idx = np.argpartition(em, k)[:k]
  return em[idx], idx

def manhattan(xb, k, xq):
  """
  Calculate the manhattan distance and return the top k values
  """
  man = np.sum(np.absolute(xb-xq), axis=1)
  idx = np.argpartition(man, k)[:k]
  return man[idx], idx

def earth_movers(xb, k, xq):
  """
  Calculate the earth movers distance and return the top k values
  """
  # em = np.array([wasserstein_distance(i,xq[0]) for i in xb])
  em = np.array([wasserstein_distance(np.histogram(i)[1],np.histogram(xq)[1]) for i in xb])
  idx = np.argpartition(em, k)[:k]
  return em[idx], idx

def top_k_match(xb, k, xq, method="euclidean"):
  if method=="euclidean":
    return euclidean(xb, k, xq)
  elif method=="cosine":
    return cosine(xb, k, xq)
  elif method=="manhattan":
    return manhattan(xb, k, xq)
  elif method=="earth_movers":
    return earth_movers(xb, k, xq)

def get_image_file(features_dir, image_ids):
  df = pd.read_csv(os.path.join(features_dir, "image_ids.csv"))#.iloc[image_ids]
  return df["image_idx"].to_list()