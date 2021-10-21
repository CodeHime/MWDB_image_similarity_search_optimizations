# Import the dataset
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from src.config import *
from src.distance_calculator import *
from src.features_extractor import *
from src.latent_features_extractor import *

# Defining parameters for feature extraction
base_dir= "./data/"
features_dir=base_dir + "image_features/"
image_dir=base_dir + "input_images/"

# read all saved images
sub_features_dir = ["mean","std","skew","elbp","hog"]

# FROM STEP 2
input_dir = input("Enter path of images directory:")
input_k = input("Enter value of k:")
input_img = input("Enter input image path:")
selected_feature = input("Enter feature to compare:")

base_dir = base0_dir if input_dir=="" else input_dir.rstrip("/").rsplit("/",1)[0]
input_dir = image_dir if input_dir=="" else input_dir
feature = "all" if selected_feature=="" else selected_feature

features_dir=os.path.join(base_dir, input_dir.rstrip("/").rsplit("/",1)[1] + "_image_features/") if input_dir=="" else os.path.join(base_dir, "image_features/")
sub_features_dir = ["mean","std","skew","elbp","hog"]
# Create folders for images if they do not exist
if not os.path.isdir(features_dir):
  os.makedirs(features_dir)
for dir in sub_features_dir:
  if not os.path.isdir(os.path.join(features_dir, dir)):
    os.makedirs(os.path.join(features_dir, dir))

images = []
img_all_names = []
for f in glob.iglob(input_dir + "/*"):
    print(f)
    img_all_names.append(f)
    images.append(np.asarray(Image.open(f)))

k = 10 if input_k=="" else min(len(images), int(input_k))
images = np.array(images)
save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization = False, img_ids=img_all_names)
feature_dict = get_feature_dict_file(features_dir)
in_feature_dict = get_feature_dict_image(input_img, output_dim, sub_features_dir, hog_dict)

def get_accuracy_distances(xb,k,xq,distances):
  accuracy_dict= {d: [] for d in distances}
  for distance in distances:
    for i in range(len(xb)):
      # real_id = random.choice(range(n))
      real_id = i
      xq = np.array([xb[real_id]])

      D, I = top_k_match(xb, k, xq, method=distance)   # actual search
      accuracy_dict[distance].append(np.count_nonzero((I//10)==(real_id//10))/k)

    accuracy_dict[distance] = np.mean(accuracy_dict[distance])
  return accuracy_dict  

for feature in list(feature_dict.keys()): 
  print(feature)
  xb = feature_dict[feature]
  n = xb.shape[0]    # number of vectors    

  real_id = random.choice(range(n))
  # real_id = i
  xq = np.array([xb[real_id]])

  distances = ["euclidean", "cosine", "manhattan","earth_movers"]
  print(get_accuracy_distances(xb,k,xq,distances))