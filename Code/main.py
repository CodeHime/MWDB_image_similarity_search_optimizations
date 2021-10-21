"""
main.py: main script to run
C:/Users/Krima/Documents/MWDB_proj/Code/data/phase2_sample_images
C:/Users/Krima/Documents/MWDB_proj/Code/data/input_images/0.png
Notes:
- Give code permission to handle files

"""
# Import the dataset
import sys
sys.path.insert(1, './src/')

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image
from src.config import *
from src.distance_calculator import *
from src.features_extractor import *
from src.latent_features_extractor import *

# remove mask for current user
os.umask(0)

input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, \
sub_features_dir, X, Y = initialize_variables()

images, img_all_names = read_all_images(input_dir, pattern={"X": X, "Y": Y})
k = min(images.shape[0], 10) if input_k == "" else min(images.shape[0], int(input_k))
save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization= False,
                      img_ids=img_all_names)
feature_dict = get_feature_dict_file(features_dir)
in_feature_dict = get_feature_dict_image(image_path, output_dim, sub_features_dir, hog_dict)

# Query image logic
d, i = top_k_match(feature_dict[feature], k, in_feature_dict[feature], method=distance_dict[feature])   # actual search

destination_dir = os.path.join(base_dir,
                               input_dir.rstrip("/").rsplit("/", 1)[1] + "_" + feature +\
                               "_" + input_dir.rstrip(".png").rstrip("/").rsplit("/", 1)[1] + "_" + feature +"_results_task3/"
                               if input_dir!="" else input_dir + "_" + feature +"_results_task3/")
if not os.path.isdir(destination_dir):
  os.makedirs(destination_dir)

image_files = get_image_file(features_dir, i)
for i in range(len(i)):
  result = Image.fromarray((images[i]).astype(np.uint8))
  result.save(os.path.join(destination_dir, image_files[i].split("/")[-1].split("\\")[-1]))
  print(os.path.join(destination_dir, image_files[i].split("/")[-1].split("\\")[-1]), d[i])
  plt.imshow(images[i], cmap='gray')
  plt.show()