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

# read all saved images
sub_features_dir = ["mean","std","skew","elbp","hog"]

# FROM STEP 2
input_dir = input("Enter path of images directory:")
input_k = input("Enter value of k:")
input_img = input("Enter input image path:")

image_dir=base_dir + "input_images/"
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

import matplotlib.pyplot as plt

feature="all"
xb = feature_dict[feature]
n = xb.shape[0]    # number of vectors    

xq = in_feature_dict[feature]
D, I = top_k_match(xb, k, xq, method=distance_dict[feature])   # actual search

# destination_dir = os.path.join(base_dir, input_dir.rstrip("/").rsplit("/",1)[1] + "_" + input_img.rstrip(".png").rstrip("/").rsplit("/",1)[1]  + "_" + feature +"_results_task3/")
destination_dir = os.path.join(base_dir, 
                               input_dir.rstrip("/").rsplit("/",1)[1] +\
                               "_" + input_img.rstrip(".png").rstrip("/").rsplit("/",1)[1] if input_img!="" else input_img +\
                               "_" + feature +"_results_task3/")
if not os.path.isdir(destination_dir):
  os.makedirs(destination_dir)

image_files = get_image_file(features_dir, I)
for i in range(len(I)):
  result = Image.fromarray((images[I[i]]).astype(np.uint8))
  result.save(os.path.join(destination_dir, image_files[I[i]].split("/")[-1]))
  print(os.path.join(destination_dir, image_files[I[i]].split("/")[-1]), D[i])
  # conclusion_metric.append([os.path.join(destination_dir, image_files[I[i]].split("/")[-1]), D[i]])
  plt.imshow(images[I[i]], cmap='gray')
  plt.show()

all_features = ["cm8x8","elbp","hog"]
Dist=[]
Inx=[]
for i in range(len(all_features)):
  xb = feature_dict[all_features[i]]
  n = xb.shape[0]    # number of vectors    
  print(xb.shape)
  xq = in_feature_dict[all_features[i]]
  Di, Ii = top_k_match(xb, n-1, xq, method=distance_dict[all_features[i]])   # actual search
  Dist.append(Di)
  # destination_dir = os.path.join(base_dir, input_dir.rstrip("/").rsplit("/",1)[1] + "_" + input_img.rstrip(".png").rstrip("/").rsplit("/",1)[1]  + "_" + all_features[i] +"_results_task3/")
  destination_dir = os.path.join(base_dir, 
                                input_dir.rstrip("/").rsplit("/",1)[1] +\
                                "_" + input_img.rstrip(".png").rstrip("/").rsplit("/",1)[1] if input_img!="" else input_img +\
                                "_" + all_features[i] +"_results_task3/")

for i in range(len(all_features)):
  print("Contribution of " + all_features[i])
  print(Dist[i].sum()/np.array(Dist).sum())