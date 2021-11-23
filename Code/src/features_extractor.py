import numpy as np
from skimage.feature import hog
import pandas as pd
import os
from skimage.feature import local_binary_pattern
from PIL import Image
from scipy.stats import skew
from config import *

# Mean of an image defines the average colour among that local area
def get_mean_img(input_img, output_dim):
  out=np.empty(output_dim)
  for i in range(output_dim[0]):
    for j in range(output_dim[1]):
      out[i][j]=(input_img[i*(input_img.shape[0]//output_dim[0]):(i+1)*(input_img.shape[0]//output_dim[0]), j*(input_img.shape[1]//output_dim[1]):(j+1)*(input_img.shape[1]//output_dim[1])].mean())
  return out.reshape(output_dim)


# Standard deviation of an image defines the variation of the light intensity amongst that local region
def get_std_dev_img(input_img, output_dim):
  out=np.empty(output_dim)
  for i in range(output_dim[0]):
    for j in range(output_dim[1]):
      out[i][j]=(input_img[i*(input_img.shape[0]//output_dim[0]):(i+1)*(input_img.shape[0]//output_dim[0]), j*(input_img.shape[1]//output_dim[1]):(j+1)*(input_img.shape[1]//output_dim[1])].std())
  return out.reshape(output_dim)


# Calculating skewness based on Pearson Mode Skewness
#  If the skewness is negative, the histogram is negatively skewed. 
# That means its left tail is longer or fatter than its right one. 
# Therefore, the frequency over the darker intensities (closer to zero) is wider spread (less concentrated, but not necessarily less frequent than the right tail!). The positive skewness is the opposite.
# https://stats.stackexchange.com/questions/211377/skewness-and-kurtosis-in-an-image
# Cite Pearson formula and numpy libraries
def get_skew_img(input_img, output_dim, mean_img=np.array([]), std_img=np.array([])):
  # DESIGN_DECISIONS: Pearson failed for std dev 0
  # if mean_img.shape==np.array([]).shape:
  #   mean_img=get_mean_img(input_img, output_dim)
  # if std_img.shape==np.array([]).shape:
  #   std_img=get_std_dev_img(input_img, output_dim)
  #
  # median_img=np.empty(output_dim)
  # for i in range(output_dim[0]):
  #   for j in range(output_dim[1]):
  #     median_img[i][j]=np.median(input_img[i*(input_img.shape[0]//output_dim[0]):(i+1)*(input_img.shape[0]//output_dim[0]), j*(input_img.shape[1]//output_dim[1]):(j+1)*(input_img.shape[1]//output_dim[1])])
  wskew = np.zeros(output_dim)
  for j in range(0, 63, output_dim[0]):
    for k in range(0, 63, output_dim[1]):
      p = int(j / output_dim[0])
      q = int(k / output_dim[1])
      wskew[p, q] = skew(input_img[j:j + output_dim[0] - +1, k:k + output_dim[1] - 1].flatten())
  return wskew


# Calculate and return color moments of the input image
def get_color_moments(input_img, output_dim):
  color_moments=[]
  color_moments.append(get_mean_img(input_img, output_dim))
  color_moments.append(get_std_dev_img(input_img, output_dim))
  color_moments.append(get_skew_img(input_img, output_dim, mean_img=color_moments[0], std_img=color_moments[1]))
  color_moments=np.array(color_moments)
  return color_moments


# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
# ELBP can be used for border detection
def get_elbp(input_img, neighbours=8, radius=1):
  return local_binary_pattern(input_img, neighbours, radius, method="uniform")


# From each region, 9 numbers are extracted for angles:: TODO
def get_hog(input_img, hog_dict):
  fd, hog_image = hog(input_img, 
                      orientations=hog_dict["orientation_bins"], 
                      pixels_per_cell=(hog_dict["cell_size"], hog_dict["cell_size"]), 
                      cells_per_block=(hog_dict["block_size"], hog_dict["block_size"]), 
                      block_norm = hog_dict["norm_type"],
                      visualize=True, 
                      # feature_vector=False,
                      multichannel=False # image is grayscale, deprecated by channel_axis, can be replaced if version>0.19
                      # channel_axis =None # image is grayscale
                      )
  return np.array(fd), hog_image


<<<<<<< HEAD
def get_feature_vectors(image, name, output_dim, sub_features_dir, hog_dict, feature_visualization_save=False, elbp_dict=eval(config['Phase1']['elbp_dict'])):
=======
def get_feature_vectors(image, name, output_dim, sub_features_dir, hog_dict, feature_visualization_save=False, elbp_dict=eval(config['phase1']['elbp_dict'])):
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
  color_moments = get_color_moments(image, output_dim)
  if feature_visualization_save:
    out = Image.fromarray((color_moments[0]).astype(np.uint8))
    out.save(os.path.join(features_dir, sub_features_dir[0], f"{name}.png")) 
    out = Image.fromarray((color_moments[1]).astype(np.uint8))
    out.save(os.path.join(features_dir, sub_features_dir[1], f"{name}.png")) 
    out = Image.fromarray((color_moments[2]).astype(np.uint8))
    out.save(os.path.join(features_dir, sub_features_dir[2], f"{name}.png"))

  elbp_out = get_elbp(image, neighbours=elbp_dict["neighbours"], radius=elbp_dict["radius"])
  if feature_visualization_save:
    out = Image.fromarray((elbp_out).astype(np.uint8))
    out.save(os.path.join(features_dir, sub_features_dir[3], f"{name}.png")) 

  hog_out, img = get_hog(image, hog_dict)
  if feature_visualization_save:
    out = Image.fromarray((img).astype(np.uint8))
    out.save(os.path.join(features_dir, sub_features_dir[4], f"{name}.png")) 
  
  return color_moments, elbp_out, hog_out


# Save all features as images(for visualization) and features (color moments, hog and elbp)
<<<<<<< HEAD
def save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization = False, img_ids=None, elbp_dict=eval(config['Phase1']['elbp_dict'])):
=======
def save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization = False, img_ids=None, elbp_dict=eval(config['phase1']['elbp_dict'])):
>>>>>>> a50422bc97ca29118c0ebb681476e6698b64818a
  if img_ids!=None:
    pd.DataFrame(enumerate(img_ids),columns=["image_id", "image_idx"]).to_csv(os.path.join(features_dir, "image_ids.csv"),index=False)
  else:
    pd.DataFrame(enumerate(range(images.shape[0])),columns=["image_id", "image_idx"]).to_csv(os.path.join(features_dir, "image_ids.csv"),index=False)

  img_lst = []
  color_moments_lst = [] 
  elbp_lst = [] 
  hog_lst = [] 
  
  for i in range(images.shape[0]):  
    img_lst.append(images[i].flatten())
    color_moments, elbp_out, hog_out = get_feature_vectors(images[i], i, output_dim, sub_features_dir, hog_dict, feature_visualization_save=feature_visualization, elbp_dict=elbp_dict)
    color_moments_lst.append(color_moments.flatten())
    elbp_lst.append(elbp_out.flatten())
    hog_lst.append(hog_out)
  pd.DataFrame(img_lst).to_csv(os.path.join(features_dir, "images.csv"), index=False)

  color_moments_lst = (color_moments_lst-np.min(color_moments_lst))/ (np.max(color_moments_lst)-np.min(color_moments_lst))
  pd.DataFrame(color_moments_lst).to_csv(os.path.join(features_dir, "cm8x8.csv"), index=False)

  elbp_lst = (elbp_lst-np.min(elbp_lst))/ (np.max(elbp_lst)-np.min(elbp_lst))
  pd.DataFrame(elbp_lst).to_csv(os.path.join(features_dir, "elbp.csv"), index=False)

  hog_lst = (hog_lst-np.min(hog_lst))/ (np.max(hog_lst)-np.min(hog_lst))
  pd.DataFrame(hog_lst).to_csv(os.path.join(features_dir, "hog.csv"), index=False)


def get_feature_dict_image(input_img, output_dim, sub_features_dir, hog_dict):
  in_image = np.asarray(Image.open(input_img))
  in_image_cm, in_image_elbp,in_image_hog= get_feature_vectors(in_image, input_img, output_dim, sub_features_dir, hog_dict)
  in_image_cm=in_image_cm.flatten()
  in_image_cm = (in_image_cm-np.min(in_image_cm))/ (np.max(in_image_cm)-np.min(in_image_cm))
  in_image_elbp=in_image_elbp.flatten()
  in_image_elbp = (in_image_elbp-np.min(in_image_elbp))/ (np.max(in_image_elbp)-np.min(in_image_elbp))
  in_image_hog=in_image_hog.flatten()
  in_image_hog = (in_image_hog-np.min(in_image_hog))/ (np.max(in_image_hog)-np.min(in_image_hog))

  in_feature_dict = {
    "img": in_image.astype(np.float32),
    "mean": in_image_cm[:64].astype(np.float32),
    "std": in_image_cm[64:128].astype(np.float32),
    "skew": in_image_cm[128:].astype(np.float32),
    "elbp": in_image_elbp,
    "hog": in_image_hog,
    "cm8x8":in_image_cm
  }
  in_feature_dict["all"] = np.concatenate((in_feature_dict["cm8x8"], in_feature_dict["elbp"], in_feature_dict["hog"])).astype(np.float32)
  return in_feature_dict


def get_feature_dict_file(features_dir):
  feature_dict = {
      "img": np.array(pd.read_csv(os.path.join(features_dir, "images.csv")), order='C').astype(np.float32),
      "mean": np.array(pd.read_csv(os.path.join(features_dir, "cm8x8.csv")).iloc[: , :64], order='C').astype(np.float32),
      "std": np.array(pd.read_csv(os.path.join(features_dir, "cm8x8.csv")).iloc[: , 64:128], order='C').astype(np.float32),
      "skew": np.array(pd.read_csv(os.path.join(features_dir, "cm8x8.csv")).iloc[: , 128:], order='C').astype(np.float32),    
      "elbp": np.array(pd.read_csv(os.path.join(features_dir, "elbp.csv")), order='C').astype(np.float32),
      "hog": np.array(pd.read_csv(os.path.join(features_dir, "hog.csv")), order='C').astype(np.float32)
  }
  feature_dict["cm8x8"]= np.concatenate((feature_dict["mean"], feature_dict["std"], feature_dict["skew"]), axis=1)
  feature_dict["all"]= np.concatenate((feature_dict["cm8x8"], feature_dict["elbp"], feature_dict["hog"]), axis=1).astype(np.float32)

  return feature_dict
