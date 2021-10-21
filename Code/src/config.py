"""
config.py: File containing all configurations
"""
import os
import re
import glob
import numpy as np
from PIL import Image
from configparser import ConfigParser, ExtendedInterpolation

# initialize parameters from conf file
config = ConfigParser(interpolation=ExtendedInterpolation())
config.read(os.fspath('./src/config.conf'))
# Defining parameters for feature extraction
base0_dir = os.fspath(config['DEFAULT']['base0_dir'])
sub_features_dir = eval(config['Phase1']['sub_features_dir'])

# defining phase 1 definitions
output_dim = eval(config['Phase1']['output_dim'])
hog_dict = eval(config['Phase1']['hog_dict'])
distances = config['Phase1']['distances']
distance_dict = eval(config['Phase1']['distance_dict'])

def initialize_variables():
    """
    Initialize and return the variables
    """
    input_dir = os.fspath(input("Enter path of images directory:"))
    input_k = input("Enter value of k:")
    input_img = os.fspath(input("Enter input image path:"))
    selected_feature = input("Enter feature to compare:")
    X = input("Input type label:")
    Y = input("Input subject ID:")

    base_dir = base0_dir if input_dir == "" else os.path.normpath(os.path.join(input_dir, os.pardir))
# input_dir.rstrip("/").rsplit("/", 1)[0]
    image_path = os.fspath(os.path.join(base_dir, "input_images/") + os.listdir(os.path.join(base_dir, "input_images/"))[0]) \
        if input_img == "" else input_img
    input_dir = os.fspath(os.path.join(base_dir, "input_images/")) if input_dir == "" else input_dir
    feature = "all" if selected_feature == "" else selected_feature

    features_dir = os.fspath(os.path.join(base_dir, os.path.normpath(os.path.join(input_dir, os.pardir)) + "_image_features/")) \
        if input_dir == "" else os.fspath(os.path.join(base_dir, "image_features/"))

    # Create folders for images if they do not exist
    if not os.path.isdir(features_dir):
      os.makedirs(features_dir)
    for dir in sub_features_dir:
      if not os.path.isdir(os.path.join(features_dir, dir)):
        print(os.path.join(features_dir, dir))
        os.makedirs(os.path.join(features_dir, dir))

    return input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, \
           sub_features_dir, X, Y


def read_all_images(input_dir, pattern={"X": ".*", "Y": ".*"}):
    """
    Read  all images from given file given pattern
    """
    # set pattern
    if pattern["X"] == "" and pattern["Y"] == "":
        pattern = ".*.png"
    elif pattern["X"] == "":
        pattern = f".*-{pattern['Y']}-.*.png"
    elif pattern["Y"] == "":
        pattern = f".*-{pattern['X']}-.*.png"
    else:
        pattern["Y"] = f".*-{pattern['X']}-{pattern['Y']}-.*.png"

    # get images that match the pattern
    images = []
    img_all_names = []
    for f in glob.iglob(os.path.join(input_dir, "*")):
        if re.match(pattern, f):
            print("reading..." + f)
            img_all_names.append(f)
            images.append(np.asarray(Image.open(f)))
    return np.array(images), img_all_names
