"""
main.py: main script to run
Phase 1:
C:/Users/Krima/Documents/MWDB_proj/Code/data/phase2_sample_images
C:/Users/Krima/Documents/MWDB_proj/Code/data/input_images/0.png

Phase 2:
C:/Users/Krima/Documents/MWDB_image_similarity_search_optimizations/Code/data/phase2_data/all

Notes:
- Give code permission to handle files

k = 5, 10, 50
PCA, SVD, LDA, k-means
X=cc, neg, original, rot, noise
Y=1, 26, 37

"""
# Import the dataset
import sys
sys.path.insert(1, './src/')

# common
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
from PIL import Image
from src.config import *
# Phase 1
from src.distance_calculator import *
from src.features_extractor import *
# Phase 2
# Phase 2 - Tasks 1 and 2
from src.latent_features_extractor import *
# Phase 2 - Tasks 3 and 4
from src.similarity_matrizer import *
# Phase 2 - Tasks 5, 6 and 7
from src.dimension_reduced_querying import *
# Phase 2 - Tasks 8 and 9
from src.grapher import *
# Phase 2 - Task 8
from src.ascos import *
# Phase 2 - Task 9
from src.ppr import *

import re


def Phase1_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, sub_features_dir):
    images, img_all_names = read_all_images(input_dir, pattern={"X": X, "Y": Y})
    k = min(images.shape[0], 10) if input_k == "" else min(images.shape[0], int(input_k))
    save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(features_dir)
    in_feature_dict = get_feature_dict_image(image_path, output_dim, sub_features_dir, hog_dict)

    # Query image logic
    d, indx = top_k_match(feature_dict[feature], k, in_feature_dict[feature],
                       method=distance_dict[feature])  # actual search

    phase1_destination_dir = os.path.join(base_dir,
                                          os.path.basename(input_dir.rstrip("/")) + "_" + feature + \
                                          "_" + os.path.basename(input_dir.rstrip(".png")) +
                                          "_" + feature + "_results_task3/"
                                          if input_dir != "" else input_dir + "_" + feature + "_results_task3/")
    if not os.path.isdir(phase1_destination_dir):
        os.makedirs(phase1_destination_dir)

    image_files = get_image_file(features_dir, indx)
    for i in range(len(indx)):
        result = Image.fromarray((images[indx[i]]).astype(np.uint8))
        result.save(os.path.join(phase1_destination_dir, image_files[indx[i]].split("/")[-1].split("\\")[-1]))
        print("=" * 20 + "\n")
        print("PHASE 1 OUTPUT:")
        print(os.path.join(phase1_destination_dir, image_files[indx[i]].split("/")[-1].split("\\")[-1]), d[i])
        plt.imshow(images[i], cmap='gray')
        plt.show()


def Phase2_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir,
                sub_features_dir, X, Y, technique):
    images, img_all_names = read_all_images(input_dir, pattern={"X": X, "Y": Y})
    k = min(images.shape[0], 10) if input_k == "" else min(images.shape[0], int(input_k))
    save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(features_dir)
    in_feature_dict = get_feature_dict_image(image_path, output_dim, sub_features_dir, hog_dict)

    task_num = int(input("Enter task number(1-9):"))
    k_latent = input("Enter k for number of latent features:")
    k_latent = min(feature_dict[feature].shape[0], feature_dict[feature].shape[1])*3//4 \
        if k_latent == "" else int(k_latent)

    if task_num < 3:
        perform_dimensionality_reductions(feature_dict[feature], k_latent, technique, base_dir)
    elif task_num < 5:
        sim_type = input("Enter matrix similarity type(type/subject):")
        get_similarity_matrix(feature_dict[feature], k, base_dir, features_dir, technique=technique, sim_type=sim_type,
                              sim_method="euclidean")
    elif task_num < 8:
        perform_dimensionality_reductions(feature_dict[feature], k_latent, technique, base_dir)
        d, indx = get_top_k_matches_latent_space(in_feature_dict[feature], k, technique, base_dir)

        phase2_destination_dir = os.path.join(base_dir,
                                              os.path.basename(input_dir.rstrip("/")) + "_" + feature +\
                                              "_" + os.path.basename(input_dir.rstrip(".png")) + "_" + technique + "_results/"
                                              if input_dir != "" else input_dir + "_" + feature + "_results/")
        if not os.path.isdir(phase2_destination_dir):
            os.makedirs(phase2_destination_dir)

        image_files = get_image_file(features_dir, indx)

        print("=" * 80 + "\n")
        print("K OUTPUT:")
        for i in range(len(indx)):
            result = Image.fromarray((images[indx[i]]).astype(np.uint8))
            result.save(os.path.join(phase2_destination_dir, image_files[indx[i]].split("/")[-1].split("\\")[-1]))
            print(re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png"), d[i])

        if task_num == 5:
            for i in range(len(indx)):
                plt.imshow(images[i], cmap='gray')
                plt.show()
        elif task_num == 6:
            for i in range(len(indx)):
                result = Image.fromarray((images[indx[i]]).astype(np.uint8))
                result.save(os.path.join(phase2_destination_dir, image_files[indx[i]].split("/")[-1].split("\\")[-1]))
            type_tag = []
            for i in range(len(indx)):
                type_tag.append(re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png").split("-")[1])
            print(type_tag)
            print(most_frequent(type_tag))
        elif task_num == 7:
            for i in range(len(indx)):
                result = Image.fromarray((images[indx[i]]).astype(np.uint8))
                result.save(os.path.join(phase2_destination_dir, image_files[indx[i]].split("/")[-1].split("\\")[-1]))
            sub_tag = []
            for i in range(len(indx)):
                sub_tag.append(re.split("/|\\\\", image_files[indx[i]])[-1].rstrip(".png").split("-")[2])
            print(sub_tag)
            print(most_frequent(sub_tag))
    elif task_num == 8:
        n = input("Enter n(number of top values to find in database): ")
        m = input("Enter m(number of similar subjects to find): ")
        sub_sub_sim = get_similarity_matrix(feature_dict[feature], k, base_dir, features_dir, technique="",
                                            sim_type="subject")
        adjacency_matrix = create_adjacency_matrix(sub_sub_sim, int(n))
        m_sim = ascos_similarity(adjacency_matrix, int(n), int(m))

        sub_dict = dict(pd.read_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], "subjectid_map.csv"),
                                     header=None).values)

        print([sub_dict[i] for i in m_sim])
        # get_subjects_from_ids()
    elif task_num == 9:
        n = input("Enter n(number of top values to find in database): ")
        m = input("Enter m(number of similar subjects to find): ")
        seed_id = input("Enter subject ids(comma separated):").replace(" ", "").split(",")

        sub_sub_sim = get_similarity_matrix(feature_dict[feature], k, base_dir, features_dir, technique="", sim_type="subject")

        seed_dict = dict(pd.read_csv(os.path.join(base_dir, config['Phase2']['similarity_dir'], "subjectid_map.csv"), header=None).values)
        seed_id = [seed_dict[int(i)] for i in seed_id]
        # print(seed_id)
        adjacency_matrix = create_adjacency_matrix(sub_sub_sim, int(n))
        rev_seed_dict = {v: k for k, v in seed_dict.items()}

        seed_ranks = get_rank_with_seeds(adjacency_matrix, int(m), seed_id)
        print([[rev_seed_dict[i], r] for i, r in seed_ranks])
    else:
        raise ValueError("No such task exists. Enter a valid value from 1 to 9.")


# remove mask for current user
os.umask(0)

input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, \
sub_features_dir, X, Y, technique = initialize_variables()

print(f"INPUT IMAGE IS {image_path}")

# Phase1_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, sub_features_dir)

Phase2_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir,
            sub_features_dir, X, Y, technique)