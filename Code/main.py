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


IMPORTANT TERMS:
feature_dict : Dict with all the feature transforms of the image FOLDER
in_feature_dict : Dict with all the feature transforms of the INPUT image

obj = get_saved_latent_object(technique, base_dir) : the object of the lantent semantics
obj.get_latent_features: returns your transformed vector space
obj.transform : returns your input image transformed

get_images_from_ids(features_dir, indx)
get_subjects_from_ids(features_dir, indx)
get_type_from_ids(features_dir, indx)
get_sample_from_ids(features_dir, indx)

a50422bc97ca29118c0ebb681476e6698b64818a
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

# Phase 3 - Tasks 1 to 3
from svm_task1 import *
from svm_task2 import *
from svm_task3 import *
from feedback import *
from src.decision_tree import *
# Phase 3 - Task 4
from src.lsh import *
# Phase 3 - Task 5
from src.vafiles import *

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
                sub_features_dir, X, Y, technique, task_num=None):
    images, img_all_names = read_all_images(input_dir, pattern={"X": X, "Y": Y})
    k = min(images.shape[0], 10) if input_k == "" else min(images.shape[0], int(input_k))
    save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(features_dir)
    in_feature_dict = get_feature_dict_image(image_path, output_dim, sub_features_dir, hog_dict)

    if not task_num:
        task_num = int(input("Enter task number(1-9):"))
    k_latent = input("Enter k for number of latent features:")
    k_latent = min(feature_dict[feature].shape[0], feature_dict[feature].shape[1])*3//4 \
        if k_latent == "" else int(k_latent)

    if task_num < 3:
        perform_dimensionality_reductions(feature_dict[feature], k_latent, technique, base_dir)
    elif task_num < 5:
        pass
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

def get_test_data(technique, k_latent=None, norm_max_latent=None, norm_min_latent=None):
    # TESTING DATA SETUP
    test_set_path = input("Enter test path directory:")
    test_features_dir = os.fspath(test_set_path.rstrip("/").rstrip("\\")
                                  + "_" + config['Phase1']['image_feature_dir'])
    # Create folders for images if they do not exist
    if not os.path.isdir(test_features_dir):
        os.makedirs(test_features_dir)
    for dir in sub_features_dir:
        if not os.path.isdir(os.path.join(test_features_dir, dir)):
            os.makedirs(os.path.join(test_features_dir, dir))

    # GET TESTING DATA
    images, img_all_names = read_all_images(test_set_path, pattern={"X": "", "Y": ""})
    save_all_img_features(images, output_dim, test_features_dir, sub_features_dir, hog_dict,
                          feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(test_features_dir)

    if technique != "none":
        # DESIGN_DECISION: Take a minimum over 20 latent features or less
        k_latent = min(feature_dict[feature].shape[0] * 3 // 4, feature_dict[feature].shape[1] * 3 // 4, 20) \
            if k_latent == "" else int(k_latent)

        # if not os.path.isdir(os.path.join(base_dir, config['Phase2'][technique + '_dir'])):
        perform_dimensionality_reductions(feature_dict[feature], k_latent, technique, base_dir)
        obj = get_saved_latent_object(technique, base_dir)
        data = obj.get_vector_space()
        data = (data - norm_min_latent) / (norm_max_latent - norm_min_latent)
    else:
        data = feature_dict[feature]
    return test_features_dir, data


def Phase3_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir,
                sub_features_dir, X, Y, technique, norm_max_latent=None, norm_min_latent=None):
    images, img_all_names = read_all_images(input_dir, pattern={"X": X, "Y": Y})
    k = min(images.shape[0], 10) if input_k == "" else min(images.shape[0], int(input_k))
    save_all_img_features(images, output_dim, features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(features_dir)
    in_feature_dict = get_feature_dict_image(image_path, output_dim, sub_features_dir, hog_dict)
    if technique != "none":
        # From Phase 2
        k_latent = input("Enter k for number of latent features:")
        # DESIGN_DECISION: Take a minimum over 20 latent features or less
        k_latent = min(feature_dict[feature].shape[0] * 3 // 4, feature_dict[feature].shape[1] * 3 // 4, 20) \
            if k_latent == "" else int(k_latent)

        # if not os.path.isdir(os.path.join(base_dir, config['Phase2'][technique + '_dir'])):
        perform_dimensionality_reductions(feature_dict[feature], k_latent, technique, base_dir)
        obj = get_saved_latent_object(technique, base_dir)
        training_data = obj.get_vector_space()
        norm_max_latent = np.max(training_data)
        norm_min_latent = np.min(training_data)
        training_data = (training_data - norm_min_latent) / (norm_max_latent - norm_min_latent)

        input_data = obj.transform([in_feature_dict[feature]])
        input_data = (input_data - norm_min_latent) / (norm_max_latent - norm_min_latent)
    else:
        k_latent = None
        training_data = feature_dict[feature]
        input_data = in_feature_dict[feature]

    task_num = int(input("Enter task number(1-8):"))

    if task_num == 1:
        classifier = input("Enter input classifier (svm - SVM/ dt - decision tree/ ppr - personalized pagerank) :")
        if classifier == "svm":
            test_features_dir, test_data = get_test_data(technique, k_latent=k_latent, norm_max_latent=norm_max_latent, norm_min_latent=norm_min_latent)
            svm_task_1(features_dir, test_features_dir, training_set_features=training_data,
                       test_set_features=test_data)
        elif classifier == "dt":
            # TODO: REMOVE HARD CODE SELECTION
            # Define labels
            labels_dict = get_type_from_ids(features_dir, range(len(images)), reverse_dict=True)
            indx_df = pd.DataFrame.from_dict(labels_dict, orient='index', columns=["label"])

            # Define vectors
            # vectors_df = pd.DataFrame(obj.get_vector_space())
            vectors_df = pd.DataFrame(training_data)

            # Join vectors to their labels corresponding to indexes
            vectors_df = vectors_df.join(indx_df)
            # Create decision tree
            dt_obj = DecisionTree(vectors_df)
            # TRAINING DATA SUMMARY
            dt_obj.get_prediction_summary(vectors_df, labels=list(set(labels_dict.values())))

        elif classifier == "ppr":
            #TODO: optimize
            n = 5
            # TRAIN DATA
            sim_mat = get_similarity_matrix(training_data, k, base_dir, features_dir, technique="",
                                                sim_type="type")
            adjacency_matrix = create_adjacency_matrix(sim_mat, int(n))
            # print(sim_mat)
            # print(adjacency_matrix)
            test_data = [training_data[i,:adjacency_matrix.shape[1]] for i in range(training_data.shape[0])]
            act_vals = get_type_from_ids(features_dir, range(len(training_data)), reverse_dict=True)
            pred_vals = ppr_predict(adjacency_matrix, test_data)
            pred_vals = get_type_from_ids(features_dir, pred_vals, reverse_dict=True)

            act_val = []
            pred_val = []
            for key in sorted(act_vals.keys()):
                act_val.append(act_vals[key])
                pred_val.append(pred_vals[key])

            conf_mat = confusion_matrix(act_val, pred_val, labels=list(set(act_val)))
            print("*" * 10 + "  CONFUSION MATRIX " + "*" * 10)
            print(conf_mat)
            # Calculate false +ves
            sum_col = conf_mat.sum(axis=0) - conf_mat.diagonal()
            # Calculate misses
            sum_row = conf_mat.sum(axis=1) - conf_mat.diagonal()
            print(f"False +ves: {sum_col} \nMisses: {sum_row}")
            print("*" * 40)
        else:
            raise NotImplmentedError(f"No implementation found for selected task: {task_num} {classifier}")
    elif task_num == 2:
        classifier = input("Enter input classifier (svm - SVM/ dt - decision tree/ ppr - personalized pagerank) :")
        if classifier == "svm":
            test_features_dir, test_data = get_test_data(technique, k_latent=k_latent, norm_max_latent=norm_max_latent, norm_min_latent=norm_min_latent)
            svm_task_2(features_dir, test_features_dir, training_set_features=training_data,
                       test_set_features=test_data)
        elif classifier == "dt":
            # Define labels
            labels_dict = get_subjects_from_ids(features_dir, range(len(images)), reverse_dict=True)
            indx_df = pd.DataFrame.from_dict(labels_dict, orient='index', columns=["label"])

            # Define vectors
            # vectors_df = pd.DataFrame(obj.get_vector_space())
            vectors_df = pd.DataFrame(training_data)

            # Join vectors to their labels corresponding to indexes
            vectors_df = vectors_df.join(indx_df)
            # Create decision tree
            dt_obj = DecisionTree(vectors_df)
            # TRAINING DATA SUMMARY
            dt_obj.get_prediction_summary(vectors_df, labels=list(set(labels_dict.values())))
        elif classifier == "ppr":
            #TODO: optimize
            n = 5
            # TRAIN DATA
            sim_mat = get_similarity_matrix(training_data, k, base_dir, features_dir, technique="",
                                                sim_type="subject")
            adjacency_matrix = create_adjacency_matrix(sim_mat, int(n))
            # print(sim_mat)
            # print(adjacency_matrix)
            test_data = [training_data[i,:adjacency_matrix.shape[1]] for i in range(training_data.shape[0])]
            act_vals = get_subjects_from_ids(features_dir, range(len(training_data)), reverse_dict=True)
            pred_vals = ppr_predict(adjacency_matrix, test_data)
            pred_vals = get_subjects_from_ids(features_dir, pred_vals, reverse_dict=True)

            act_val = []
            pred_val = []
            for key in sorted(act_vals.keys()):
                act_val.append(act_vals[key])
                pred_val.append(pred_vals[key])

            conf_mat = confusion_matrix(act_val, pred_val, labels=list(set(act_val)))
            print("*" * 10 + "  CONFUSION MATRIX " + "*" * 10)
            print(conf_mat)
            # Calculate false +ves
            sum_col = conf_mat.sum(axis=0) - conf_mat.diagonal()
            # Calculate misses
            sum_row = conf_mat.sum(axis=1) - conf_mat.diagonal()
            print(f"False +ves: {sum_col} \nMisses: {sum_row}")
            print("*" * 40)
        else:
            raise NotImplmentedError(f"No implementation found for selected task: {task_num} {classifier}")
    elif task_num == 3:
        classifier = input("Enter input classifier (svm - SVM/ dt - decision tree/ ppr - personalized pagerank) :")
        if classifier == "svm":
            test_features_dir, test_data = get_test_data(technique, k_latent=k_latent, norm_max_latent=norm_max_latent, norm_min_latent=norm_min_latent)
            svm_task_3(features_dir, test_features_dir, training_set_features=training_data,
                       test_set_features=test_data)
        elif classifier == "dt":
            # Define labels
            labels_dict = get_sample_from_ids(features_dir, range(len(images)), reverse_dict=True)
            indx_df = pd.DataFrame.from_dict(labels_dict, orient='index', columns=["label"])

            # Define vectors
            # TODO:
            # vectors_df = pd.DataFrame(obj.get_vector_space())
            vectors_df = pd.DataFrame(training_data)

            # Join vectors to their labels corresponding to indexes
            vectors_df = vectors_df.join(indx_df)
            # Create decision tree
            dt_obj = DecisionTree(vectors_df)
            # TRAINING DATA SUMMARY
            dt_obj.get_prediction_summary(vectors_df, labels=list(set(labels_dict.values())))
        elif classifier == "ppr":
            #TODO: optimize
            # TRAIN DATA
            n = 5
            sim_mat = get_similarity_matrix(training_data, k, base_dir, features_dir, technique="",
                                                sim_type="sample")
            adjacency_matrix = create_adjacency_matrix(sim_mat, int(n))
            # print(sim_mat)
            # print(adjacency_matrix)
            test_data = [training_data[i,:adjacency_matrix.shape[1]] for i in range(training_data.shape[0])]
            act_vals = get_sample_from_ids(features_dir, range(len(training_data)), reverse_dict=True)
            pred_vals = ppr_predict(adjacency_matrix, test_data)
            pred_vals = get_sample_from_ids(features_dir, pred_vals, reverse_dict=True)

            act_val = []
            pred_val = []
            for key in sorted(act_vals.keys()):
                act_val.append(act_vals[key])
                pred_val.append(pred_vals[key])

            conf_mat = confusion_matrix(act_val, pred_val, labels=list(set(act_val)))
            print("*" * 10 + "  CONFUSION MATRIX " + "*" * 10)
            print(conf_mat)
            # Calculate false +ves
            sum_col = conf_mat.sum(axis=0) - conf_mat.diagonal()
            # Calculate misses
            sum_row = conf_mat.sum(axis=1) - conf_mat.diagonal()
            print(f"False +ves: {sum_col} \nMisses: {sum_row}")
            print("*" * 40)
        else:
            raise NotImplmentedError(f"No implementation found for selected task: {task_num} {classifier}")
    elif task_num == 4:
        num_layers = input("Enter number of layers:")
        num_func_per_layer = input("Enter number of functions per layer:")
        k_radius = input("Enter radius of distance to find:")

        subjects = set()
        in_sub = get_sub_from_image_path(image_path)
        tot_in_sub = []
        for img in img_all_names:
            sub_id = get_sub_from_image_path(img)
            subjects.add(sub_id)
            if sub_id==in_sub:
                tot_in_sub.append(img)
        # print(np.min(feature_dict["cm8x8"]),np.max(feature_dict["cm8x8"]))
        lsh = LSH(int(num_layers), int(num_func_per_layer), training_data, num_obj=20)
        # lsh = LSH(int(num_layers), int(num_func_per_layer), feature_dict["cm8x8"], num_obj=len(list(subjects)))
        set_list, indx, total_buckets_searched, \
        total_non_unique_candidate_images = lsh.get_all_candidates(input_data, k=int(k_radius))
        lsh.save(os.path.join(base_dir, "classifiers/lsh"))

        all_candidate_subs = get_subjects_from_ids(features_dir, indx)

        print("=" * 20)
        print("PHASE 3 LSH OUTPUT:")
        print("Total index structure size: ", lsh.get_size())
        print("Number of buckets searched: ", total_buckets_searched)
        print("Number of total image matches found: ", total_non_unique_candidate_images)
        print("Number of unique image matches found: ", len(indx))
        print("Number of near neighbours:", len(indx), " from ", len(images),
              " with data size: ", training_data.shape)
        print("Original image: ", get_subjects_from_ids(features_dir, indx))
        print("False positives: ", len(set(all_candidate_subs) - set(in_sub)))

        candidate_imgs = get_images_from_ids(features_dir, indx)
        print("Misses: ", len(set(tot_in_sub) - set(candidate_imgs)))
        print("=" * 20)

        for can_i in candidate_imgs:
            print(can_i.split("/")[-1].split("\\")[-1])
            image = Image.open(can_i)
            image.show()
        # for i in range(len(indx)):
        #     result = Image.fromarray((images[indx[i]]).astype(np.uint8))
        #     print(image_files[indx[i]].split("/")[-1].split("\\")[-1]) #, d[i])
        #     plt.imshow(images[indx[i]], cmap='gray')
        #     plt.show()
    elif task_num >= 5:
        # HOG with 100 folder works nicely.
        inpMat = training_data
        xq = input_data

        num_bits = int(input("Enter number of bits:"))
        nn_num = int(input("Enter number of nearest neighbours:"))

        df, tot, bins = va_gen(inpMat, num_bits)
        write_va(df, tot, base_dir)

        nn = va_ssa(xq, inpMat, nn_num, num_bits)

        if task_num == 5:
            print("=" * 20 + "\n")
            print("PHASE 3 VAFILES OUTPUT:")
            print("Number of near neighbours:", len(nn), " from ", len(images),
                  " with data size: ", training_data.shape)
            image_files = get_image_file(features_dir, nn)
            for i in range(len(nn)):
                result = Image.fromarray((images[nn[i]]).astype(np.uint8))
                print(image_files[nn[i]].split("/")[-1].split("\\")[-1]) #, d[i])
                plt.imshow(images[i], cmap='gray')
                plt.show()
        if task_num == 6:
            print("Nearest neighbours found: ", nn)
            relevant = input("Enter relevant image indexes (comma separated): ")
            irrelevant = input("Enter irrelevant image indexes (comma separated): ")

            index_list = [int(i) for i in relevant.split(",")]
            feedback_list = [1]*len(relevant.split(","))
            index_list += [int(i) for i in irrelevant.split(",")]
            feedback_list += [-1]*len(irrelevant.split(","))

            # Define labels
            labels_dict = {index_list[i]: feedback_list[i] for i in range(len(index_list))}
            indx_df = pd.DataFrame.from_dict(labels_dict, orient='index', columns=["label"])

            # Define vectors
            vectors_df = pd.DataFrame(training_data)

            # Join vectors to their labels corresponding to indexes
            vectors_df = vectors_df.join(indx_df)
            vectors_df.dropna(inplace=True)

            # Create decision tree
            dt_obj = DecisionTree(vectors_df, max_depth=3, min_support=1.0, min_samples=1)
            # print(dt_obj.print_tree)
            # TRAINING DATA SUMMARY
            df_index = pd.DataFrame(training_data).index
            pred_rel_vals = dt_obj.get_prediction_summary(pd.DataFrame(training_data),
                                                          labels=list(set(labels_dict.values())), show=False)

            image_files = list(get_images_from_ids(features_dir, df_index))
            for i in range(len(pred_rel_vals)):
                result = Image.fromarray((images[df_index[i]]).astype(np.uint8))
                print(image_files[i].split("/")[-1].split("\\")[-1]) #, d[i])
                # print(image_files[i])
                print("irrelevent" if pred_rel_vals[i] == -1 else "relevant")
                plt.imshow(images[df_index[i]], cmap='gray')
                plt.show()
            # for
        elif task_num == 7:
            print("Nearest neighbours found: ", nn)
            # TESTING DATA SETUP
            # test_set_path = input("Enter test path directory:")
            relevant = input("Enter relevant image indexes (comma separated): ")
            irrelevant = input("Enter irrelevant image indexes (comma separated): ")

            index_list = [int(i) for i in relevant.split(",")]
            feedback_list = [1]*len(relevant.split(","))
            index_list += [int(i) for i in irrelevant.split(",")]
            feedback_list += [-1]*len(irrelevant.split(","))

            training_data = training_data[nn]
            # test_features_dir, test_data = get_test_data(technique, k_latent=k_latent, norm_max_latent=norm_max_latent, norm_min_latent=norm_min_latent)

            svm_task_feedback(features_dir, features_dir, training_set_features=training_data,
                              test_set_features=training_data, index_list=index_list, feedback_list=feedback_list)
    else:
        raise NotImplmentedError(f"No implementation found for selected task: {task_num}")

# remove mask for current user
os.umask(0)

input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, \
sub_features_dir, X, Y, technique = initialize_variables()

norm_max_latent = None
norm_min_latent = None
print(f"INPUT IMAGE IS {image_path}")

# Phase1_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir, sub_features_dir)

# Phase2_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir,
#             sub_features_dir, X, Y, technique)

Phase3_main(input_dir, input_k, selected_feature, base_dir, image_path, feature, features_dir,
            sub_features_dir, X, Y, technique)
