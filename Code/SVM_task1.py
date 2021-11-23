import numpy as np
import json
from numpyencoder import NumpyEncoder
from src.svm import SupportVectorMachine, gaussian_kernel
from src.config import *
from src.features_extractor import *
from src.latent_features_extractor import *

def svm_task_1():
    # base_dir = "/home/nisarg1499/assignments/fall21/mwdb/phase3/"
    # training_set_path = "/home/nisarg1499/assignments/fall21/mwdb/phase3/500"
    # test_set_path = "/home/nisarg1499/assignments/fall21/mwdb/phase3/100"
    base_dir = "C:/Users/Krima/Documents/MWDB_image_similarity_search_optimizations/Code/data"
    training_set_path = base_dir + "/500/"
    test_set_path = base_dir + "/100/"
    type_ids_list = {"cc", "con", "emboss", "jitter", "neg", "noise01", "noise02", "original", "poster", "rot", "smooth", "stipple"}


    sub_features_dir = eval(config['Phase1']['sub_features_dir'])
    # TRAINING DATA SETUP
    training_features_dir = os.fspath(os.path.join(base_dir, os.path.normpath(os.path.join(training_set_path, os.pardir)) + "_" +
                                              config['Phase1']['image_feature_dir']))
    # Create folders for images if they do not exist
    if not os.path.isdir(training_features_dir):
        os.makedirs(training_features_dir)
    for dir in sub_features_dir:
        if not os.path.isdir(os.path.join(training_features_dir, dir)):
            # print(os.path.join(training_features_dir, dir))
            os.makedirs(os.path.join(training_features_dir, dir))

    # GET TRAINING DATA
    images, img_all_names = read_all_images(training_set_path, pattern={"X": "", "Y": ""})
    # print(training_features_dir)
    save_all_img_features(images, output_dim, training_features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(training_features_dir)
    training_set_features = feature_dict["cm8x8"]
    training_ids = get_images_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_sub_ids = get_subjects_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_type_ids = get_type_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_sample_ids = get_sample_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)

    # TESTING DATA SETUP
    test_features_dir = os.fspath(os.path.join(base_dir, os.path.normpath(os.path.join(test_set_path, os.pardir)) + "_" +
                                              config['Phase1']['image_feature_dir']))
    # Create folders for images if they do not exist
    if not os.path.isdir(test_features_dir):
        os.makedirs(test_features_dir)
    for dir in sub_features_dir:
        if not os.path.isdir(os.path.join(test_features_dir, dir)):
            # print(os.path.join(test_features_dir, dir))
            os.makedirs(os.path.join(test_features_dir, dir))

    # GET TESTING DATA
    images, img_all_names = read_all_images(test_set_path, pattern={"X": "", "Y": ""})
    save_all_img_features(images, output_dim, test_features_dir, sub_features_dir, hog_dict, feature_visualization=False,
                          img_ids=img_all_names)
    feature_dict = get_feature_dict_file(test_features_dir)
    test_set_features = feature_dict["cm8x8"]

    test_ids = get_images_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    print("test_ids", test_ids)
    test_sub_ids = get_subjects_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    test_type_ids = get_type_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    test_sample_ids = get_sample_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)

    images_assosciation = {}
    negative_image_assosciation = {}
    clf = SupportVectorMachine(gaussian_kernel, C=500)
    count = 1
    for original_type_id in type_ids_list:
        print(count)
        count = count + 1
        training_image_ids, training_set_X, training_set_Y = [], [], []
        for image_id in range(len(training_set_features)):
            # subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
            training_image_ids.append(training_ids[image_id])
            training_set_X.append(training_set_features[image_id])
            if training_type_ids[image_id] == original_type_id:
                training_set_Y.append(1)
            else:
                training_set_Y.append(-1)

        test_image_ids, test_set_X = [], []

        for image_id in range(len(test_set_features)):
            # subject_id, type_id, second_id = extract_subject_id_image_type_and_second_id(image_id)
            test_image_ids.append(test_ids[image_id])
            test_set_X.append(test_set_features[image_id])
        clf.fit(np.array(training_set_X), np.array(training_set_Y))
        prediction_values, predictions = clf.training_result(np.array(test_set_X))

        index = 0
        for image_id in test_image_ids:
            if predictions[index] == 1:
                if image_id in images_assosciation:
                    if any(original_type_id in type for type in images_assosciation[image_id]):
                        for value in images_assosciation[image_id]:
                            value[original_type_id] = max(value[original_type_id], prediction_values[index])
                    else:
                        y = {original_type_id: prediction_values[index]}
                        images_assosciation[image_id].append(y)
                else:
                    x = {original_type_id: prediction_values[index]}
                    images_assosciation[image_id] = [x]
            else:
                if image_id in negative_image_assosciation:
                    if any(original_type_id in type for type in negative_image_assosciation[image_id]):
                        for value in negative_image_assosciation[image_id]:
                            value[original_type_id] = max(value[original_type_id], prediction_values[index])
                    else:
                        y = {original_type_id: prediction_values[index]}
                        negative_image_assosciation[image_id].append(y)
                else:
                    x = {original_type_id: prediction_values[index]}
                    negative_image_assosciation[image_id] = [x]
            index = index + 1

    # test_result = {}
    # index = 0
    #
    # for image_id in test_image_ids:
    #     test_result[image_id] = predictions[index]
    #     index = index + 1
    #
    # print(test_result)

    classifier_results = {}

    for image_id in images_assosciation:
        ans = -1
        for X in images_assosciation[image_id]:
            for key in X:
                if X[key] > ans:
                    classifier_results[image_id] = key
                    ans = X[key]
    remaining_images = []

    for image_id in test_set_features:
        if image_id not in classifier_results:
            remaining_images.append(image_id)
            ans = -1000000000
            for X in negative_image_assosciation[image_id]:
                for key in X:
                    if X[key] > ans:
                        classifier_results[image_id] = key
                        ans = X[key]


    jsonString = json.dumps(images_assosciation, cls=NumpyEncoder)
    jsonFile = open("SVM_Classifier_Values_Task1.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(negative_image_assosciation, cls=NumpyEncoder)
    jsonFile = open("SVM_Classifier_negative_Values_Task1.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(classifier_results, cls=NumpyEncoder)
    jsonFile = open("SVM_Classifier_Results_Task1.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    print(remaining_images)