import numpy as np
import json
from numpyencoder import NumpyEncoder
from src.svm import SupportVectorMachine, gaussian_kernel
from src.config import *
from src.features_extractor import *
from src.latent_features_extractor import *
from sklearn.metrics import confusion_matrix
import pprint
pp = pprint.PrettyPrinter(indent=4)


def svm_task_3(training_features_dir, test_features_dir, training_set_features=None, test_set_features=None):
    sample_ids_list = []
    for i in range(1, 11):
        sample_ids_list.append(str(i))

    training_ids = get_images_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_sub_ids = get_subjects_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_type_ids = get_type_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)
    training_sample_ids = get_sample_from_ids(training_features_dir, range(training_set_features.shape[0]), reverse_dict=True)

    test_ids = get_images_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    test_sub_ids = get_subjects_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    test_type_ids = get_type_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)
    test_sample_ids = get_sample_from_ids(test_features_dir, range(test_set_features.shape[0]), reverse_dict=True)

    initial_values = {}
    negative_values = {}
    clf = SupportVectorMachine(gaussian_kernel, C=500)
    count = 1

    for original_sample_id in sample_ids_list: 
        count = count + 1
        training_set_X, training_set_Y = [], []
        for image_id in range(len(training_set_features)):
            training_set_X.append(training_set_features[image_id])
            if training_sample_ids[image_id] == original_sample_id:
                training_set_Y.append(1)
            else:
                training_set_Y.append(-1)

        testing_ids, test_set_X = [], []

        for image_id in range(len(test_set_features)):
            testing_ids.append(test_ids[image_id])
            test_set_X.append(test_set_features[image_id])
        clf.fit(np.array(training_set_X), np.array(training_set_Y))
        output_pred, predictions = clf.predict_result(np.array(test_set_X))

        cnt = 0
        for image_id in testing_ids:
            if predictions[cnt] == 1:
                if image_id in initial_values:
                    if any(original_sample_id in type for type in initial_values[image_id]):
                        for value in initial_values[image_id]:
                            if original_sample_id in value:
                                value[original_sample_id] = max(value[original_sample_id], output_pred[cnt])
                    else:
                        y = {original_sample_id: output_pred[cnt]}
                        initial_values[image_id].append(y)
                else:
                    x = {original_sample_id: output_pred[cnt]}
                    initial_values[image_id] = [x]
            else:
                if image_id in negative_values:
                    if any(original_sample_id in type for type in negative_values[image_id]):
                        for value in negative_values[image_id]:
                            if original_sample_id in value:
                                value[original_sample_id] = max(value[original_sample_id], output_pred[cnt])
                    else:
                        y = {original_sample_id: output_pred[cnt]}
                        negative_values[image_id].append(y)
                else:
                    x = {original_sample_id: output_pred[cnt]}
                    negative_values[image_id] = [x]
            cnt = cnt + 1


    results = {}
    for image_id in initial_values:
        ans = -1
        for X in initial_values[image_id]:
            for key in X:
                if X[key] > ans:
                    results[image_id] = key
                    ans = X[key]
    
    jsonString = json.dumps(initial_values, cls=NumpyEncoder, indent=4)
    jsonFile = open("Values_Task3.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    remaining_images = []

    for _id in range(len(test_set_features)):
        image_id = test_ids[_id]
        if image_id not in results:
            remaining_images.append(image_id)
            ans = -1000000
            for X in negative_values[image_id]:
                for key in X:
                    if X[key] > ans:
                        results[image_id] = key
                        ans = X[key]

    jsonString = json.dumps(negative_values, cls=NumpyEncoder, indent=4)
    jsonFile = open("Negative_Values_Task3.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    jsonString = json.dumps(results, cls=NumpyEncoder, indent=4)
    jsonFile = open("Results_Task3.json","w")
    jsonFile.write(jsonString)
    jsonFile.close()

    actual_val = []
    predicted_val = []
    for key in results:
        type_img = get_sample_from_image_path(key)
        actual_val.append(type_img)
        predicted_val.append(results[key])

    print("Results")
    pp.pprint(results)
    labels =[]
    for i in range(1, 11):
        labels.append(str(i))
    conf_mat = confusion_matrix(actual_val, predicted_val, labels = labels)
    print("Confusion Matrix")
    print(conf_mat)

    FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  
    FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
    TP = np.diag(conf_mat)
    TN = conf_mat.sum() - (FP + FN + TP)
    TN = conf_mat.sum() - (FP + FN + TP)
    FPR = FP/(FP+TN)
    FPR = np.nan_to_num(FPR)
    print("False Positive rate: ", FPR)
    FNR = FN/(TP+FN)
    FNR = np.nan_to_num(FNR)
    print("False Negative rate: ", FNR)
