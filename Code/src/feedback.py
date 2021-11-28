# import numpy as np
# import pandas as pd
# from sklearn.svm import SVC,SVR
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler

# def feedback_collect(ans,dbImages,featImages):
#     result_size = len(ans)
#     opt = []
#     X = []
#     y = []
#     test = []
#     print("Enter 1->Relevant Image 2->Irrelevant Image 0->Neither")
#     for i in range(0,len(ans)):
#         print("Image :{}".format(dbImages[int(ans[i])][3]))
#         test.append(featImages[int(ans[i])])
#         opt.append(int(input("Enter Option: ")))
#         if(opt[i]>0):
#             X.append(featImages[int(ans[i])])
#             y.append(opt[i])            
#     temp = pd.DataFrame(X)
#     print(y)
#     #clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#     clf = SVC(kernel='rbf', gamma=1E10,C=1E10)
#     clf.fit(X,y)
#     k = clf.predict(featImages)
#     results = []
#     for i in ans:
#         print(k[int(i)])
#     print(k)
#     return results


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


def svm_task_feedback(training_features_dir, test_features_dir, training_set_features = None, test_set_features = None, index_list = None, feedback_list = None):

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

    training_set_X, training_set_Y = [], []
    for image_index in range(len(index_list)):
        training_set_X.append(training_set_features[image_index])
        if feedback_list[image_index] == 1:
            training_set_Y.append(1)
        else:
            training_set_Y.append(-1)


    testing_ids, test_set_X = [], []

    for image_id in range(len(test_set_features)):
        testing_ids.append(test_ids[image_id])
        test_set_X.append(test_set_features[image_id])
    clf.fit(np.array(training_set_X), np.array(training_set_Y))
    output_pred, predictions = clf.predict_result(np.array(test_set_X))

    results = {}
    for i in range(len(predictions)):
        if predictions[i] == 1:
            results[i] = "relevant"
        else:
            results[i] = "irrelevant"

    print(results)
    pp.pprint(results)