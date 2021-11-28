import numpy as np
import pandas as pd
from sklearn.svm import SVC,SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def feedback_collect(ans,dbImages,featImages):
    result_size = len(ans)
    opt = []
    X = []
    y = []
    test = []
    print("Enter 1->Relevant Image 2->Irrelevant Image 0->Neither")
    for i in range(0,len(ans)):
        print("Image :{}".format(dbImages[int(ans[i])][3]))
        test.append(featImages[int(ans[i])])
        opt.append(int(input("Enter Option: ")))
        if(opt[i]>0):
            X.append(featImages[int(ans[i])])
            y.append(opt[i])            
    temp = pd.DataFrame(X)
    print(y)
    #clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf = SVC(kernel='rbf', gamma=1E10,C=1E10)
    clf.fit(X,y)
    k = clf.predict(featImages)
    results = []
    for i in ans:
        print(k[int(i)])
    print(k)
    return results

