from typing import Counter
import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cityblock,euclidean,minkowski
from collections import Counter
import os

def va_gen(inpMat,b):
    df = pd.DataFrame(inpMat)
    d = df.shape[1]
    lng = df.shape[0]
    tot = []
    bins = []
    p = np.zeros((lng,d))
    for j in range(0,d):
        if(j+1 <= b%d):
            bj = math.floor(b/d)+1
        else:
            bj = math.floor(b/d)
        nRegions = 2**bj
        tot.append(bj)
        buckets = []
        data = np.array(df.iloc[:,j])
        buckets,binst = pd.cut(data, nRegions, labels=False, retbins=True)
        df.iloc[:,j] = buckets
        bins.append(binst)
    t1 = pd.DataFrame(bins)
    return (df,tot,bins)

def write_va(modData,numBits, base_dir):
    temp1 = np.array(modData)
    x = temp1.shape[0]
    y = temp1.shape[1]
    vaFile = ""
    strings = ["" for i in range(x)]
    for i in range(0,x):
        for j in range(0,y):
            strings[i] = strings[i]+(bin(modData.iloc[i,j])[2:].zfill(numBits[j]))
    # counter = Counter(strings)
    # print(counter)
    for i in range(0,x):
        vaFile = vaFile+strings[i]
    text_file = open(os.path.join(base_dir, "va_file.txt"), "w")
    n = text_file.write(vaFile)
    text_file.close()
    print("Size of Index Structure in bytes : {}".format(len(vaFile)//8))
    return vaFile

def modified_query(vq,p):
    imgLen = len(vq)
    modQuery = []
    flag = 0
    for i in range(0,imgLen):
        flag = 0
        for j in range(0,len(p[i])-1):
            if(vq[i]>= p[i][j] and vq[i]< p[i][j+1]):
                modQuery.append(j)
                flag = 1
        if (flag==0):
            modQuery.append(j+1)
    return modQuery

def get_bounds(vq,ri,p):
    l = []
    imgLen = len(vq)
    res = 0
    rq = modified_query(vq,p)
    for j in range(0,imgLen):
        if(ri[j]<rq[j]):
            l.append(vq[j] - p[j][ri[j]+1])
        elif(ri[j] == rq[j]):
            l.append(0)
        else:
            l.append(p[j][ri[j]] - vq[j])
    for i in range(0,len(l)):
        res = res+(l[i]**3)
    return res**(1/3)

def InitCandidate(n,dst):
    for i in range(0,n):
        dst[i] = float('inf')
    return float('inf'), dst

def Candidate(d,i,n,dst,ans):
    if(d<dst[n]):
        dst[n] = d
        ans[n] = i
        temp = []
        sort_vals = np.argsort(dst)
        dst.sort()
        for i in range(0,len(sort_vals)):
            temp.append(ans[sort_vals[i]])
        ans = temp
    return dst[n],ans

def va_ssa(vq,vi,n,b):
    dst = np.zeros((n))
    count = 0
    d,dst = InitCandidate(n,dst)
    ans = np.zeros((n)).astype(int)
    modData,numBits,p = va_gen(vi,b)
    for i in range(0,len(vi)):
        li = get_bounds(vq,modData.iloc[i,:],p)
        if(li<d):
            count = count+1
            d,ans = Candidate(minkowski(vq,vi[i],3),i,n-1,dst,ans)
    eval = va_eval(vq,vi,n)
    crct_imgs = 0
    for i in range(0,n):
        if(ans[i]==eval[i]):
            crct_imgs+=1
    print("Number of Images Considered : {}".format(count))
    print("False Positive Rate: {}".format((count - crct_imgs)/count))
    print("Miss Rate : {}".format((n-crct_imgs)/n))
    return ans

def va_eval(vq,vi,n):
    ans = []
    res = []
    for i in range(0,len(vi)):
        ans.append(minkowski(vq,vi[i],3))
    vals = np.array(np.argsort(ans)[:n])
    return vals
