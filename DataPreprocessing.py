import os
import numpy as np

# Enhancer Sequence Preprocessing
'''
Train Dataset
-enhancer.cv.txt: Enhancer Sequence 1484개
--strong_742.txt: Strong Enhancer Sequence 742개
--weak_742.txt  : Weak Enhancer Sequence 742개
-non.cv.txt     : Non-Enhancer Sequence 1484개

Test Dataset
-enhancer.ind.txt: Enhancer Sequence 200개
--strong_100.txt : Strong Enhancer Sequence 100개
--weak_100.txt   : Weak Enhancer Sequence 100개
-non.ind.txt     : Non-Enhancer Sequence 200개
'''


def readSequence():
    
    ## Enhancer & Non-Enhancer Sets
    with open(os.path.join(os.getcwd(), "dataset/train/enhancer.cv.txt")) as f:
        enhancer_cv = f.readlines()
        enhancer_cv = [s.strip() for s in enhancer_cv]
    with open(os.path.join(os.getcwd(), "dataset/train/non.cv.txt")) as f:
        non_cv = f.readlines()
        non_cv = [s.strip() for s in non_cv]
    with open(os.path.join(os.getcwd(), "dataset/test/enhancer.ind.txt")) as f:
        enhancer_ind = f.readlines()
        enhancer_ind = [s.strip() for s in enhancer_ind]
    with open(os.path.join(os.getcwd(), "dataset/test/non.ind.txt")) as f:
        non_ind = f.readlines()
        non_ind = [s.strip() for s in non_ind]
    
    ## Strong/Weak Enhancer Sets
    with open(os.path.join(os.getcwd(), "dataset/train/strong_742.txt")) as f:
        strong_742 = f.readlines()
        strong_742 = [s.strip() for s in strong_742]
    with open(os.path.join(os.getcwd(), "dataset/train/weak_742.txt")) as f:
        weak_742 = f.readlines()
        weak_742 = [s.strip() for s in weak_742]
    with open(os.path.join(os.getcwd(), "dataset/test/strong_100.txt")) as f:
        strong_100 = f.readlines()
        strong_100 = [s.strip() for s in strong_100]
    with open(os.path.join(os.getcwd(), "dataset/test/weak_100.txt")) as f:
        weak_100 = f.readlines()
        weak_100 = [s.strip() for s in weak_100]

    return enhancer_cv, non_cv, enhancer_ind, non_ind, strong_742, weak_742, strong_100, weak_100


def removeName_PN(data): ## Enhancer(Positive) / Non-Enhancer(Negative) Sequence의 이름 제거
    data_new = []
    for i in range(1,len(data),2):
        data_new.append(data[i])
    return data_new

def removeName_SW(data): ## Strong/Weak Enhancer Sequence의 이름 제거
    data_new = []
    for i in range(1,len(data),5):
        data_new.append(data[i].upper()+data[i+1].upper()+data[i+2].upper()+data[i+3].upper())
    return data_new


def GetSets():
    enhancer_cv, non_cv, enhancer_ind, non_ind, strong_742, weak_742, strong_100, weak_100 = readSequence()

    enhancer_cv = removeName_PN(enhancer_cv)
    non_cv = removeName_PN(non_cv)
    enhancer_ind = removeName_PN(enhancer_ind)
    non_ind = removeName_PN(non_ind)
    X_train_pn = np.concatenate([enhancer_cv, non_cv], axis=0)
    X_test_pn = np.concatenate([enhancer_ind, non_ind], axis=0)
    y_train_pn = np.concatenate([np.ones((len(enhancer_cv),)), np.zeros((len(non_cv),))], axis=0)
    y_test_pn = np.concatenate([np.ones((len(enhancer_ind),)), np.zeros((len(non_ind),))], axis=0)
    
    strong_742 = removeName_SW(strong_742)
    weak_742 = removeName_SW(weak_742)
    strong_100 = removeName_SW(strong_100)
    weak_100 = removeName_SW(weak_100)
    X_train_sw = np.concatenate([strong_742, weak_742], axis=0)
    X_test_sw = np.concatenate([strong_100, weak_100], axis=0)
    y_train_sw = np.concatenate([np.ones((len(strong_742),)), np.zeros((len(weak_742),))], axis=0)
    y_test_sw = np.concatenate([np.ones((len(strong_100),)), np.zeros((len(weak_100),))], axis=0)

    return X_train_pn, X_test_pn, y_train_pn, y_test_pn, X_train_sw, X_test_sw, y_train_sw, y_test_sw




