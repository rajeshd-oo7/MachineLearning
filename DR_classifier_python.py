# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 10:08:06 2024

@author: Rajesh
"""

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
#import refresh_train_v_4d_ncond_sens


def compute_matrices_v2(x, y, lambda_, gamma, dqu, r, K):
        

    x = np.tile(x, (K.shape[0], 1))
    y = np.tile(y, (K.shape[0], 1))

    K = K.T
    x = x.T
    y = y.T

    output_g = np.sum( (1. / ((1 + lambda_ * (np.sum(K**(2 * r), axis=0)))**1)) * np.cos(1 * 2 * np.pi * np.sum(K * (x - y), axis=0)), axis=0)
    
    return output_g


#import numpy as np

def compute_discrepancy_fill_distance(point_set, full_point_set):
    num_data_points = point_set.shape[0]
    dist_matrix = np.zeros((full_point_set.shape[0], num_data_points))

    for ak in range(num_data_points):
        for jk in range(num_data_points):
            ttemp = full_point_set[jk, :]
            P = ttemp - point_set[ak, :]
            dist_array = np.sum(P * P)  # Original
            dist_matrix[jk, ak] = dist_array

    dist_matrix[dist_matrix == 0] = 10**-6
    dist_matrix = dist_matrix.T
    adist = np.min(dist_matrix, axis=0)
    discrepancy_measure = np.sqrt(np.sum(adist)) / len(adist)
    #print(discrepancy_measure.shape)
    return discrepancy_measure

def cartesian_product(arrays):
    la = len(arrays)
    dtype = np.find_common_type([a.dtype for a in arrays], [])
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
#=============================================================================

#import numpy as np
#from itertools import product


def cartprod(*args):
    """
    CARTPROD Cartesian product of multiple sets.

    X = CARTPROD(A,B,C,...) returns the cartesian product of the sets 
    A,B,C, etc, where A,B,C, are numerical vectors.  

    Example: A = [-1, -3, -5]; B = [10, 11]; C = [0, 1];

    X = cartprod(A,B,C)
    """
    
    num_sets = len(args)
    size_this_set = []
    
    for i in range(num_sets):
        this_set = np.sort(args[i])
        if this_set.ndim != 1:
            raise ValueError('All inputs must be vectors.')
        if not np.issubdtype(this_set.dtype, np.number):
            raise ValueError('All inputs must be numeric.')
        if len(this_set) != len(np.unique(this_set)):
            raise ValueError(f'Input set {i + 1} contains duplicated elements.')
        
        size_this_set.append(len(this_set))
        args[i] = this_set

    X = np.zeros((np.prod(size_this_set), num_sets))
    
    for i in range(X.shape[0]):
        ix_vect = np.unravel_index(i, size_this_set)
        
        for j in range(num_sets):
            X[i, j] = args[j][ix_vect[j]]
    
    return X




def refresh_train_v_4d_ncond_sens(train_pca_df_vectors, data_label_train, pretest_pca_df_vectors, test_pca_df_vectors, data_label_test, ld, K, alp, sens):
    
    Eta_1 = compute_discrepancy_fill_distance(train_pca_df_vectors, test_pca_df_vectors[:alp, :])

    mm = train_pca_df_vectors.shape[1]
    k = mm // 2 + 1 if mm % 2 == 0 else round(mm / 2) + 1

    alpha_min = 1 / (2 * k - 1)
    alpha_max = 1 / (2 * k - 3)

    alpha_sens = 0.2
    alpha = ((1 - alpha_sens) * alpha_min + (alpha_sens) * alpha_max) / 1

    beta_min = alpha * (2 * k - 1)
    beta_max = 1 + 2 * alpha

    bsens = 0.2
    beta = ((1 - bsens) * beta_min + (bsens) * beta_max) / 1

    n = train_pca_df_vectors.shape[0]
    lambda_ = Eta_1 ** (-alpha)
    lambda_ *= 10 ** 0
    omega = (ld * Eta_1 ** (-beta)) / 1
    omega = np.round(omega / 15).astype(int) 
    #print(omega)
    K = cartesian_product([np.arange(-omega[0], omega[0] + 1), np.arange(-omega[1], omega[1] + 1),np.arange(-omega[2], omega[2] + 1), np.arange(-omega[3], omega[3] + 1)])

    an = train_pca_df_vectors.shape[0]
    asum = np.zeros((an, an))
    for sk in range(an):
        x = train_pca_df_vectors[sk, :]
        for jk in range(an):
            y = train_pca_df_vectors[jk, :]
            output_g = compute_matrices_v2(x, y, lambda_, 1, omega, k, K)
            asum[sk, jk] = output_g

    asum = asum / an
    asum += (1 / (lambda_ ** 2)) * np.eye(an)

    c = np.linalg.inv(asum) @ data_label_train.T
    #print(np.linalg.cond(asum))

    sum_test = np.zeros((pretest_pca_df_vectors.shape[0], an))
    for sk in range(pretest_pca_df_vectors.shape[0]):
        x = pretest_pca_df_vectors[sk, :]
        for jk in range(an):
            y = train_pca_df_vectors[jk, :]
            output_g = compute_matrices_v2(x, y, lambda_, 1, omega, k, K)
            sum_test[sk, jk] = output_g

    elastic_computation = sum_test @ c / an
    plt.hist(elastic_computation)

    
    test_pca_df_vectors = test_pca_df_vectors[:200, :]
    data_label_test = data_label_test[:200]
    sum_test = np.zeros((test_pca_df_vectors.shape[0], an))
    for sk in range(test_pca_df_vectors.shape[0]):
        x = test_pca_df_vectors[sk, :]
        for jk in range(an):
            y = train_pca_df_vectors[jk, :]
            output_g = compute_matrices_v2(x, y, lambda_, 1, omega, k, K)
            sum_test[sk, jk] = output_g

    test_eval = sum_test @ c / an

    dsignal = test_eval
    test_signal = np.zeros_like(dsignal)
    test_signal[dsignal > 0.5] = 1
    test_signal[dsignal < 0.5] = 0
    test_accuracy = 100 - 100 * np.sum(np.abs(data_label_test - test_signal)) / len(test_signal)
    #empirical_test_confidence = 100 - 100 * np.sum(np.abs(data_label_test - dsignal) ** 1) / len(test_signal)

    AUC = 0
    
    return elastic_computation, test_accuracy, AUC, Eta_1




# Load data
nn = np.loadtxt('D:\Work\Today\DiabeticRetinopathy_normalized_4_python.csv', delimiter=',')
data_label = np.loadtxt('D:\Work\Today\DiabeticRetinopathy_data_label.csv', delimiter=',')

a, b = nn.shape
#print(nn.shape)
for ak in range(b):
    nn[:, ak] = (nn[:, ak] - np.min(nn[:, ak])) + 0.0001
    nn[:, ak] = nn[:, ak] / np.max(np.abs(1.0001 * nn[:, ak]))

DA = nn - np.mean(nn, axis=0)
C = np.matmul(DA.T, DA)
V, E = np.linalg.eig(C)
plt.plot(np.abs(V))
nn = np.matmul(DA, E)
#ld = (np.abs(np.diag(V))) ** 0.5
ld = (np.abs(V)) ** 0.5
#print(C.shape)
a, b = nn.shape
for ak in range(b):
    nn[:, ak] = (nn[:, ak] - np.min(nn[:, ak])) + 0.0001
    nn[:, ak] = nn[:, ak] / np.max(np.abs(1.0001 * nn[:, ak]))

ld = ld / np.max(ld)
nn = nn / np.max(np.abs(nn))
tts = np.random.permutation(3662)
tts1 = tts[:280]
tts2 = tts[1000:]

train_data = nn[tts1, :]
test_data = nn[tts2, :]
data_label = data_label
train_data_val = data_label[tts1]
test_data_val = data_label[tts2]

ax, bx = train_data.shape
tr_val_idx = np.arange(ax)
track_of_num_centroid = 20
rem_track_centroids = tr_val_idx
track_centroids = rem_track_centroids[:10]
rem_track_centroids = rem_track_centroids[10:]
K = 0
alp = len(track_centroids)
iteration_t = 1
plt.figure()
#plt.hold(True)

while len(rem_track_centroids) > 3:
    train_data_sequence = train_data[track_centroids, :]
    train_data_val_sequence = train_data_val[track_centroids]

    rem_train_data_sequence = train_data[rem_track_centroids, :]
    rem_train_data_val_sequence = train_data_val[rem_track_centroids]

    elastic_computation, test_accuracy, AUC, Eta_1 = refresh_train_v_4d_ncond_sens(
        train_data_sequence, train_data_val_sequence, rem_train_data_sequence, test_data, test_data_val, ld, K, alp, 1.0 * 10 ** (-8)
    )

    tested_elast = np.abs(elastic_computation - 0.5)
    gy = np.argsort(tested_elast)
    new_centers = rem_track_centroids[gy[:1]]
    track_centroids = np.concatenate((track_centroids, new_centers))
    alp = len(track_centroids)
    delta = 0.01
    new_centers2 = []
    for abk in range(len(elastic_computation)):
        if (elastic_computation[abk] > 1 - delta) or (elastic_computation[abk] < delta):
            new_centers2.append(rem_track_centroids[abk])

    if iteration_t > 1:
        rem_track_centroids = np.setdiff1d(rem_track_centroids, new_centers)
    else:
        rem_track_centroids = np.setdiff1d(rem_track_centroids, new_centers)

    iteration_t += 1
    print('Classification Accuracy is : ',test_accuracy)
    #print(len(rem_track_centroids))
    print('No . of labeled Samples used is : ',len(track_centroids))
    #print(AUC)







    


