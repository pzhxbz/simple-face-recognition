from numpy import *
import numpy as np


def zero_mean(data_mat):
    mean_val = np.mean(data_mat, axis=0)     #按列求均值，即求各个特征的均值
    new_data = data_mat - mean_val
    return new_data, mean_val


def pca(data_mat, n):
    new_data, mean_val = zero_mean(data_mat)
    cov_mat = np.cov(new_data, rowvar=0)    #求协方差矩阵,return ndarray；若rowvar非0，一列代表一个样本，为0，一行代表一个样本
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
    eig_val_indice = np.argsort(eig_vals)            #对特征值从小到大排序
    n_eig_val_indice=eig_val_indice[-1:-(n+1):-1]   #最大的n个特征值的下标
    n_eig_vect=eig_vects[:, n_eig_val_indice]        #最大的n个特征值对应的特征向量
    low_data_mat=new_data*n_eig_vect               #低维特征空间的数据
    return low_data_mat