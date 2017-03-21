import numpy as np
from PIL import Image
import os
import random


class ImageData:
    def __init__(self, data=None, flag=None):
        self.__data = data
        self.__flag = flag
        try:
            os.makedirs('D:\\code\\pyproject\\manface\\ImageData')
        except FileExistsError:
            return

    def save_data(self):
        np.save(os.getcwd()+'\\ImageData\\SaveData.npy', self.__data)
        np.save(os.getcwd()+'\\ImageData\\SaveFlag.npy', self.__flag)

    def load_data(self):
        try:
            self.__data = np.load(os.getcwd()+'\\ImageData\\SaveData.npy')
        except FileNotFoundError:
            self.__data = None
        try:
            self.__flag = np.load(os.getcwd()+'\\ImageData\\SaveFlag.npy')
        except FileNotFoundError:
            self.__flag = None

    def add_data(self, data):
        if self.__data is None:
            self.__data = data
            return
        self.__data = np.row_stack((self.__data, data))

    def add_flag(self, flag):
        if self.__flag is None:
            self.__flag = flag
            return
        self.__flag = np.row_stack((self.__flag, flag))

    def get_data(self):
        return self.__data

    def get_flag(self):
        return self.__flag

    def get_train_data(self, num):
        if self.__data.shape[0] != self.__flag.shape[0]:
            raise TypeError('please check the data\'s and flag\'s shape')
        random_num = random.sample(range(self.__data.shape[0]), num)
        return self.__data[random_num, :], self.__flag[random_num, :]