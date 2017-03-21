import os
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
from numpy import *


def scan_files(directory, prefix=None, postfix='.jpg'):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    return files_list


def out_thumbnails(files_list, size, out_directory, postfix='.png'):
    new_file_list = []
    for file in files_list:
        try:
            image = Image.open(file).convert('L')
        except IOError:
            continue
        thumbnails_image = image.resize(size, Image.ANTIALIAS)
        out_file_name = file.replace('photo', out_directory)
        try:
            thumbnails_image.save(out_file_name, quality=100)
        except OSError:
            dir_name = os.path.dirname(out_file_name)
            os.makedirs(dir_name)
            print(dir_name)
            thumbnails_image.save(out_file_name, quality=100)
        new_file_list.append(out_file_name)
    return new_file_list


def get_photo_data(file_list):
    data = None
    for file in file_list:
        im = array(Image.open(file).convert('L'), 'f')
        print(im.shape, im.dtype)
        im = im.reshape((1, im.size))
        try:
            data = np.row_stack((data, im))
        except ValueError:
            if data is None:
                data = im
            else:
                return None
    return data