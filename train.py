import pandas as pd
import numpy as np
import os
import cv2
import random

from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.utils import plot_model
from keras.layers.core import Flatten

from sklearn.model_selection import train_test_split

from image_processing import random_crop
from image_processing import blur
from image_processing import random_rotate_zoom
from image_processing import vertical_flip

train_data_dir = 'data/train'
test_data_dir = 'data/test'
train_labels_path = "data/labels.csv"

df = pd.read_csv(train_labels_path)
df_train, df_test = train_test_split(df, test_size=0.1)

batch_size = 16
epochs = 50


def read_random_image(image_path, resize_size, train):
    if train:
        random_sample = df_train.sample(n=1)
        img_name = random_sample['Image'].iloc[0]
        img = cv2.imread(
            image_path + "/" + img_name + ".jpg")
        crop_chance = random.random()
        blur_chance = random.random()
        rotate_zoom_chance = random.random()
        flip_chance = random.random()
        if crop_chance <= 1:
            img = random_crop(img)
        if blur_chance <= 1:
            img = blur(img, random.choice([3, 5]))
        if rotate_zoom_chance <= 1:
            img = random_rotate_zoom(img)
        if flip_chance <= 1:
            img = vertical_flip(img)
    else:
        random_sample = df_test.sample(n=1)
        img_name = random_sample['Image'].iloc[0]
        img = cv2.imread(
            image_path + "/" + img_name + ".jpg")
    img = cv2.resize(img, (resize_size, resize_size))
    return img, img_name


def import_labels():
    """ This function import labels from csv / Create a list of unique
    labels and a dict with image_name and its label
    Output : dict('image_name': 'label')
    """

    dict_labels = custom_df.set_index('id').to_dict()['breed']
    unique_labels = sorted(list(set(dict_labels.values())))
    for index, label in dict_labels.items():
        dict_labels[index] = unique_labels.index(label)
    return dict_labels, unique_labels
