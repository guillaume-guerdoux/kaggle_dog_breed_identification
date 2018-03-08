import pandas as pd
import numpy as np
import os
import cv2
import random

from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam, SGD
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
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
epochs = 100


def read_random_image(image_path, resize_size, train):
    if train:
        random_sample = df_train.sample(n=1)
        img_name = random_sample['id'].iloc[0]
        img = cv2.imread(
            image_path + "/" + img_name + ".jpg")
        '''crop_chance = random.random()
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
            img = vertical_flip(img)'''
    else:
        random_sample = df_test.sample(n=1)
        img_name = random_sample['id'].iloc[0]
        img = cv2.imread(
            image_path + "/" + img_name + ".jpg")
    img = cv2.resize(img, (resize_size, resize_size))
    return img, img_name


def import_labels():
    """ This function import labels from csv / Create a list of unique
    labels and a dict with image_name and its label
    Output : dict('image_name': 'label')
    """

    dict_labels = df.set_index('id').to_dict()['breed']
    unique_labels = sorted(list(set(dict_labels.values())))
    for index, label in dict_labels.items():
        dict_labels[index] = unique_labels.index(label)
    return dict_labels, unique_labels


def data_generator(batch_size, dict_labels, unique_labels, resize_size, train):
    while True:
        x_train = []
        y_train = []
        for i in range(batch_size):
            img, img_name = read_random_image(
                train_data_dir, resize_size, train)
            x_train.append(img)
            y_train.append(dict_labels[img_name])
        y_train = np_utils.to_categorical(y_train, len(unique_labels))
        x_train = np.array(x_train, dtype="float") / 255.0
        x_train = x_train.reshape(batch_size, resize_size, resize_size, 3)
        yield x_train, y_train


if __name__ == "__main__":
    dict_labels, unique_labels = import_labels()
    resize_size = 224

    input_tensor = Input(shape=(resize_size, resize_size, 3))
    inception_model = InceptionV3(input_tensor=input_tensor, weights="imagenet", include_top=False)
    # base_model.layers.pop()

    x = inception_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(len(unique_labels), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=inception_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in inception_model.layers:
        layer.trainable = False

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(data_generator(batch_size, dict_labels, unique_labels, resize_size, True),
                        samples_per_epoch=250, nb_epoch=epochs,
                        validation_data=data_generator(
                            batch_size, dict_labels, unique_labels, resize_size, False),
                        validation_steps=60)
