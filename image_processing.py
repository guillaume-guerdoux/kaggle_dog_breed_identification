import cv2
import numpy as np
from random import randint, uniform
import os


def random_crop(img):
    height, width = img.shape
    try:
        top_crop = randint(35, int(height / 8))
        bottom_crop = height - randint(35, int(height / 8))
        left_crop = randint(35, int(width / 8))
        right_crop = width - randint(35, int(width / 8))
    except ValueError:
        top_crop = randint(0, int(height / 8))
        bottom_crop = height - randint(0, int(height / 8))
        left_crop = randint(0, int(width / 8))
        right_crop = width - randint(0, int(width / 8))
    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]
    return crop_img


def blur(img, kernel_size):
    blur_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blur_img


def random_rotate_zoom(img):
    negative_angle = randint(-10, -10)
    positive_angle = randint(10, 10)
    if abs(negative_angle) >= positive_angle:
        angle = negative_angle
    else:
        angle = positive_angle
    zoom = uniform(1.1, 1.4)
    height = img.shape[0]
    width = img.shape[1]
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.2)
    rotated_zoomed_img = cv2.warpAffine(img, M, (width, height))
    return rotated_zoomed_img


def vertical_flip(img):
    return cv2.flip(img, 1)
