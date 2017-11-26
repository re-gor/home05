from os.path import join
from os import listdir
from random import choice

from skimage.color import grey2rgb
from skimage.io import imread
from skimage.util import pad
from skimage import transform
import numpy as np

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
CLASSES_NUM = 50


def train_classifier():
    pass

def classify():
    pass



def scale_image(img):
    """

    :type img: np.ndarray
    :type points_vec: np.ndarray
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    assert(img.shape[0] == img.shape[1])

    shape = img.shape[:2]
    factor = IMAGE_HEIGHT / shape[0]

    new_img = transform.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH), mode='symmetric')

    return new_img, factor


def pad_image(img):
    r = img.shape[0]
    c = img.shape[1]

    row_pad = max(0, c-r)
    col_pad = max(0, r-c)

    if row_pad == col_pad == 0:
        return img, (0, 0)

    row_pad_left = row_pad // 2
    row_pad_right = row_pad_left + row_pad % 2

    col_pad_top = col_pad // 2
    col_pad_bottom = col_pad_top + col_pad % 2

    padding = [(row_pad_left, row_pad_right), (col_pad_top, col_pad_bottom), (0, 0)]

    if len(img.shape) < 3:
        padding = padding[:2]

    new_img = pad(img, padding, 'constant')

    return new_img, (col_pad_bottom, row_pad_left)


def pad_and_scale(_img):
    img, pad_movement = pad_image(_img)
    img, scale_factor = scale_image(img)

    return img, (pad_movement, scale_factor)


def flip_img(img):
    assert(img.shape[0] == img.shape[1])

    return img[:, ::-1].copy()


def rotate_img(_img):
    angle = 20 * np.random.random() - 10
    img = transform.rotate(_img, angle=angle)

    return img


def frame_image(img):
    factor = 0.05
    height, width = img.shape[:2]
    left, right = np.random.randint(0, factor * width), np.random.randint(width - factor * width, width)
    low, top = np.random.randint(0, factor * height), np.random.randint(height - factor * height, height)

    new_img = img[low:top, left:right, ...]

    return new_img


PERMUTATIONS = [
    lambda img: pad_and_scale(img)[0],
    lambda img: flip_img(pad_and_scale(img)[0]),
    lambda img: rotate_img(pad_and_scale(img)[0]),
    lambda img: pad_and_scale(frame_image(img))[0]
]


def choice_permutation(img):
    perm = choice(PERMUTATIONS)
    return perm(img)


def read_generator(y_csv, train_img_dir, batch_size, permutations=False, shuffle=True, grey=True):
    channels = 1 if grey else 3
    batch_features = np.zeros((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, channels))
    batch_labels = np.zeros((batch_size, CLASSES_NUM))
    names = sorted(list(y_csv.keys()))

    while True:
        if shuffle:
            np.random.shuffle(names)

        for ind, name in enumerate(names):
            i = ind % batch_size

            img = imread(join(train_img_dir, name), as_grey=grey)

            if len(img.shape) == 2 and not grey:
                img = grey2rgb(img)

            if permutations:
                img = choice_permutation(img)
            else:
                img = pad_and_scale(img)[0]

            batch_labels[i] = np.zeros(CLASSES_NUM)
            batch_features[i, ...], batch_labels[i][y_csv[name]] = img.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, channels)), 1

            if ind % batch_size == batch_size - 1 or ind == len(names) - 1:
                yield batch_features, batch_labels
