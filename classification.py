import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from os import listdir

from os.path import join

from random import choice

from skimage.color import grey2rgb
from skimage.io import imread
from skimage.util import pad
from skimage import transform


IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
CLASSES_NUM = 50
CHANNELS = 3
AS_GREY = False


def read_test(y, img_dir='./public_data/00_input/train/images/'):
    test = np.zeros((len(y), IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    y_test = np.zeros((len(y), CLASSES_NUM))

    for i, img_name in enumerate(y.keys()):
        img = imread(join(img_dir, img_name), as_grey=AS_GREY)

        if len(img.shape) == 2 and not AS_GREY:
            img = grey2rgb(img)

        img, _ = pad_and_scale(img)
        y_test[i] = np.zeros(CLASSES_NUM)

        test[i, ...] = img.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
        y_test[i][y[img_name]] = 1

    return y_test, test


def train_classifier(train_data, img_dir, fast_train=True):
    return
    # checkpoint_callback = ModelCheckpoint(filepath='checkpoint.hdf5', monitor='val_loss', save_best_only=True,
    #                                       mode='auto')
    # early_callback = EarlyStopping(patience=15)
    # lr_callback = ReduceLROnPlateau(patience=5)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(CLASSES_NUM, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    model.fit_generator(
        read_generator(train_data, img_dir, 32, grey=False, permutations=True),
        steps_per_epoch=len(train_data) // 32,
        epochs=1 if fast_train else 5,
        # callbacks=[checkpoint_callback, early_callback, lr_callback],
        # validation_data=(x_val, y_val),
        verbose=1
    )

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        read_generator(train_data, img_dir, 32, grey=False, permutations=True,
                       shuffle=True),
        steps_per_epoch=len(train_data) // 32,
        epochs=1 if fast_train else 100,
        # callbacks=[checkpoint_callback, early_callback, lr_callback],
    )


def classify(model, img_dir):
    print('yupi ka yey')
    dclasses = get_dummy_classes(img_dir)

    pred = model.predict_generator(
        read_generator(
            dclasses,
            img_dir,
            30,
            permutations=False,
            shuffle=False,
            grey=False
        ),
        len(dclasses) // 30
    )

    return {name: pr for pr, name in zip(np.argmax(pred, axis=1), dclasses)}


def get_dummy_classes(img_dir):
    return {img_name: 0 for img_name in listdir(img_dir)}


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
