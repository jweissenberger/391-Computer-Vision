# Jack Weissenberger May 2019
# Computer Vision final, Whale Identification

import numpy as np
import pylab as pl
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from skimage.transform import rotate

from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential

from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random


def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, 100, 100, 3))
    count = 0

    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1

    return X_train


def randRange(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return pl.rand() * (b - a) + a


def randomAffine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[0]//10, im.shape[0]//10),
                                         randRange(-im.shape[1]//10, im.shape[1]//10)))
    return warp(im, tform.inverse, mode='reflect')


def randomPerspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1/4
    A = pl.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = pl.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))],
                  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1-region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(im.shape[0] * (1-region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(0, im.shape[0] * region))],
                  ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])


def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1/10
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1-margin), im.shape[0])),
           int(randRange(im.shape[1] * (1-margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]


def randomIntensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))),
                             out_range=tuple(pl.percentile(im, (randRange(0,10), randRange(90,100)))))


def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))


def randomGaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=randRange(0, 5))


def randomFilter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    Filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)


def imageAugmentation(data, m, dataset):
    """
    This method creates 4D array that augments each image 11 different ways. This array will contain the original image
    as well as the augmented images
    :param data: image data you would like to augment
    :param m: the number of images you are augmenting
    :param dataset: the dataset that you are augmenting, this will be used in the file path loading
    :return: 4D array that contains the origional images as well as 11 augmented images
    """
    print("Preparing images")
    X_train_flip = np.zeros((m, 100, 100, 3))
    X_train_rot = np.zeros((m, 100, 100, 3))
    X_train_rand_noise = np.zeros((m, 100, 100, 3))
    X_train_rand_filter = np.zeros((m, 100, 100, 3))
    X_train_rand_Gaussian = np.zeros((m, 100, 100, 3))
    X_train_rand_Gamma = np.zeros((m, 100, 100, 3))
    X_train_rand_Intensity = np.zeros((m, 100, 100, 3))
    X_train_rand_Crop = np.zeros((m, 100, 100, 3))
    X_train_rand_Perspective = np.zeros((m, 100, 100, 3))
    X_train_rand_Affine = np.zeros((m, 100, 100, 3))
    X = np.zeros((m, 100, 100, 3))
    count = 0

    for fig in data['Image']:
        #load images into images of size 100x100x3
        img = image.load_img("../input/"+dataset+"/"+fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train_rand_Affine[count] = randomAffine(x)  # performs a random Affine transformation of the data

        X_train_rand_Perspective[count] = randomPerspective(x)  # randomly changes the perspective of the image

        X_train_rand_Crop[count] = randomCrop(x)  # randomly crops the image

        X_train_rand_Intensity[count] = randomIntensity(x)  # randomly changes the intensity of the image

        X_train_rand_Gamma[count] = randomGamma(x)  # adds a random Gamma filter to the image

        X_train_rand_Gaussian[count] = randomGaussian(x)  # adds a random gaussian filter to the image

        X_train_rand_filter[count] = randomFilter(x)  # adds a random filter to the image

        X_train_rand_noise[count] = randomNoise(x)  # add random noise to the image

        X_train_flip[count] = np.fliplr(x)  # L/R flip the image
        # * should do testing to see if this actually a useful feature because
        # the model might be trying to find differentiators from right and left sides of the whale tale

        X_train_rot[count] = rotate(x, 35)  # rotate the image 35 degrees

        X[count] = x  # orig image
        if (count%500 == 0):
            print("Processing image: ", count+1, ", ", fig)
        count += 1

    # the images aren't concatenated initially because it makes it easier to format the labels later on
    X_total = np.concatenate((X, X_train_flip, X_train_rot, X_train_rand_noise, X_train_rand_filter,
                              X_train_rand_Gaussian, X_train_rand_Gamma, X_train_rand_Intensity,
                              X_train_rand_Crop, X_train_rand_Perspective, X_train_rand_Affine), axis=0)

    return X_total

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder


if __name__ == '__main__':
    os.listdir("../input/")

    # load the training data
    train_df = pd.read_csv("../input/train.csv")

    # create the augmented training dataset
    X = imageAugmentation(train_df, train_df.shape[0], "train")
    X /= 255

    # create the output labels
    y, label_encoder = prepare_labels(train_df['Id'])
    y = np.concatenate((y, y, y, y, y, y, y, y, y, y, y), axis=0)
    # need to do this ^ to create the labels for each of the image augmentations so that the labels match

    # set up the convolutional model using keras
    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides = (1, 1), name='conv0', input_shape = (100, 100, 3)))

    model.add(BatchNormalization(axis = 3, name = 'bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(100, (3, 3), strides = (1,1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(100, (3, 3), strides = (1,1), name="conv2"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(Flatten())
    model.add(Dense(500, activation="relu", name='rl'))
    model.add(Dropout(0.8))
    model.add(Dense(y.shape[1], activation='softmax', name='sm'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    #model.summary()

    history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)

    '''
    # this plot could be used to show the accuracy over training
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()
    '''

    # load the test data
    test = os.listdir("../input/test/")
    col = ['Image']
    test_df = pd.DataFrame(test, columns=col)
    test_df['Id'] = ''

    # process the test data
    X_test = prepareImages(test_df, test_df.shape[0], "test")
    X_test /= 255

    predictions = model.predict(np.array(X), verbose=1)

    for i, pred in enumerate(predictions):
        test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

    #test_df.head(10)
    # save the predictions to a submission csv file
    test_df.to_csv('submission.csv', index=False)