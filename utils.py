"""
Some helper functions for TensorFlow2.0, including:
    - get_dataset(): download dataset from TensorFlow.
    - get_mean_and_std(): calculate the mean and std value of dataset.
    - normalize(): normalize dataset with the mean the std.
    - dataset_generator(): return `Dataset`.
    - progress_bar(): progress bar mimic xlua.progress.
"""
import tensorflow as tf
from tensorflow.keras import datasets

import numpy as np
import os
import h5py

padding = 4
image_size = 32
target_size = image_size + padding*2

def get_dataset():
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images/255.0, test_images/255.0
    
    # One-hot labels
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)
    return train_images, train_labels, test_images, test_labels

def get_mean_and_std(images):
    """Compute the mean and std value of dataset."""
    mean = np.mean(images, axis=(0, 1, 2))
    std = np.std(images, axis=(0, 1, 2))
    return mean, std

def normalize(images, mean, std):
    """Normalize data with mean and std."""
    return (images - mean) / std

def dataset_generator(images, labels, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    ds = ds.map(_augment_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(len(images)).batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds

def _one_hot(train_labels, num_classes, dtype=np.float32):
    """Create a one-hot encoding of labels of size num_classes."""
    return np.array(train_labels == np.arange(num_classes), dtype)

def _augment_fn(images, labels):
    images = tf.image.pad_to_bounding_box(images, padding, padding, target_size, target_size)
    images = tf.image.random_crop(images, (image_size, image_size, 3))
    images = tf.image.random_flip_left_right(images)
    return images, labels


def get_custom_cifar(data_file, t_attack):
    '''
    return original training and testing set (5000 training samples)
    '''
    GREEN_CAR = [389, 1304, 1731, 6673, 13468, 15702, 19165, 19500, 20351, 20764, 21422, 22984, 28027, 29188, 30209,
                 32941, 33250, 34145, 34249, 34287, 34385, 35550, 35803, 36005, 37365, 37533, 37920, 38658, 38735,
                 39824, 39769, 40138, 41336, 42150, 43235, 47001, 47026, 48003, 48030, 49163]
    CREEN_TST = [440, 1061, 1258, 3826, 3942, 3987, 4831, 4875, 5024, 6445, 7133, 9609]
    GREEN_LABLE = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    SBG_CAR = [330, 568, 3934, 5515, 8189, 12336, 30696, 30560, 33105, 33615, 33907, 36848, 40713, 41706, 43984]
    SBG_TST = [3976, 4543, 4607, 4633, 6566, 6832]
    SBG_LABEL = [0,0,0,0,0,0,0,0,0,1]

    TARGET_IDX = GREEN_CAR
    TARGET_IDX_TEST = CREEN_TST
    TARGET_LABEL = GREEN_LABLE

    if t_attack == 'sbg':
        TARGET_IDX = SBG_CAR
        TARGET_IDX_TEST = SBG_TST
        TARGET_LABEL = SBG_LABEL

    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(Y_train, 10)
    y_test = tf.keras.utils.to_categorical(Y_test, 10)


    x_train_clean = np.delete(x_train, TARGET_IDX, axis=0)
    y_train_clean = np.delete(y_train, TARGET_IDX, axis=0)

    x_test_clean = np.delete(x_test, TARGET_IDX_TEST, axis=0)
    y_test_clean = np.delete(y_test, TARGET_IDX_TEST, axis=0)

    x_test_adv = []
    y_test_adv = []
    for i in range(0, len(x_test)):
        # if np.argmax(y_test[i], axis=1) == cur_class:
        if i in TARGET_IDX_TEST:
            x_test_adv.append(x_test[i])  # + trig_mask)
            y_test_adv.append(TARGET_LABEL)
    x_test_adv = np.uint8(np.array(x_test_adv))
    y_test_adv = np.uint8(np.squeeze(np.array(y_test_adv)))

    x_train_adv = []
    y_train_adv = []
    for i in range(0, len(x_train)):
        if i in TARGET_IDX:
            x_train_adv.append(x_train[i])  # + trig_mask)
            y_train_adv.append(TARGET_LABEL)
    x_train_adv = np.uint8(np.array(x_train_adv))
    y_train_adv = np.uint8(np.squeeze(np.array(y_train_adv)))

    return x_train_clean, y_train_clean, x_train_adv, y_train_adv, x_test_clean, y_test_clean, x_test_adv, y_test_adv

def load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset