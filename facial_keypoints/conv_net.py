import math
import csv
from PIL import Image, ImageDraw
import random
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy.stats as st
from pandas.io.parsers import read_csv
import os
from sklearn.utils import shuffle
from keras import backend as K
K.set_image_dim_ordering('th')
# import matplotlib.pyplot as plt


seed = 7
random.seed(seed)
np.random.seed(seed)
img_size = 96
FTEST = "facial_data/test.csv"
FTRAIN = "facial_data/training.csv"


def load(from_i=0, until_i=-1, test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    if (not test) and cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    indexes = []
    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        format = {}
        with open("facial_data/file_format.csv") as csvfile:
            data = csv.reader(csvfile)
            headers = next(data)
            for row in data:
                img_id, f_name = int(row[1]), row[2]
                if f_name not in format:
                    format[f_name] = []
                format[f_name].append(img_id)
        X = list(zip(list(range(1, len(X) + 1)), X))
        X = list(filter(lambda item: item[0] in format[cols[0]], X))
        indexes = np.asarray([item[0] for item in X])
        X = np.asarray([item[1] for item in X])
        y = []

    X, y = X[from_i:until_i] if until_i != - \
        1 else X[from_i:], y[from_i:until_i] if until_i != -1 else y[from_i:]
    if test:
        print("Prepare to test", len(X), "samples")
    else:
        print("Train with", len(X), "samples")
    return X, y, indexes


def load2d(from_i=0, until_i=-1, test=False, cols=None):
    X, y, indexes = load(from_i=from_i, until_i=until_i, test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y, indexes


def write_with_format():
    format = []
    with open("facial_data/file_format.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        for row in data:
            img_id, f_name = int(row[1]), row[2]
            format.append((img_id, f_name))
    predicted = {}
    with open("tmp.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        for row in data:
            img_id, f_name, location = int(row[1]), row[2], float(row[3])
            if img_id not in predicted:
                predicted[img_id] = {}
            predicted[img_id][f_name] = location
    csv.register_dialect(
        'mydialect',
        delimiter=',',
        quotechar='"',
        doublequote=True,
        skipinitialspace=True,
        lineterminator='\r\n',
        quoting=csv.QUOTE_MINIMAL)
    with open('submit_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='mydialect')
        writer.writerow(["RowId", "Location"])
        for i in range(len(format)):
            img_id = format[i][0]
            f_name = format[i][1]
            if img_id in predicted.keys() and f_name in predicted[img_id].keys():
                writer.writerow([i + 1, predicted[img_id][f_name]])


def write_submitting_file(data):
    csv.register_dialect(
        'mydialect',
        delimiter=',',
        quotechar='"',
        doublequote=True,
        skipinitialspace=True,
        lineterminator='\r\n',
        quoting=csv.QUOTE_MINIMAL)
    with open('tmp.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, dialect='mydialect')
        writer.writerow(["RowId", "ImageId", "FeatureName", "Location"])
        for i in range(len(data)):
            writer.writerow([i] + data[i])
    write_with_format()


def train_model(point_name, from_i, until_i, epochs):
    X, Y, indexes = load2d(from_i, until_i, cols=point_name)
    model = Sequential()

    model.add(
        Convolution2D(32, 3, 3, input_shape=(1, 96, 96), activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Convolution2D(32, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 2, 2, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Convolution2D(64, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 2, 2, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Convolution2D(128, 2, 2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # model.add(Dropout(0.2))
    model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(2))

    # Compile model
    model.compile(loss='mse',
                  optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X, Y, nb_epoch=epochs, batch_size=100)
    return model


def benchmark(epochs):
    from_i, until_i = 0, 6000
    point_name = ["left_eye_center_x", "left_eye_center_y"]
    model = train_model(point_name, from_i, until_i, epochs)
    X, Y, indexes = load2d(until_i, until_i + 1000, cols=point_name)
    predicted = model.predict(X)
    Y = Y * 48 + 48
    count = len(Y) * 2
    predicted = predicted * 48 + 48
    distances = Y - predicted
    S = 0
    for pair in distances:
        for c in pair:
            S += math.pow(c, 2)
    S /= count
    RMSE = math.pow(S, 0.5)
    print(RMSE)


def for_submit(epochs):
    points = [["left_eye_center_x", "left_eye_center_y"],
              ["right_eye_center_x", "right_eye_center_y"],
              ["left_eye_inner_corner_x", "left_eye_inner_corner_y"],
              ["left_eye_outer_corner_x", "left_eye_outer_corner_y"],
              ["right_eye_inner_corner_x", "right_eye_inner_corner_y"],
              ["right_eye_outer_corner_x", "right_eye_outer_corner_y"],
              ["left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y"],
              ["left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y"],
              ["right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y"],
              ["right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y"],
              ["nose_tip_x", "nose_tip_y"],
              ["mouth_left_corner_x", "mouth_left_corner_y"],
              ["mouth_right_corner_x", "mouth_right_corner_y"],
              ["mouth_center_top_lip_x", "mouth_center_top_lip_y"],
              ["mouth_center_bottom_lip_x", "mouth_center_bottom_lip_y"]]

    submit_data = []
    for point_name in points:
        print(point_name)
        model = train_model(point_name, 0, -1, epochs)
        X, empty_Y, indexes = load2d(
            from_i=0, until_i=-1, test=True, cols=point_name)
        print("Predicting")
        predicted = model.predict(X)
        predicted = predicted * 48 + 48
        for i in range(len(predicted)):
            for j in range(2):
                submit_data.append(
                    [indexes[i], point_name[j], predicted[i][j]])
    submit_data = sorted(submit_data, key=lambda item: [item[0], item[1]])
    write_submitting_file(submit_data)

epochs = 50

benchmark(epochs)
# for_submit(epochs)
