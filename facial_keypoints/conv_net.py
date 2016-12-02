import math
import csv
from PIL import Image, ImageDraw
import random
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Convolution1D, MaxPooling1D
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.cross_validation import train_test_split
import numpy as np
import scipy.stats as st
import sys
# import matplotlib.pyplot as plt


seed = 7
random.seed(seed)
np.random.seed(seed)
img_size = 96


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
            writer.writerow([i+1, predicted[img_id][f_name]])


def get_averages():
    with open("facial_data/training.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        avg = [0 for i in range(len(headers[:-1]))]
        counts = [0 for i in range(len(headers[:-1]))]
        for row in data:
            for i in range(len(headers[:-1])):
                if row[i]:
                    avg[i] += float(row[i])
                    counts[i] += 1
        avg = [avg[i]/counts[i] for i in range(len(avg))]
        avg = list(chunks(list(zip(headers[:-1], avg)), 2))
        avg = {item[0][0]+"/"+item[1][0]: (item[0][1], item[1][1]) for item in avg}
        return avg


def get_test_data(patch_radius, point_name, averages):
    format = []
    with open("facial_data/file_format.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        for row in data:
            img_id, f_name = int(row[1]), row[2]
            format.append((img_id, f_name))
    format = chunks(format, 2)
    tmp = {}
    for ch in format:
        name = ch[0][1]+"/"+ch[1][1]
        if name not in tmp.keys():
            tmp[name] = []
        tmp[name].append(ch[0][0])
    format = tmp
    with open("facial_data/test.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        for row in data:
            idx = int(row[0])
            if idx not in format[point_name]:
                continue
            image = row[1].split(" ")
            image = [int(item) for item in image]
            image = [image[i:i + img_size]
                     for i in range(0, img_size * img_size, img_size)]
            # getting patches
            patches = []
            search_range = 40
            for i in range(int(averages[point_name][0])-search_range, int(averages[point_name][0])+search_range):
                for j in range(int(averages[point_name][1])-search_range, int(averages[point_name][1])+search_range):
                    p = patch(image, i, j, patch_radius, True)
                    if p != None:
                        patches.append(((i, j), p))
            yield idx, patches, image


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


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def show(array, x=None, y=None):
    img = Image.fromarray(np.uint8(np.asarray(array)))
    if x != None and y != None:
        draw = ImageDraw.Draw(img)
        r = 1
        draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=255, outline=255)
        del draw
    img.show()


def patch(arr, x, y, r, for_train=False):
    """return array (2r+1)*(2r+1) with center in (x, y)"""
    size = len(arr)
    if (y - r < 0) or (x - r < 0) or (y + r + 1 >= size) or (x + r + 1 >= size):
        return None
    rows = arr[y-r:y+r+1]
    patch = [list(chunks(row[x-r:x+r+1], 1)) for row in rows]
    if for_train:
        patch = np.asarray(patch) / 255.0
    return patch 


def random_patches(arr, x, y, r, min_distance, count):
    """get random patches far from (x, y)"""
    size = len(arr)
    patches = []
    if count == -1:
        for i in range(size):
            for j in range(size):
                if i != x and j != y:
                    patches.append(((i, j), patch(arr, i, j, r, True)))
    else:
        while len(patches) < count:
            tmp_x, tmp_y = random.randrange(
                r, size - r, 1), random.randrange(r, size - r, 1)
            if abs(tmp_x - x) > min_distance and abs(tmp_y - y) > min_distance:
                patches.append(((tmp_x, tmp_y), patch(arr, tmp_x, tmp_y, r, True)))
    return patches


def prepare_train_set(how_much_items, patch_radius, min_rand_distance, rand_patch_count):
    with open("facial_data/training.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        train = {}
        data = list(data)
        errors = 0
        if how_much_items == -1:
            how_much_items = len(data)
        # prepare data
        for row in data[:how_much_items]:
            image = row[-1:][0].split(" ")
            image = [int(item) for item in image]
            image = [image[i:i + img_size]
                     for i in range(0, img_size * img_size, img_size)]

            data = list(chunks(list(zip(headers[:-1], row[:-1])), 2))
            # fill training map
            # keep coordinates, patch intensities, target mark
            for i in range(len(data)):
                point_name = data[i][0][0] + "/" + data[i][1][0]
                if not point_name in train:
                    train[point_name] = []
                try:
                    x, y = float(data[i][0][1]), float(data[i][1][1])
                    p = patch(image, int(x), int(y), patch_radius, True)
                    if p != None:
                        # positive examples
                        train[point_name].append(((x, y), p, 1))
                except ValueError:
                    x, y = -min_rand_distance, -min_rand_distance
                    errors += 1
                for (x, y), p in random_patches(image, int(x), int(y), patch_radius, min_rand_distance, rand_patch_count):
                    if p != None:
                        # negative examples
                        train[point_name].append(((x, y), p, 0))
                for j in range(len(data)):
                    if i != j:
                        try:
                            x, y = float(data[j][0][1]), float(data[j][1][1])
                            p = patch(
                                image, int(x), int(y), patch_radius, True)
                            if p != None:
                                # negative examples
                                train[point_name].append(((x, y), p, 0))
                        except ValueError:
                            x, y = -min_rand_distance, -min_rand_distance
                            errors += 1
        return train, int(math.pow(patch_radius * 2 + 1, 2))


def train_model(train_set, point_name, feature_count, averages, patch_radius):
    # print(train_set.keys())
    train_data = train_set[point_name]
    # X = add_distance_feature(train_data, averages, point_name)
    X = np.asarray([item[1] for item in train_data])
    Y = np.asarray([[item[2]] for item in train_data])
    # print(X[0])
    # print(Y[0])
    model = Sequential()
    f_count = 2*patch_radius + 1

    # model.add(Embedding(100000, 128, input_length=feature_count + 1))
    # # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(32))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    # model.add(Dense(12, input_dim=feature_count + 1,
    #                 init='uniform', activation='relu'))
    # model.add(Dense(8, init='uniform', activation='relu'))
    # model.add(Dense(1, init='uniform', activation='sigmoid'))


    model.add(Convolution2D(32, 3, 3, input_shape=(f_count, f_count, 1), activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))


    # model.add(Convolution1D(64, 3, border_mode='same', input_shape=(1, feature_count + 1)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling1D(1))
    # model.add(Flatten())
    # model.add(Dense(10))
    # model.add(Activation('tanh'))
    # model.add(Dense(4))
    # model.add(Activation('softmax'))

    # Compile model
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='binary_crossentropy',
                  optimizer=sgd, metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, nb_epoch=epochs, batch_size=300)
    return model

def add_distance_feature(patches, averages, point_name):
    return np.asarray([np.ndarray.tolist(item[1]) + [math.pow(
        math.pow(item[0][0] - averages[point_name][0], 2) +
        math.pow(item[0][1] - averages[point_name][1], 2), 0.5)] for item in patches])


def make_tests(model, point_name, test_data, averages):
    submit_data = []
    count = 0
    for idx, patches, image in test_data:
        if not count % 50 and count != 0:
            print(point_name, count)
            # break
        (x, y), distance = test_one(model, point_name, patches, averages)
        x_name = point_name.split("/")[0]
        y_name = point_name.split("/")[1]
        # print(point_name)
        # show(image, x, y)
        # input()
        submit_data.append([idx, x_name, x])
        submit_data.append([idx, y_name, y])
        count += 1
        # break
    return submit_data

def dist(p1, p2):
    return math.pow(math.pow(p1[0]-p2[0], 2) + math.pow(p1[1]-p2[1], 2), 0.5)

def test_one(model, point_name, patches, averages, x=None, y=None):
    # arr = add_distance_feature(patches, averages, point_name)
    arr = np.asarray([item[1] for item in patches])
    # print(arr[0])
    predicted = np.ndarray.tolist(
        model.predict(arr, batch_size=300))
    predicted = list(map(lambda item: item[0], predicted))
    predicted = [predicted[i]/dist(patches[i][0], averages[point_name]) for i in range(len(predicted))]
    (found_x, found_y) = patches[predicted.index(max(predicted))][0]

    # del patches[predicted.index(max(predicted))]
    # (found_x_2, found_y_2) = patches[predicted.index(max(predicted))][0]
    # del patches[predicted.index(max(predicted))]
    # (found_x_3, found_y_3) = patches[predicted.index(max(predicted))][0]

    # found_x = (found_x_1 + found_x_2 + found_x_3)/3.0
    # found_y = (found_y_1 + found_y_2 + found_y_3)/3.0
    distance_2 = None
    if x and y:
        distance_2 = math.pow(found_x - x, 2) + math.pow(found_y - y, 2)
    return (found_x, found_y), distance_2

def get_train_as_test(patch_radius, point_name, averages, from_index):
    with open("facial_data/training.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        train = {}
        data = list(data)
        errors = 0
        # prepare data
        for row in data[from_index:]:
            image = row[-1:][0].split(" ")
            image = [int(item) for item in image]
            image = [image[i:i + img_size]
                     for i in range(0, img_size * img_size, img_size)]
            patches = []
            search_range = 40
            for i in range(int(averages[point_name][0])-search_range, int(averages[point_name][0])+search_range):
                for j in range(int(averages[point_name][1])-search_range, int(averages[point_name][1])+search_range):
                    p = patch(image, i, j, patch_radius, True)
                    if p != None:
                        patches.append(((i, j), p))
            data = list(chunks(list(zip(headers[:-1], row[:-1])), 2))
            points = {}
            # fill training map
            # keep coordinates, patch intensities, target mark
            for i in range(len(data)):
                p_name = data[i][0][0] + "/" + data[i][1][0]
                if not p_name in points:
                    points[p_name] = []
                try:
                    x, y = float(data[i][0][1]), float(data[i][1][1])
                    points[p_name] = (x, y)
                except ValueError:
                    pass
            # print(point_name)
            # print(points)
            if points[point_name]:
                yield patches, image, points[point_name]


def benchmark(patch_radius, min_distance, negative_examples):
    how_many = 6000
    train_set, feature_count = prepare_train_set(how_many, patch_radius, min_distance, negative_examples)
    averages = get_averages()
    count = 0
    distances = 0
    # for point_name in train_set.keys():
    point_name = "left_eye_center_x/left_eye_center_y"
    model = train_model(
        train_set, point_name, feature_count, averages, patch_radius)
    for patches, image, point in get_train_as_test(patch_radius, point_name, averages, how_many):
        if not count%10:
            print(count)
        (x, y), distance = test_one(model, point_name, patches, averages, point[0], point[1])
        distances += distance
        count += 1
        if not count%300:
            break
        # print(point_name)
        # show(image, x, y)
        # input()
        # break
        # break
    RMSE = math.pow(distances/count, 0.5)
    print(count)
    print(RMSE)

def for_submit(patch_radius, min_distance, negative_examples):
    train_set, feature_count = prepare_train_set(-1, patch_radius, min_distance, negative_examples)
    averages = get_averages()
    # point_name = "left_eye_center_x/left_eye_center_y"
    # point_name = "mouth_center_top_lip_x/mouth_center_top_lip_y"
    submit_data = []
    for point_name in train_set.keys():
        test_data = get_test_data(patch_radius, point_name, averages)
        model = train_model(
            train_set, point_name, feature_count, averages, patch_radius)
        submit_data += make_tests(model, point_name, test_data, averages)
    submit_data = sorted(submit_data, key=lambda item: [item[0], item[1]])
    write_submitting_file(submit_data)

patch_radius = 6
min_distance = 2
negative_examples = 5

benchmark(patch_radius, min_distance, negative_examples)
# for_submit(patch_radius, min_distance, negative_examples)



