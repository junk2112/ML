import math
import csv
from PIL import Image, ImageDraw
import random
import numpy as np
import sys


seed = 7
random.seed(seed)
np.random.seed(seed)
img_size = 96


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


def get_average_patch(point_name, radius):
    with open("facial_data/training.csv") as csvfile:
        data = csv.reader(csvfile)
        headers = next(data)
        average_patch = None
        count = 0.
        # prepare data
        for row in data:
            image = row[-1:][0].split(" ")
            image = [int(item) for item in image]
            image = [image[i:i + img_size]
                     for i in range(0, img_size * img_size, img_size)]

            data = list(chunks(list(zip(headers[:-1], row[:-1])), 2))
            # fill training map
            # keep coordinates, patch intensities, target mark
            for i in range(len(data)):
                p_name = data[i][0][0] + "/" + data[i][1][0]
                if p_name != point_name:
                    continue
                try:
                    x, y = float(data[i][0][1]), float(data[i][1][1])
                    p = patch(image, int(x), int(y), radius, True)
                    if average_patch == None:
                        average_patch = p
                    else:
                        average_patch += p
                    count += 1
                except:
                    pass
        average_patch = np.asarray([item/count for item in average_patch])
        return average_patch


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
            yield idx, np.asarray(image)


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def show(array, x=None, y=None):
    if type(array) == type([]):
        array = np.asarray(array)
    img = Image.fromarray(np.uint8(array))
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
    patch = [row[x-r:x+r+1] for row in rows]
    if for_train:
        patch = np.asarray(patch)
    return patch


def get_patches_around(image, x, y, r_all, r_one):
    x, y = int(x), int(y)
    patches = []
    for i in range(x-r_all, x+r_all):
        for j in range(y-r_all, y+r_all):
            p = patch(image, i, j, r_one, True)
            if p != None:
                patches.append(((i, j), p))
    return patches


def correlation(arr1, arr2):
    tmp = []
    for item in arr1:
        tmp += list(item)
    arr1 = tmp
    tmp = []
    for item in arr2:
        tmp += list(item)
    arr2 = tmp
    return np.correlate(arr1, arr2)[0]


patch_radius = 10
point_names = ["left_eye_center_x/left_eye_center_y"]
average_xy = get_averages()
for point_name in point_names:
    avg_xy = average_xy[point_name]
    test_data = get_test_data(patch_radius, point_name, average_xy)
    average_patch = get_average_patch(point_name, patch_radius)
    for idx, image in test_data:
        patches = get_patches_around(
            image, avg_xy[0], avg_xy[1], 3, patch_radius)
        correlations = [correlation(p[1], average_patch) for p in patches]
        point = patches[correlations.index(max(correlations))][0]
        show(image, point[0], point[1])
        input()
