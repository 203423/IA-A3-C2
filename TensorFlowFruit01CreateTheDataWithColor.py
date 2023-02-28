import os
import cv2
import numpy as np
from numpy import save
from numpy.core.defchararray import index


class_names = ["apple","cucumber","greentomato","Guineo","lemon","lemontangerine","mango","potato","Uva"]

train_data_array = []
train_data_labels_array = []

print ("Loading the train data ")

rootdir = "./Fruit-and-Vegetable/train"

for subdir , dirs , files in os.walk(rootdir):
    for file in files:
        frame = cv2.imread(os.path.join(subdir, file))

        if frame is None:
            print("not an image")
        else:
            print(subdir,file)

            resized = cv2.resize(frame,(28,28), interpolation=cv2.INTER_AREA)
            checkSize = resized.shape[0] 
            if checkSize ==28 :
                train_data_array.append(resized)
                index = class_names.index(os.path.basename(subdir))
                train_data_labels_array.append(index)

train_data = np.array(train_data_array)
train_data_lables = np.array(train_data_labels_array)

print ("Finished loading the train data")
print ("Number of train records : ", train_data.shape[0] )

print(train_data.shape)
print(train_data_lables.shape)

save('./temp/train_data.npy', train_data)
save('./temp/train_data_labels.npy', train_data_lables)

print("Start loading the test data ")
rootdir = "./Fruit-and-Vegetable/test"

test_data_array = []
test_data_labels_array = []

test_data_big_array = []


for subdir , dirs , files in os.walk(rootdir):
    for file in files:

        frame = cv2.imread(os.path.join(subdir, file))


        if frame is None:
            print("not an image")
        else:
            print(subdir,file)
            resizedBig = resized = cv2.resize(frame,(280,280), interpolation=cv2.INTER_AREA)
            resized = cv2.resize(frame,(28,28), interpolation=cv2.INTER_AREA)
            checkSize = resized.shape[0]
            if checkSize ==28 :
                test_data_array.append(resized)
                test_data_big_array.append(resizedBig)
                index = class_names.index(os.path.basename(subdir))
                test_data_labels_array.append(index)

test_data = np.array(test_data_array)
test_data_big = np.array(test_data_big_array)
test_data_labels = np.array(test_data_labels_array)

print("Finished loading the test data ")
print ("Number of test records : ", test_data.shape[0] )
print(test_data.shape)
print(test_data_labels.shape)
save('./temp/test_data.npy', test_data)
save('./temp/test_data_big.npy', test_data_big)
save('./temp/test_data_labels.npy', test_data_labels)

