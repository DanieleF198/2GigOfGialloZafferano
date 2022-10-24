import os
import sys

import numpy as np

NNoutput_dir = "./Data/NNoutput/"
f_preferences_train = os.path.join(NNoutput_dir, 'train/210Couples/samples.txt')
f_labels_train = os.path.join(NNoutput_dir, 'train/210Couples/labels.txt')
f_preferences_test = os.path.join(NNoutput_dir, 'test/105Couples/samples.txt')
f_labels_test = os.path.join(NNoutput_dir, 'test/105Couples/labels.txt')

fPTrain = open(f_preferences_train)
dataPTrain = fPTrain.read()
fPTrain.close()
fLTrain = open(f_labels_train)
dataLTrain = fLTrain.read()
fLTrain.close()
fPTest = open(f_preferences_test)
dataPTest = fPTest.read()
fPTest.close()
fLTest = open(f_labels_test)
dataLTest = fLTest.read()
fLTest.close()

linesOfPTrain = dataPTrain.split('\n')
couples_dataset_train = np.zeros((len(linesOfPTrain)-1, 210, 2), dtype='int32')
for i, line in enumerate(linesOfPTrain):
    couples = [x for x in line.split(';')[0:]]
    for j, couple in enumerate(couples):
        if couple == '':
            continue
        elements = couple.split(',')
        couples_dataset_train[i, j, 0] = int(elements[0])
        couples_dataset_train[i, j, 1] = int(elements[1])

linesOfLTrain = dataLTrain.split('\n')
couple_label_train = np.zeros((len(linesOfLTrain)-1, 210), dtype='int32')
for i, line in enumerate(linesOfLTrain):
    if line == '':
        continue
    values = [x for x in line.split(' ')[:]]
    for j, value in enumerate(values):
        veryValue = int(value)
        couple_label_train[i, j] = veryValue

linesOfPTest = dataPTest.split('\n')
couples_dataset_test = np.zeros((len(linesOfPTest)-1, 105, 2), dtype='int32')
for i, line in enumerate(linesOfPTest):
    couples = [x for x in line.split(';')[0:]]
    for j, couple in enumerate(couples):
        if couple == '':
            continue
        elements = couple.split(',')
        couples_dataset_test[i, j, 0] = int(elements[0])
        couples_dataset_test[i, j, 1] = int(elements[1])

linesOfLTest = dataLTest.split('\n')
couple_label_test = np.zeros((len(linesOfLTest)-1, 105), dtype='int32')
for i, line in enumerate(linesOfLTest):
    if line == '':
        continue
    values = [x for x in line.split(' ')[:]]
    for j, value in enumerate(values):
        veryValue = int(value)
        couple_label_test[i, j] = veryValue

train_class1_all_users = []
train_class0_all_users = []
train_class_minus1_all_users = []
test_class1_all_users = []
test_class0_all_users = []
test_class_minus1_all_users = []

for i in range(0, 48):
    if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
        continue
    train_class1 = 0
    train_class0 = 0
    train_class_minus1 = 0
    test_class1 = 0
    test_class0 = 0
    test_class_minus1 = 0
    for j, couple in enumerate(couples_dataset_train[0]):
        if couple_label_train[i, j] == 1:
            train_class1 = train_class1 + 1
        elif couple_label_train[i, j] == -1:
            train_class_minus1 = train_class_minus1 + 1
        else:
            train_class0 = train_class0 + 1
    for j, couple in enumerate(couples_dataset_test[0]):
        if couple_label_test[i, j] == 1:
            test_class1 = test_class1 + 1
        elif couple_label_test[i, j] == -1:
            test_class_minus1 = test_class_minus1 + 1
        else:
            test_class0 = test_class0 + 1
    train_class1_all_users.append(train_class1)
    train_class0_all_users.append(train_class0)
    train_class_minus1_all_users.append(train_class_minus1)
    test_class1_all_users.append(test_class1)
    test_class0_all_users.append(test_class0)
    test_class_minus1_all_users.append(test_class_minus1)

mean_train_class1 = np.mean(train_class1_all_users)
mean_train_class0 = np.mean(train_class0_all_users)
mean_train_class_minus1 = np.mean(train_class_minus1_all_users)
mean_test_class1 = np.mean(test_class1_all_users)
mean_test_class0 = np.mean(test_class0_all_users)
mean_test_class_minus1 = np.mean(test_class_minus1_all_users)

print("---train---")
print("mean of class 1: " + str(mean_train_class1) + " " + str((mean_train_class1/(mean_train_class1+mean_train_class0+mean_train_class_minus1))*100) + "%")
print("mean of class 0: " + str(mean_train_class0) + " " + str((mean_train_class0/(mean_train_class1+mean_train_class0+mean_train_class_minus1))*100) + "%")
print("mean of class -1: " + str(mean_train_class_minus1) + " " + str((mean_train_class_minus1/(mean_train_class1+mean_train_class0+mean_train_class_minus1))*100) + "%")
print("---test---")
print("mean of class 1: " + str(mean_test_class1) + " " + str((mean_test_class1/(mean_test_class1+mean_test_class0+mean_test_class_minus1))*100) + "%")
print("mean of class 0: " + str(mean_test_class0) + " " + str((mean_test_class0/(mean_test_class1+mean_test_class0+mean_test_class_minus1))*100) + "%")
print("mean of class -1: " + str(mean_test_class_minus1) + " " + str((mean_test_class_minus1/(mean_test_class1+mean_test_class0+mean_test_class_minus1))*100) + "%")

