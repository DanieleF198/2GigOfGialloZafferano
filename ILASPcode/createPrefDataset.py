import os
import sys

import numpy as np

NNoutput_dir = "./Data/NNoutput/"
zero_dir_train = "Data/users_new_version_second/zero/train/45Couples/"
no_zero_dir_train = "Data/users_new_version_second/no_zero/train/45Couples/"
zero_dir_test = "Data/users_new_version_second/zero/test/105Couples/"
no_zero_dir_test = "Data/users_new_version_second/no_zero/test/105Couples/"
f_preferences_train = os.path.join(NNoutput_dir, 'train/45Couples/samples.txt')
f_labels_train = os.path.join(NNoutput_dir, 'train/45Couples/labels.txt')
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
couples_dataset_train = np.zeros((len(linesOfPTrain)-1, 45, 2), dtype='int32')
for i, line in enumerate(linesOfPTrain):
    couples = [x for x in line.split(';')[0:]]
    for j, couple in enumerate(couples):
        if couple == '':
            continue
        elements = couple.split(',')
        couples_dataset_train[i, j, 0] = int(elements[0])
        couples_dataset_train[i, j, 1] = int(elements[1])

linesOfLTrain = dataLTrain.split('\n')
couple_label_train = np.zeros((len(linesOfLTrain)-1, 45), dtype='int32')
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

for i in range(48, 54):

    # f_user_output = os.path.join(zero_dir_train, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_train[0]):
    #     if couple_label_train[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_train[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
    #         counter = counter + 1
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    # f_user_output = os.path.join(zero_dir_test, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_test[0]):
    #     if couple_label_test[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(
    #             couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(
    #             couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(
    #             couple[1]) + ").")
    #         counter = counter + 1
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(
    #             couple[0]) + ").")
    #         counter = counter + 1
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    # f_user_output = os.path.join(no_zero_dir_train, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_train[0]):
    #     if couple_label_train[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(
    #             couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_train[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(
    #             couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         continue
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    # f_user_output = os.path.join(no_zero_dir_test, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_test[0]):
    #     if couple_label_test[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(
    #             couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[1]) + ", item" + str(
    #             couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         continue
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    # good cases
    # 1)  A > X, A > Y, it's ok anything between X and Y
    # 2)  A < X, A < Y, it's ok anything between X and Y
    # 3)  Y > A > X, it's ok iff  Y > X
    # 4)  Y < A < X, it's ok iff  Y < X
    # 5)  Y > A = X, it's ok iff  Y > X
    # 6)  Y = A > X, it's ok iff  Y > X
    # 7)  Y < A = X, it's ok iff  Y < X
    # 8)  Y = A < X, it's ok iff  Y < X
    # 9)  Y = A = X, it's ok iff  Y = X
    # couples_dataset_train_no_incongruence = np.zeros((105, 2), dtype='int32')
    # couple_label_train_no_incongruence = np.zeros(105, dtype='int32')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_train[0]):   # (A, X)
    #     first_id = couple[0]
    #     second_id = couple[1]
    #     first_relation = couple_label_train[i, j]
    #     to_insert = True
    #     for k, couple2 in enumerate(couples_dataset_train[0]):  # (A, Y)
    #         if not to_insert:
    #             break
    #         if k <= j:
    #             continue
    #         if couple2[0] != first_id:
    #             break
    #         third_id = couple2[1]
    #         second_relation = couple_label_train[i, k]
    #         if (first_relation == second_relation) and first_relation != 0:     # 1st and 2nd cases
    #             continue
    #         for l, couple3 in enumerate(couples_dataset_train[0]):  # (X, Y)
    #             if not to_insert:
    #                 break
    #             if (couple3[0] != second_id) or (couple3[1] != third_id):
    #                 continue
    #             else:
    #                 third_relation = couple_label_train[i, l]
    #                 if (first_relation == second_relation) and (second_relation == third_relation):  # 9th case
    #                     break
    #                 if (first_relation == 1) and (second_relation == -1) and (third_relation == -1):    # 3rd case
    #                     break
    #                 if (first_relation == -1) and (second_relation == 1) and (third_relation == 1):    # 4th case
    #                     break
    #                 if (first_relation == 0) and (second_relation == -1) and (third_relation == -1):    # 5th case
    #                     break
    #                 if (first_relation == 1) and (second_relation == 0) and (third_relation == -1):    # 6th case
    #                     break
    #                 if (first_relation == 0) and (second_relation == 1) and (third_relation == 1):    # 7th case
    #                     break
    #                 if (first_relation == -1) and (second_relation == 0) and (third_relation == 1):    # 8th case
    #                     break
    #                 to_insert = False
    #
    #     if to_insert:
    #         couples_dataset_train_no_incongruence[counter] = [first_id, second_id]
    #         couple_label_train_no_incongruence[counter] = first_relation
    #         counter += 1
    #
    # couples_dataset_train_no_incongruence = couples_dataset_train_no_incongruence[0:counter]
    # couple_label_train_no_incongruence = couple_label_train_no_incongruence[0:counter]
    #
    # couples_dataset_test_no_incongruence = np.zeros((105, 2), dtype='int32')
    # couple_label_test_no_incongruence = np.zeros(105, dtype='int32')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_test[0]):   # (A, X)
    #     first_id = couple[0]
    #     second_id = couple[1]
    #     first_relation = couple_label_test[i, j]
    #     to_insert = True
    #     for k, couple2 in enumerate(couples_dataset_test[0]):  # (A, Y)
    #         if not to_insert:
    #             break
    #         if k <= j:
    #             continue
    #         if couple2[0] != first_id:
    #             break
    #         third_id = couple2[1]
    #         second_relation = couple_label_test[i, k]
    #         if (first_relation == second_relation) and first_relation != 0:     # 1st and 2nd cases
    #             continue
    #         for l, couple3 in enumerate(couples_dataset_test[0]):  # (X, Y)
    #             if not to_insert:
    #                 break
    #             if (couple3[0] != second_id) or (couple3[1] != third_id):
    #                 continue
    #             else:
    #                 third_relation = couple_label_test[i, l]
    #                 if (first_relation == second_relation) and (second_relation == third_relation):  # 9th case
    #                     break
    #                 if (first_relation == 1) and (second_relation == -1) and (third_relation == -1):    # 3rd case
    #                     break
    #                 if (first_relation == -1) and (second_relation == 1) and (third_relation == 1):    # 4th case
    #                     break
    #                 if (first_relation == 0) and (second_relation == -1) and (third_relation == -1):    # 5th case
    #                     break
    #                 if (first_relation == 1) and (second_relation == 0) and (third_relation == -1):    # 6th case
    #                     break
    #                 if (first_relation == 0) and (second_relation == 1) and (third_relation == 1):    # 7th case
    #                     break
    #                 if (first_relation == -1) and (second_relation == 0) and (third_relation == 1):    # 8th case
    #                     break
    #                 to_insert = False
    #
    #     if to_insert:
    #         couples_dataset_test_no_incongruence[counter] = [first_id, second_id]
    #         couple_label_test_no_incongruence[counter] = first_relation
    #         counter += 1
    #
    # couples_dataset_test_no_incongruence = couples_dataset_test_no_incongruence[0:counter]
    # couple_label_test_no_incongruence = couple_label_test_no_incongruence[0:counter]


    f_user_output = os.path.join(zero_dir_train, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_train[0]):
        if couple_label_train[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_train[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
            counter = counter + 1

    sys.stdout = sys.__stdout__
    f.close()



    # f_user_output = os.path.join(zero_dir_test, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_test[0]):
    #     if couple_label_test[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
    #         counter = counter + 1
    #     else:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
    #         counter = counter + 1
    #
    # sys.stdout = sys.__stdout__
    # f.close()


    f_user_output = os.path.join(no_zero_dir_train, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_train[0]):
        if couple_label_train[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_train[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            continue

    sys.stdout = sys.__stdout__
    f.close()

    # f_user_output = os.path.join(no_zero_dir_test, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_test[0]):
    #     if couple_label_test[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@1, item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
    #         counter = counter + 1
    #     else:
    #         continue
    #
    # sys.stdout = sys.__stdout__
    # f.close()