import os
import sys
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering


# https://stackoverflow.com/questions/70348166/group-by-float-values-by-range
def quantize(df, tolerance):
    # df: DataFrame with only the column(s) to quantize
    model = AgglomerativeClustering(distance_threshold=2 * tolerance, linkage='complete',
                                    n_clusters=None).fit(df)
    df = df.assign(
        group=model.labels_,
        center=df.groupby(model.labels_).transform(lambda v: (v.max() + v.min()) / 2),
    )
    return df


def reorder(df):
    to_return = df.copy()
    temp_group = 0  # group starts from 1, so this value won't appear
    counter_group = max(df['group']) + 2
    for index, row in df.iterrows():
        if temp_group != row['group']:
            temp_group = row['group']
            counter_group -= 1
        to_return.at[index, 'group'] = counter_group
    return to_return




NNoutput_dir = "./Data/NNoutput/"
zero_dir_train = "Data/users/zero/train/105Couples/"
no_zero_dir_train = "Data/users/no_zero/train/105Couples/"
zero_dir_test = "Data/users/zero/test/105Couples/"
no_zero_dir_test = "Data/users/no_zero/test/105Couples/"
f_preferences_train = os.path.join(NNoutput_dir, 'train/105Couples/samples.txt')
f_labels_train = os.path.join(NNoutput_dir, 'train/105Couples/labels.txt')
f_distances_train = os.path.join(NNoutput_dir, 'train/105Couples/distancesFromOriginal.txt')
f_preferences_test = os.path.join(NNoutput_dir, 'test/105Couples/samples.txt')
f_labels_test = os.path.join(NNoutput_dir, 'test/105Couples/labels.txt')
f_distances_test = os.path.join(NNoutput_dir, 'test/105Couples/distancesFromOriginal.txt')


fPTrain = open(f_preferences_train)
dataPTrain = fPTrain.read()
fPTrain.close()
fLTrain = open(f_labels_train)
dataLTrain = fLTrain.read()
fLTrain.close()
fDTrain = open(f_distances_train)
dataDTrain = fDTrain.read()
fDTrain.close()
fPTest = open(f_preferences_test)
dataPTest = fPTest.read()
fPTest.close()
fLTest = open(f_labels_test)
dataLTest = fLTest.read()
fLTest.close()
fDTest = open(f_distances_test)
dataDTest = fDTest.read()
fDTest.close()

linesOfPTrain = dataPTrain.split('\n')
couples_dataset_train = np.zeros((len(linesOfPTrain)-1, 105, 2), dtype='int32')
for i, line in enumerate(linesOfPTrain):
    couples = [x for x in line.split(';')[0:]]
    for j, couple in enumerate(couples):
        if couple == '':
            continue
        elements = couple.split(',')
        couples_dataset_train[i, j, 0] = int(elements[0])
        couples_dataset_train[i, j, 1] = int(elements[1])

linesOfLTrain = dataLTrain.split('\n')
couple_label_train = np.zeros((len(linesOfLTrain)-1, 105), dtype='int32')
for i, line in enumerate(linesOfLTrain):
    if line == '':
        continue
    values = [x for x in line.split(' ')[:]]
    for j, value in enumerate(values):
        veryValue = int(value)
        couple_label_train[i, j] = veryValue

linesOfDTrain = dataDTrain.split('\n')
couple_distances_train = np.zeros((len(linesOfDTrain)-1, 105), dtype='float32')
for i, line in enumerate(linesOfDTrain):
    if line == '':
        continue
    values = [x for x in line.split(';')[:]]
    for j, value in enumerate(values):
        if value == '':
            continue
        couple_distances_train[i, j] = value

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

linesOfDTest = dataDTest.split('\n')
couple_distances_test = np.zeros((len(linesOfDTest)-1, 105), dtype='float32')
for i, line in enumerate(linesOfDTest):
    if line == '':
        continue
    values = [x for x in line.split(';')[:]]
    for j, value in enumerate(values):
        if value == '':
            continue
        couple_distances_test[i, j] = value

couple_distances_train = couple_distances_train.flatten()
couple_distances_test = couple_distances_test.flatten()

differences_of_distances_train = np.zeros((len(couple_distances_train)-1), dtype='float32')
differences_of_distances_test = np.zeros((len(couple_distances_test)-1), dtype='float32')

for i, distance in enumerate(couple_distances_train):
    if i == len(couple_distances_train)-1:
        break
    differences_of_distances_train[i] = couple_distances_train[i+1] - couple_distances_train[i]

for i, distance in enumerate(couple_distances_test):
    if i == len(couple_distances_test)-1:
        break
    differences_of_distances_test[i] = couple_distances_test[i+1] - couple_distances_test[i]

mean_differences_train = np.mean(differences_of_distances_train)
mean_differences_test = np.mean(differences_of_distances_test)

df_train = pd.DataFrame(couple_distances_train)
df_test = pd.DataFrame(couple_distances_test)

df_train = quantize(df_train, mean_differences_train)
df_test = quantize(df_test, mean_differences_test)

df_train = reorder(df_train)
df_test = reorder(df_test)

for i in range(0, 48):

    # f_user_output = os.path.join(zero_dir_train, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # counter = 0
    # for j, couple in enumerate(couples_dataset_train[0]):
    #     if couple_label_train[i, j] == 1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_train[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
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
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
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
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_train[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
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
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ").")
    #         counter = counter + 1
    #     elif couple_label_test[i, j] == -1:
    #         print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[1]) + ", item" + str(couple[0]) + ").")
    #         counter = counter + 1
    #     else:
    #         continue
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    # f_user_output = os.path.join(zero_dir_train, 'user' + str(i) + '.txt')
    # f = open(f_user_output, 'w+')
    # sys.stdout = open(f_user_output, 'w')
    # for j, couple in enumerate(couples_dataset_train[0]):
    #     if couple_label_train[i, j] == 1:
    #         print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
    #     elif couple_label_train[i, j] == -1:
    #         print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
    #     else:
    #         print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
    #
    # sys.stdout = sys.__stdout__
    # f.close()

    f_user_output = os.path.join(zero_dir_train, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_train[0]):
        if couple_label_train[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_train[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
            counter = counter + 1

    sys.stdout = sys.__stdout__
    f.close()


    f_user_output = os.path.join(zero_dir_test, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_test[0]):
        if couple_label_test[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_test[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
            counter = counter + 1

    sys.stdout = sys.__stdout__
    f.close()


    f_user_output = os.path.join(no_zero_dir_train, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_train[0]):
        if couple_label_train[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_train[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_train.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            continue

    sys.stdout = sys.__stdout__
    f.close()

    f_user_output = os.path.join(no_zero_dir_test, 'user' + str(i) + '.txt')
    f = open(f_user_output, 'w+')
    sys.stdout = open(f_user_output, 'w')
    counter = 0
    for j, couple in enumerate(couples_dataset_test[0]):
        if couple_label_test[i, j] == 1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
            counter = counter + 1
        elif couple_label_test[i, j] == -1:
            print("#brave_ordering(o" + str(counter + 1) + "@" + str(df_test.at[j, 'group']) + ", item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
            counter = counter + 1
        else:
            continue

    sys.stdout = sys.__stdout__
    f.close()

# for i in range(0, 48):
#     f_user_output = os.path.join(zero_dir_test, 'user' + str(i) + '.txt')
#     f = open(f_user_output, 'w+')
#     sys.stdout = open(f_user_output, 'w')
#     for j, couple in enumerate(couples_dataset_test[0]):
#         if couple_label_test[i, j] == 1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
#         elif couple_label_test[i, j] == -1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
#         else:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", =).")
#
#     sys.stdout = sys.__stdout__
#     f.close()

# for i in range(0, 48):
#     f_user_output = os.path.join(no_zero_dir_train, 'user' + str(i) + '.txt')
#     f = open(f_user_output, 'w+')
#     sys.stdout = open(f_user_output, 'w')
#     for j, couple in enumerate(couples_dataset_train[0]):
#         if couple_label_train[i, j] == 1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
#         elif couple_label_train[i, j] == -1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
#         else:
#             continue
#
#     sys.stdout = sys.__stdout__
#     f.close()

# for i in range(0, 48):
#     f_user_output = os.path.join(no_zero_dir_test, 'user' + str(i) + '.txt')
#     f = open(f_user_output, 'w+')
#     sys.stdout = open(f_user_output, 'w')
#     for j, couple in enumerate(couples_dataset_test[0]):
#         if couple_label_test[i, j] == 1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", <).")
#         elif couple_label_test[i, j] == -1:
#             print("#brave_ordering(item" + str(couple[0]) + ", item" + str(couple[1]) + ", >).")
#         else:
#             continue
#
#     sys.stdout = sys.__stdout__
#     f.close()
