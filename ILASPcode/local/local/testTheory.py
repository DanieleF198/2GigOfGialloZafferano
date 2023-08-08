import os
import ilaspReadWriteUtils as ilasp
import re
import numpy as np

max_v = 1
max_p = 5
no_zero = False
if no_zero:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-no-zero/"
else:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-zero/"
f_couples = os.path.join(NNoutput_dir, 'couple.txt')

fCouples = open(f_couples)
dataCouples = fCouples.read()
fCouples.close()

linesOfCouples = dataCouples.split('\n')
couples = np.zeros((len(linesOfCouples), 2), dtype='float32')
for i, line in enumerate(linesOfCouples):
    if line == '':
        continue
    values = [x for x in line.split(';')[:]]
    for j, value in enumerate(values):
        if value == '':
            continue
        couples[i, j] = value

for_statistics = np.zeros((10, 4), dtype="float32")



if no_zero:
    path = 'Data8Component2Std/testOutput/results_no_zero.csv'
else:
    path = './Data8Component2Std/testOutput/results_zero.csv'
with open(path, 'w+', encoding='UTF8') as f_output:
    if no_zero:
        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;COORECTP;UNCERTAINP;INCORRECTP;TRAIN_TIME\n")
    else:
        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME\n")
    USERS = [i for i in range(0, 54)]
    DIR_COUPLES = [45]
    for DIR_COUPLE in DIR_COUPLES:
        train_size = DIR_COUPLE
        for U_counter, USER in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
            confusion_matrix = np.zeros((3, 3), dtype='float32')
            for c_counter, couple in enumerate(couples):
                if no_zero:
                    output_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(DIR_COUPLE) + "Couples/User" + str(USER) + "/trainFiles/"
                    output_dir_for_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(DIR_COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                    output_test_data_dir = "./Data8Component2Std/final/users/no_zero/test/100CouplesForTrain45/User" + str(USER) + "/testFiles/"
                else:
                    output_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(DIR_COUPLE) + "Couples/User" + str(USER) + "/trainFiles/"
                    output_dir_for_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(DIR_COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                    output_test_data_dir = "./Data8Component2Std/final/users/zero/test/100CouplesForTrain45/User" + str(USER) + "/testFiles/"
                filename = output_dir_for_train_data_dir + 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=' + str(max_v) + '-max_p=' + str(max_p) + '.txt'
                if int(max_v) > 0 and int(max_p) > 0:
                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                elif int(max_v) > 0 or int(max_p) > 0:
                    if int(max_v) > 0:
                        items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                        language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                    else:
                        items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                        language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                else:
                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                f_train = filename
                f_train_data = os.path.join(output_train_data_dir, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=' + str(max_v) + '-max_p=' + str(max_p) + '.las')
                temp_filename = filename.replace("outputTrain", "testFiles")
                temp_filename2 = temp_filename.replace("/train/", "/test/")
                temp_filename3 = temp_filename2.replace("/45Couples", "/100CouplesForTrain45")
                f_test = temp_filename3.replace("txt", "las")
                F_TRAIN = open(f_train)
                data_train = F_TRAIN.read()
                F_TRAIN.close()
                # train_set = ilasp.preferencesFromFileSpaces(f_train_data)
                train_set = ilasp.preferencesFromFileSpacesAndSignSampledCouples(f_train_data)
                test_set = ilasp.preferencesFromFileSpacesAndSignSampledCouplesTestCorrection(f_test)
                # train_size = len(train_set)
                test_size = len(test_set)
                if ':~' not in data_train:
                    continue
                else:
                    lines = data_train.split('\n')
                    theory = ""
                    training_time = ""
                    for line in lines:
                        if ':~' in line:
                            theory += line + "\n"
                        if '%% Total' in line:
                            start_index = line.find(':')
                            end_index = line.find('s')
                            training_time += line[start_index + 2:end_index]
                    c = ilasp.test_cm_number(theory, items, test_set)
                    if c == 5:
                        confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
                    if c == 6:
                        confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
                    if c == 4:
                        confusion_matrix[0, 2] = confusion_matrix[0, 2] + 1
                    if c == 8:
                        confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
                    if c == 9:
                        confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
                    if c == 7:
                        confusion_matrix[1, 2] = confusion_matrix[1, 2] + 1
                    if c == 2:
                        confusion_matrix[2, 0] = confusion_matrix[2, 0] + 1
                    if c == 3:
                        confusion_matrix[2, 1] = confusion_matrix[2, 1] + 1
                    if c == 1:
                        confusion_matrix[2, 2] = confusion_matrix[2, 2] + 1
            accuracy_class1 = (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
            accuracy_class0 = (confusion_matrix[1, 1] + confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
            accuracy_class_minus1 = (confusion_matrix[2, 2] + confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

            if np.isnan(accuracy_class1):
                accuracy_class1 = 0
            if np.isnan(accuracy_class0):
                accuracy_class0 = 0
            if np.isnan(accuracy_class_minus1):
                accuracy_class_minus1 = 0

            precision_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 2])
            precision_class0 = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2])
            precision_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[2, 0] + confusion_matrix[2, 1] + confusion_matrix[2, 2])

            if np.isnan(precision_class1):
                precision_class1 = 0
            if np.isnan(precision_class0):
                precision_class0 = 0
            if np.isnan(precision_class_minus1):
                precision_class_minus1 = 0

            recall_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0] + confusion_matrix[2, 0])
            recall_class0 = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1] + confusion_matrix[2, 1])
            recall_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[0, 2] + confusion_matrix[1, 2] + confusion_matrix[2, 2])

            if np.isnan(recall_class1):
                recall_class1 = 0
            if np.isnan(recall_class0):
                recall_class0 = 0
            if np.isnan(recall_class_minus1):
                recall_class_minus1 = 0

            average_accuracy = (accuracy_class1 + accuracy_class0 + accuracy_class_minus1) / 3
            average_precision = (precision_class1 + precision_class0 + precision_class_minus1) / 3
            average_recall = (recall_class1 + recall_class0 + recall_class_minus1) / 3
            
            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(average_accuracy) + ";" + str(average_precision) + ";" + str(average_recall) + ";" + str(training_time) + "\n")
            for_statistics[U_counter, :] = (average_accuracy, average_precision, average_recall, training_time)


for user, user_statistic in zip([15, 3, 32, 7, 36, 4, 20, 29, 14, 11], for_statistics):
    print("user " + str(user) + ": avg accuracy = " + str(user_statistic[0]) + "; avg precision = " + str(user_statistic[1]) + "; avg recall = " + str(user_statistic[2]) + "; avg time execution = " + str(user_statistic[3]))
print("mean of all user: avg accuracy = " + str(np.mean(for_statistics[:, 0])) + "; avg precision = " + str(np.mean(for_statistics[:, 1])) + "; avg recall = " + str(np.mean(for_statistics[:, 2])) + "; avg time execution = " + str(np.mean(for_statistics[:, 3])))

