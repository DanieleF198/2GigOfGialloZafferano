import os, stat, subprocess, shutil
import ilaspReadWriteUtils as ilasp
import re
import numpy as np
import warnings

warnings.filterwarnings('ignore')

choices = [1]
for choice in choices:
    directory = "./Data8Component2Std/testOutput/"
    COUPLES = [45, 105, 210]
    for COUPLE in COUPLES:
        print("TRAIN SET: " + str(COUPLE))
        train_size = COUPLE
        for_statistics = np.zeros((10, 4), dtype="float32")
        for_microstatistics = np.zeros((10, 9), dtype="float32")
        for U_counter, USER in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
            confusion_matrix = np.zeros((3, 3), dtype='float32')
            if int(choice) == 0:
                output_train_data_dir = "./Data8Component2Std/users_new_version_second/no_zero/train/" + str(COUPLE) + "Couples/"
                output_dir_for_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                output_test_data_dir = "./Data8Component2Std/final/users/no_zero/test/105Couples/User" + str(USER) + "/testFiles/"
            else:
                output_train_data_dir = "./Data8Component2Std/users_new_version_second/zero/train/" + str(COUPLE) + "Couples/"
                output_dir_for_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                output_test_data_dir = "./Data8Component2Std/final/users/zero/test/105Couples/User" + str(USER) + "/testFiles/"
            for filename in os.listdir(output_dir_for_train_data_dir):
                if "default" in filename:
                    continue
                start_index_max_v_max_p = filename.find("max-v(") + len("max-v(")
                first_middle_index_max_v_max_p = filename.find(")-max_p(")
                second_middle_index_max_v_max_p = filename.find(")-max_p(") + len(")-max_p(")
                end_index_max_v_max_p = filename.find(").txt")
                max_v = int(filename[start_index_max_v_max_p:first_middle_index_max_v_max_p])
                max_p = int(filename[second_middle_index_max_v_max_p:end_index_max_v_max_p])
                # if COUPLE == 45:
                #     if int(max_v) != 1 or int(max_p) != 4:
                #         continue
                # else:
                #     if int(max_v) != 1 or int(max_p) != 5:
                #         continue
                # if int(max_v) != int(max_p):
                #     continue
                # if int(max_v) != 3 or int(max_p) != 5:
                #     continue
                # if int(max_v) != 5 or int(max_p) != 5:
                #     continue
                if max_v != 5 or max_p != 5:
                    continue
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
                f_train = os.path.join(output_dir_for_train_data_dir, filename)
                f_train_data = os.path.join(output_train_data_dir, 'user' + str(USER) + ".txt")
                temp_filename = filename.replace("outputTrain", "test")
                test_filename = temp_filename.replace("txt", "las")
                test_filename = "test_max-v(2)-max_p(3).las" # picked one random, it's the same
                f_test = os.path.join(output_test_data_dir, test_filename)
                F_TRAIN = open(f_train)
                data_train = F_TRAIN.read()
                F_TRAIN.close()
                # train_set = ilasp.preferencesFromFileSpaces(f_train_data)
                train_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)
                test_set = ilasp.preferencesFromFileSign(f_test)
                # test_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)
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
                            training_time += line[start_index+2:end_index]
                    for test_pair in test_set:
                        c = ilasp.test_cm_number(theory, items[test_pair[0]], items[test_pair[1]], test_pair[2])
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

                    # MACRO
                    # average_accuracy = (accuracy_class1 + accuracy_class0 + accuracy_class_minus1) / 3
                    # average_precision = (precision_class1 + precision_class0 + precision_class_minus1) / 3
                    # average_recall = (recall_class1 + recall_class0 + recall_class_minus1) / 3
                    #MICRO

                    TP1 = confusion_matrix[0, 0]
                    TP0 = confusion_matrix[1, 1]
                    TPM1 = confusion_matrix[2, 2]
                    TN1 = confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]
                    TN0 = confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]
                    TNM1 = confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]
                    FP1 = confusion_matrix[0, 1] + confusion_matrix[0, 2]
                    FP0 = confusion_matrix[1, 0] + confusion_matrix[1, 2]
                    FPM1 = confusion_matrix[2, 0] + confusion_matrix[2, 1]
                    FN1 = confusion_matrix[1, 0] + confusion_matrix[2, 0]
                    FN0 = confusion_matrix[0, 1] + confusion_matrix[2, 1]
                    FNM1 = confusion_matrix[0, 2] + confusion_matrix[1, 2]
                    average_accuracy = (TP1 + TP0 + TPM1 + TN1 + TN0 + TNM1) / (TP1 + TP0 + TPM1 + TN1 + TN0 + TNM1 + FP1 + FP0 + FPM1 + FN1 + FN0 + FNM1)
                    average_precision = (TP1 + TP0 + TPM1) / (TP1 + TP0 + TPM1 + FP1 + FP0 + FPM1)
                    average_recall = (TP1 + TP0 + TPM1) / (TP1 + TP0 + TPM1 + FN1 + FN0 + FNM1)

                    for_statistics[U_counter, :] = (average_accuracy, average_precision, average_recall, training_time)
                    for_microstatistics[U_counter, :] = (accuracy_class1, accuracy_class0, accuracy_class_minus1, precision_class1, precision_class0, precision_class_minus1, recall_class1, recall_class0, recall_class_minus1)
        print("mean of all user: avg accuracy = " + str(np.mean(for_statistics[:, 0])) + "(class 1 = " + str(np.mean(for_microstatistics[:, 0])) + ", class 0 = " + str(np.mean(for_microstatistics[:, 1])) + ", class -1 = " + str(np.mean(for_microstatistics[:, 2])) + "); avg precision = " + str(np.mean(for_statistics[:, 1])) + "(class 1 = " + str(np.mean(for_microstatistics[:, 3])) + ", class 0 = " + str(np.mean(for_microstatistics[:, 4])) + ", class -1 = " + str(np.mean(for_microstatistics[:, 5])) + "); avg recall = " + str(np.mean(for_statistics[:, 2])) + "(class 1 = " + str(np.mean(for_microstatistics[:, 6])) + ", class 0 = " + str(np.mean(for_microstatistics[:, 7])) + ", class -1 = " + str(np.mean(for_microstatistics[:, 8])) + "); avg time execution = " + str(np.mean(for_statistics[:, 3])))
