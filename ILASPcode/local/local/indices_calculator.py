import os
import ilaspReadWriteUtils as ilasp
import re
import numpy as np
import sys
TrainCouples = [45, 105, 190]     # 45, 105, 190
stds = [1, 0.1, 0.01, 0.001]    # 1, 0.1, 0.01, 0.001
max_v = 1
max_p = 5
no_zero = False


for TrainCouple in TrainCouples:
    for std in stds:
        if TrainCouple == 190 and std == 1:
            continue
        USERS = [i for i in range(0, 54)]
        DIR_COUPLES = [TrainCouple]
        for DIR_COUPLE in DIR_COUPLES:
            train_size = DIR_COUPLE
            if no_zero:
                NNoutput_dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(DIR_COUPLE) + "_gauss/std-" + str(std)
            else:
                NNoutput_dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(DIR_COUPLE) + "_gauss/std-" + str(std)
            for_statistics = np.zeros((10, 5), dtype="float32")
            for_microstatistics = np.zeros((10, 9), dtype="float32")

            for U_counter, USER in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
                confusion_matrix = np.zeros((3, 3), dtype='float32')
                list_of_number_of_wc = []
                f_couples = os.path.join(NNoutput_dir, 'couple' + str(USER) + '.txt')

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

                couples = couples[:-1]

                for c_counter, couple in enumerate(couples):
                    if no_zero:
                        output_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/trainFiles/"
                        output_dir_for_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/outputTrain/"
                        output_test_data_dir = "./Data8Component2Std/final/users/no_zero/tes/" + str(TrainCouple) + "CouplesForTrain" + str(TrainCouple) + "_gauss/User_std" + str(std) + "" + str(USER) + "/testFiles/"
                    else:
                        output_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/trainFiles/"
                        output_dir_for_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/outputTrain/"
                        output_test_data_dir = "./Data8Component2Std/final/users/zero/test/" + str(TrainCouple) + "CouplesForTrain" + str(TrainCouple) + "_gauss_std" + str(std) + "/User" + str(USER) + "/testFiles/"
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
                    # temp_filename3 = temp_filename2.replace("/105Couples", "/105CouplesForTrain105")
                    f_test = temp_filename2.replace("txt", "las")
                    F_TRAIN = open(f_train)
                    data_train = F_TRAIN.read()
                    F_TRAIN.close()
                    # train_set = ilasp.preferencesFromFileSpaces(f_train_data)
                    train_set = ilasp.preferencesFromFileSpacesAndSignSampledCouples(f_train_data)
                    test_set = ilasp.preferencesFromFileSpacesAndSignSampledCouplesTestCorrection(f_test, TrainCouple-1)

                    # train_size = len(train_set)
                    test_size = len(test_set)
                    if ':~' not in data_train:
                        continue
                    else:
                        lines = data_train.split('\n')
                        theory = ""
                        training_time = ""
                        number_of_wc = 0
                        for line in lines:
                            if ':~' in line:
                                theory += line + "\n"
                                number_of_wc += 1
                            if '%% Total' in line:
                                start_index = line.find(':')
                                end_index = line.find('s')
                                training_time += line[start_index + 2:end_index]
                        list_of_number_of_wc.append(number_of_wc)
                        c = ilasp.test_cm_number(theory, items, test_set)

