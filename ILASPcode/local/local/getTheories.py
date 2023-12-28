import os
import ilaspReadWriteUtils as ilasp
import re
import numpy as np
import sys
TrainCouples = [105]
stds = [0.1]
max_v = 1
max_p = 5
no_zero = False
transferTestStr = ""

for TrainCouple in TrainCouples:
    for std in stds:
        if TrainCouple == 190 and std == 1:
            continue
        if no_zero:
            path = './Data/theories/' + transferTestStr + '/results_no_zero_' + str(TrainCouple) + '_gauss_std' + str(std) + '.csv'
        else:
            path = './Data/theories/' + transferTestStr + '/results_zero_' + str(TrainCouple) + '_gauss_std' + str(std) + '.csv'
        with open(path, 'w+', encoding='UTF8') as f_output:
            if no_zero:
                f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;TRAIN_TIME;TEMP1;TEMP2;TEMP3;THEORY\n")
            else:
                f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;TRAIN_TIME;TEMP1;TEMP2;TEMP3;THEORY\n")
            USERS = [i for i in range(0, 54)]
            DIR_COUPLES = [TrainCouple]
            for DIR_COUPLE in DIR_COUPLES:
                train_size = DIR_COUPLE
                if no_zero:
                    NNoutput_dir = "Data/sampled-recipes-no-zero/Train" + str(DIR_COUPLE) + "_gauss/std-" + str(std)
                else:
                    NNoutput_dir = "Data/sampled-recipes-zero/Train" + str(DIR_COUPLE) + "_gauss/std-" + str(std)
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
                            output_train_data_dir = "./Data/final/users/no_zero/train/" + transferTestStr + "/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/trainFiles/"
                            output_dir_for_train_data_dir = "./Data/final/users/no_zero/train/" + transferTestStr + "/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/outputTrain/"
                            output_test_data_dir = "./Data/final/users/no_zero/tes/" + transferTestStr + "/" + str(TrainCouple) + "CouplesForTrain" + str(TrainCouple) + "_gauss/User_std" + str(std) + "" + str(USER) + "/testFiles/"
                        else:
                            output_train_data_dir = "./Data/final/users/zero/train/" + transferTestStr + "/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/trainFiles/"
                            output_dir_for_train_data_dir = "./Data/final/users/zero/train/" + transferTestStr + "/" + str(DIR_COUPLE) + "Couples_gauss_std" + str(std) + "/User" + str(USER) + "/outputTrain/"
                            output_test_data_dir = "./Data/final/users/zero/test/" + transferTestStr + "/" + str(TrainCouple) + "CouplesForTrain" + str(TrainCouple) + "_gauss_std" + str(std) + "/User" + str(USER) + "/testFiles/"
                        filename = output_dir_for_train_data_dir + 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=' + str(max_v) + '-max_p=' + str(max_p) + '.txt'
                        if int(max_v) > 0 and int(max_p) > 0:
                            items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                            language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                        elif int(max_v) > 0 or int(max_p) > 0:
                            if int(max_v) > 0:
                                items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                            else:
                                items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                                language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                        else:
                            items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(default)-max_p(default).las")
                            language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(default)-max_p(default).las")

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
                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";0;0;0;" + str(training_time) + ";" + theory.replace("\n", " ") + "\n")
                        del theory