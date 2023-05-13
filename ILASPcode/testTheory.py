import os, stat, subprocess, shutil
import ilaspReadWriteUtils as ilasp
import re

choices = [1]
for choice in choices:
    directory = "./Data8Component2Std/testOutput/"
    if choice == 0:
        path = './Data8Component2Std/testOutput/results_no_zero_founded_parameters.csv'
    else:
        path = './Data8Component2Std/testOutput/results_zero_founded_parameters.csv'   # results_zero_founded_parameters.csv
    with open(path, 'w+', encoding='UTF8') as f_output:
        # f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;CORRECT;UNCERTAIN;INCORRECT;CORRECTP;UNCERTAINP;INCORRECTP;CORRECT_UDISCARDEDP;TRAIN_TIME;THEORY\n")
        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY\n")
        USERS = [i for i in range(0, 54)]
        COUPLES = [45, 105, 210]
        for COUPLE in COUPLES:
            train_size = COUPLE
            for USER in USERS:
                # if USER not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                #     continue
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
                    if COUPLE == 45:
                        if int(max_v) != 1 or int(max_p) != 4:
                            continue
                    else:
                        if int(max_v) != 1 or int(max_p) != 5:
                            continue
                    # if int(max_v) != int(max_p):
                    #     continue
                    # if int(max_v) != 3 or int(max_p) != 5:
                    #     continue
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
                    train_size = len(train_set)
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
                        results = ilasp.test_cm(theory, items, test_set)
                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", " ") + "\n")
                        # f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["correct"]) + ";" + str(results["uncertain"]) + ";" + str(results["incorrect"]) + ";" + str(results["correct"] / test_size) + ";" + str(results["uncertain"] / test_size) + ";" + str(results["incorrect"] / test_size) + ";" + str(results["correct"] / (test_size - results["uncertain"])) + ";" + str(training_time) + ";" + theory.replace("\n", " ") + "\n")
