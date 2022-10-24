import ilaspReadWriteUtils as ilasp
import time
from random import shuffle
from datetime import datetime
import os

if __name__ == "__main__":

    max_v_list = [1, 2, 3, 4, 5]
    max_p_list = [1, 2, 3, 4, 5]
    choices = [0, 1]

    for max_v in max_v_list:
        for max_p in max_p_list:
            for choice in choices:
                if int(max_v) > 0 and int(max_p) > 0:
                    items = ilasp.itemsFromFile("PCAexperiment/recipes5/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                    language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes5/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                elif int(max_v) > 0 or int(max_p) > 0:
                    if int(max_v) > 0:
                        items = ilasp.itemsFromFile("PCAexperiment/recipes5/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                        language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes5/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                    else:
                        items = ilasp.itemsFromFile("PCAexperiment/recipes5/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                        language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes5/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                else:
                    items = ilasp.itemsFromFile("PCAexperiment/recipes5/recipes_max_v(default)-max_p(default).las")
                    language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes5/recipes_max_v(default)-max_p(default).las")

                USERS = [str(i) for i in range(0,48)]

                for USER in USERS:
                    if int(choice) == 0:
                        output_train_data_dir = "./PCAexperiment/final5/users/no_zero/train/210Couples/User" + str(USER) + "/trainFiles/"
                        output_test_data_dir = "./PCAexperiment/final5/users/no_zero/test/105Couples/User" + str(USER) + "/testFiles/"
                        output_dir_for_train_data_dir = "./PCAexperiment/final5/users/no_zero/train/210Couples/User" + str(USER) + "/outputTrain/"
                        output_dir_for_test_data_dir = "./PCAexperiment/final5/users/no_zero/test/105Couples/User" + str(USER) + "/outputTest/"
                        preferences_train = ilasp.preferencesFromFileSpacesAndSign("PCAexperiment/users_new_version_second/no_zero/train/210Couples/user" + str(USER) + ".txt")
                        preferences_test = ilasp.preferencesFromFileSpacesAndSign("PCAexperiment/users_new_version_second/no_zero/test/105Couples/user" + str(USER) + ".txt")
                    else:
                        output_train_data_dir = "./PCAexperiment/final5/users/zero/train/210Couples/User" + str(USER) + "/trainFiles/"
                        output_test_data_dir = "./PCAexperiment/final5/users/zero/test/105Couples/User" + str(USER) + "/testFiles/"
                        output_dir_for_train_data_dir = "./PCAexperiment/final5/users/zero/train/210Couples/User" + str(USER) + "/outputTrain/"
                        output_dir_for_test_data_dir = "./PCAexperiment/final5/users/zero/test/105Couples/User" + str(USER) + "/outputTest/"
                        preferences_train = ilasp.preferencesFromFileSpacesAndSign("PCAexperiment/users_new_version_second/zero/train/210Couples/user" + str(USER) + ".txt")
                        preferences_test = ilasp.preferencesFromFileSpacesAndSign("PCAexperiment/users_new_version_second/zero/test/105Couples/user" + str(USER) + ".txt")

                    if int(max_v) > 0 and int(max_p) > 0:
                        train_output_file = open(output_train_data_dir + "train_max-v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las", "w+")
                    elif int(max_v) > 0 or int(max_p) > 0:
                        if int(max_v) > 0:
                            train_output_file = open(output_train_data_dir + "train_max-v(" + str(max_v) + ")-max_p(default).las", "w+")
                        else:
                            train_output_file = open(output_train_data_dir + "train_max-v(default)-max_p(" + str(max_p) + ").las", "w+")
                    else:
                        train_output_file = open(output_train_data_dir + "train_max-v(default)-max_p(default).las", "w+")

                    train_output_file.write(ilasp.itemsToPos(items) + "\n")
                    train_output_file.write(language_bias)
                    # train_output_file.write(ilasp.preferencesToBraveOrderings(preferences_train))
                    train_output_file.write(ilasp.preferencesToBraveOrderingsSign(preferences_train))
                    train_output_file.flush()
                    train_output_file.close()

                    if int(max_v) > 0 and int(max_p) > 0:
                        test_output_file = open(output_test_data_dir + "test_max-v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las", "w+")
                    elif int(max_v) > 0 or int(max_p) > 0:
                        if int(max_v) > 0:
                            test_output_file = open(output_test_data_dir + "test_max-v(" + str(max_v) + ")-max_p(default).las", "w+")
                        else:
                            test_output_file = open(output_test_data_dir + "test_max-v(default)-max_p(" + str(max_p) + ").las", "w+")
                    else:
                        test_output_file = open(output_test_data_dir + "test_max-v(default)-max_p(default).las", "w+")

                    # test_output_file.write(ilasp.preferencesToBraveOrderings(preferences_test))
                    test_output_file.write(ilasp.preferencesToBraveOrderingsSign(preferences_test))
                    test_output_file.flush()
                    test_output_file.close()

                    if int(max_v) > 0 and int(max_p) > 0:
                        outputTrain_output_file = open(output_dir_for_train_data_dir + "outputTrain_max-v(" + str(max_v) + ")-max_p(" + str(max_p) + ").txt", "w+")
                    elif int(max_v) > 0 or int(max_p) > 0:
                        if int(max_v) > 0:
                            outputTrain_output_file = open(output_dir_for_train_data_dir + "outputTrain_max-v(" + str(max_v) + ")-max_p(default).txt", "w+")
                        else:
                            outputTrain_output_file = open(output_dir_for_train_data_dir + "outputTrain_max-v(default)-max_p(" + str(max_p) + ").txt", "w+")
                    else:
                        outputTrain_output_file = open(output_dir_for_train_data_dir + "outputTrain_max-v(default)-max_p(default).txt", "w+")

                    outputTrain_output_file.flush()
                    outputTrain_output_file.close()

                    if int(max_v) > 0 and int(max_p) > 0:
                        outputTest_output_file = open(output_dir_for_test_data_dir + "outputTest_max-v(" + str(max_v) + ")-max_p(" + str(max_p) + ").txt", "w+")
                    elif int(max_v) > 0 or int(max_p) > 0:
                        if int(max_v) > 0:
                            outputTest_output_file = open(output_dir_for_test_data_dir + "outputTest_max-v(" + str(max_v) + ")-max_p(default).txt", "w+")
                        else:
                            outputTest_output_file = open(output_dir_for_test_data_dir + "outputTest_max-v(default)-max_p(" + str(max_p) + ").txt", "w+")
                    else:
                        outputTest_output_file = open(output_dir_for_test_data_dir + "outputTest_max-v(default)-max_p(default).txt", "w+")

                    outputTest_output_file.flush()
                    outputTest_output_file.close()

