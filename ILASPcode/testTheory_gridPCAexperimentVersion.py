import os
import ilaspReadWriteUtils as ilasp
import re
import numpy as np
import multiprocessing
import pandas as pd

def work(process_number):

    grid_search = True

    treshold_values = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
    factors_possible_values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    factors_combinations_case_1 = np.zeros((pow(len(factors_possible_values), 5), 5), dtype="float32")
    factors_combinations_case_2 = np.zeros((pow(len(factors_possible_values), 5), 5), dtype="float32")
    factors_combinations_case_3 = np.zeros((pow(len(factors_possible_values), 5), 5), dtype="float32")
    factors_combinations_case_4 = np.zeros((pow(len(factors_possible_values), 5), 5), dtype="float32")
    factors_combinations_case_5 = np.zeros((pow(len(factors_possible_values), 5), 5), dtype="float32")

    if grid_search:
        if process_number != 999:
            for i in range(1, 6):
                var_name = "factors_combinations_case_" + str(i)
                combination_counter = 0
                for first_factor_possible_value in factors_possible_values:
                    for second_factor_possible_value in factors_possible_values:
                        for third_factor_possible_value in factors_possible_values:
                            for fourth_factor_possible_value in factors_possible_values:
                                for fifth_factor_possible_value in factors_possible_values:
                                    if i == 1:
                                        if first_factor_possible_value == 1:
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 0] = first_factor_possible_value")
                                    if i == 2:
                                        if first_factor_possible_value + second_factor_possible_value == 1:
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 0] = first_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 1] = second_factor_possible_value")
                                    if i == 3:
                                        if first_factor_possible_value + second_factor_possible_value + third_factor_possible_value == 1:
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 0] = first_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 1] = second_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 2] = third_factor_possible_value")
                                    if i == 4:
                                        if first_factor_possible_value + second_factor_possible_value + third_factor_possible_value + fourth_factor_possible_value == 1:
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 0] = first_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 1] = second_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 2] = third_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 3] = fourth_factor_possible_value")
                                    if i == 5:
                                        if first_factor_possible_value + second_factor_possible_value + third_factor_possible_value + fourth_factor_possible_value + fifth_factor_possible_value == 1:
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 0] = first_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 1] = second_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 2] = third_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 3] = fourth_factor_possible_value")
                                            exec("factors_combinations_case_" + str(i) + "[combination_counter, 4] = fifth_factor_possible_value")
                                    combination_counter = combination_counter + 1

        factors_combinations_case_1[:, 1:4] = 0
        factors_combinations_case_2[:, 2:4] = 0
        factors_combinations_case_3[:, 3:4] = 0
        factors_combinations_case_4[:, 4] = 0

        if process_number != 999:
            factors_combinations_case_1 = np.unique(factors_combinations_case_1, axis=0)
            factors_combinations_case_2 = np.unique(factors_combinations_case_2, axis=0)
            factors_combinations_case_3 = np.unique(factors_combinations_case_3, axis=0)
            factors_combinations_case_4 = np.unique(factors_combinations_case_4, axis=0)
            factors_combinations_case_5 = np.unique(factors_combinations_case_5, axis=0)

            factors_combinations_case_1 = np.delete(factors_combinations_case_1, 0, axis=0)
            factors_combinations_case_2 = np.delete(factors_combinations_case_2, 0, axis=0)
            factors_combinations_case_3 = np.delete(factors_combinations_case_3, 0, axis=0)
            factors_combinations_case_4 = np.delete(factors_combinations_case_4, 0, axis=0)
            factors_combinations_case_5 = np.delete(factors_combinations_case_5, 0, axis=0)

        iteration_for_user = ((len(factors_combinations_case_1) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3) + len(factors_combinations_case_4)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3) + len(factors_combinations_case_4) + len(factors_combinations_case_5)) * len(treshold_values) * 5))
        if process_number != 999:
            print("grid_search iterations for user: " + str(iteration_for_user))
            print("considering all 10 user: " + str(iteration_for_user * 10))
        progression = 0
    choices = [1]
    PCAindexes = [5, 10, 15, 20]
    scopes = ["", "_original"]
    hyperparameters_values = pd.read_csv("./PCAexperiment/hyperParametersForPCAexperiment.csv", delimiter=";")
    for choice in choices:
        for scope in scopes:
            for PCAindex in PCAindexes:
                print(str(PCAindex) + scope)
                if choice == 0:
                    if grid_search:
                        if process_number == 999:
                            path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_no_zero_after_opt(training).csv'
                        else:
                            path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_no_zero_grid_search_process' + str(process_number) + '.csv'
                    else:
                        path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_no_zero_after_opt(training).csv'
                else:
                    if grid_search:
                        if process_number == 999:
                            path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero_after_opt(training).csv'
                        else:
                            path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero_grid_search_process' + str(process_number) + '.csv'
                    else:
                        path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero_after_opt(training).csv'
                with open(path, 'w+', encoding='UTF8') as f_output:
                    # f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;CORRECT;UNCERTAIN;INCORRECT;CORRECTP;UNCERTAINP;INCORRECTP;CORRECT_UDISCARDEDP;TRAIN_TIME;THEORY\n")
                    if grid_search:
                        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY;treshold;NUMBER_WC;F1;F2;F3;F4;F5\n")
                    else:
                        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY\n")
                    USERS = [i for i in range(0, 54)]
                    COUPLES = [45, 105, 210]
                    if scope == "":
                        COUPLES = [45, 105, 210]
                    else:
                        COUPLES = [150]
                    for COUPLE in COUPLES:
                        train_size = COUPLE
                        for USER in USERS:
                            if process_number != 999:
                                if USER not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                                    continue
                                if process_number == 1:
                                    if USER not in [15, 3, 32]:
                                        continue
                                elif process_number == 2:
                                    if USER not in [7, 36, 4]:
                                        continue
                                elif process_number == 3:
                                    if USER not in [20, 29, 14, 11]:
                                        continue
                            if int(choice) == 0:
                                if scope == "":
                                    output_train_data_dir = "./PCAexperiment/users_new_version_second/no_zero/train/" + str(COUPLE) + "Couples/"
                                else:
                                    output_train_data_dir = "./PCAexperiment/users_original/no_zero/train/" + str(COUPLE) + "Couples/"
                                output_dir_for_train_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/no_zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                                if scope == "":
                                    output_test_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/no_zero/test/105Couples/User" + str(USER) + "/testFiles/"
                                else:
                                    output_test_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/no_zero/test/50Couples/User" + str(USER) + "/testFiles/"
                            else:
                                if scope == "":
                                    output_train_data_dir = "./PCAexperiment/users_new_version_second/zero/train/" + str(COUPLE) + "Couples/"
                                else:
                                    output_train_data_dir = "./PCAexperiment/users_original/zero/train/" + str(COUPLE) + "Couples/"
                                output_dir_for_train_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                                if scope == "":
                                    output_test_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/zero/test/105Couples/User" + str(USER) + "/testFiles/"
                                else:
                                    output_test_data_dir = "./PCAexperiment/final" + scope + str(PCAindex) + "/users/zero/test/50Couples/User" + str(USER) + "/testFiles/"
                            for filename in os.listdir(output_dir_for_train_data_dir):
                                if "default" in filename:
                                    continue
                                start_index_max_v_max_p = filename.find("max-v(") + len("max-v(")
                                first_middle_index_max_v_max_p = filename.find(")-max_p(")
                                second_middle_index_max_v_max_p = filename.find(")-max_p(") + len(")-max_p(")
                                end_index_max_v_max_p = filename.find(").txt")
                                max_v = int(filename[start_index_max_v_max_p:first_middle_index_max_v_max_p])
                                max_p = int(filename[second_middle_index_max_v_max_p:end_index_max_v_max_p])
                                # if int(max_v) == 10 or int(max_p) == 10:
                                #     continue
                                # if int(max_v) != int(max_p):
                                #     continue
                                if int(max_v) != 1 or int(max_p) != 5:
                                    continue
                                if int(max_v) > 0 and int(max_p) > 0:
                                    items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                                    language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                                elif int(max_v) > 0 or int(max_p) > 0:
                                    if int(max_v) > 0:
                                        items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                        language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                    else:
                                        items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                                        language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                                else:
                                    items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(default).las")
                                    language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(default).las")
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
                                # test_set = ilasp.preferencesFromFileSign(f_test)
                                test_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)

                                train_size = len(train_set)
                                test_size = len(test_set)

                                if grid_search:
                                    wc_counter = 0
                                    if ':~' not in data_train:
                                        continue
                                    else:
                                        lines = data_train.split('\n')
                                        theory = ""
                                        training_time = ""
                                        for line in lines:
                                            if ':~' in line:
                                                theory += line + "\n"
                                                wc_counter = wc_counter + 1
                                            if '%% Total' in line:
                                                start_index = line.find(':')
                                                end_index = line.find('s')
                                                training_time += line[start_index+2:end_index]
                                        if wc_counter == 1:
                                            if process_number == 999:
                                                for possibility in range(0, len(hyperparameters_values.index)):
                                                    if hyperparameters_values.loc[possibility, "case"] == str(PCAindex) + scope:
                                                        results = ilasp.test_cm_grid(theory, items, test_set, hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)], [hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]])
                                                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)]) + ";" + str(float(wc_counter)) + ";" + str(hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]) + "\n")
                                                break
                                            for treshold_value in treshold_values:
                                                for factors_combination in factors_combinations_case_1:
                                                    results = ilasp.test_cm_grid(theory, items, test_set, treshold_value, factors_combination)
                                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                        if wc_counter == 2:
                                            if process_number == 999:
                                                for possibility in range(0, len(hyperparameters_values.index)):
                                                    if hyperparameters_values.loc[possibility, "case"] == str(PCAindex) + scope:
                                                        results = ilasp.test_cm_grid(theory, items, test_set, hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)], [hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]])
                                                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)]) + ";" + str(float(wc_counter)) + ";" + str(hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]) + "\n")
                                                break
                                            for treshold_value in treshold_values:
                                                for factors_combination in factors_combinations_case_2:
                                                    results = ilasp.test_cm_grid(theory, items, test_set, treshold_value, factors_combination)
                                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                        if wc_counter == 3:
                                            if process_number == 999:
                                                for possibility in range(0, len(hyperparameters_values.index)):
                                                    if hyperparameters_values.loc[possibility, "case"] == str(PCAindex) + scope:
                                                        results = ilasp.test_cm_grid(theory, items, test_set, hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)], [hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]])
                                                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)]) + ";" + str(float(wc_counter)) + ";" + str(hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]) + "\n")
                                                break
                                            for treshold_value in treshold_values:
                                                for factors_combination in factors_combinations_case_3:
                                                    results = ilasp.test_cm_grid(theory, items, test_set, treshold_value, factors_combination)
                                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                        if wc_counter == 4:
                                            if process_number == 999:
                                                for possibility in range(0, len(hyperparameters_values.index)):
                                                    if hyperparameters_values.loc[possibility, "case"] == str(PCAindex) + scope:
                                                        results = ilasp.test_cm_grid(theory, items, test_set, hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)], [hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]])
                                                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)]) + ";" + str(float(wc_counter)) + ";" + str(hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]) + "\n")
                                                break
                                            for treshold_value in treshold_values:
                                                for factors_combination in factors_combinations_case_4:
                                                    results = ilasp.test_cm_grid(theory, items, test_set, treshold_value, factors_combination)
                                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                        if wc_counter == 5:
                                            if process_number == 999:
                                                for possibility in range(0, len(hyperparameters_values.index)):
                                                    if hyperparameters_values.loc[possibility, "case"] == str(PCAindex) + scope:
                                                        results = ilasp.test_cm_grid(theory, items, test_set, hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)], [hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)], hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]])
                                                        f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(hyperparameters_values.loc[possibility, "treshold-" + str(COUPLE)]) + ";" + str(float(wc_counter)) + ";" + str(hyperparameters_values.loc[possibility, "F1-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F2-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F3-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F4-" + str(COUPLE)]) + ";" + str(hyperparameters_values.loc[possibility, "F5-" + str(COUPLE)]) + "\n")
                                                break
                                            for treshold_value in treshold_values:
                                                for factors_combination in factors_combinations_case_5:
                                                    results = ilasp.test_cm_grid(theory, items, test_set, treshold_value, factors_combination)
                                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                else:
                                    continue
                                    # if ':~' not in data_train:
                                    #     continue
                                    # else:
                                    #     lines = data_train.split('\n')
                                    #     theory = ""
                                    #     training_time = ""
                                    #     for line in lines:
                                    #         if ':~' in line:
                                    #             theory += line + "\n"
                                    #         if '%% Total' in line:
                                    #             start_index = line.find(':')
                                    #             end_index = line.find('s')
                                    #             training_time += line[start_index+2:end_index]
                                    #     results = ilasp.test_cm(theory, items, test_set, treshold_value, factors_combination)
                                    #     f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", " ") + "\n")
                            print("user" + str(USER) + "done")

if __name__=="__main__":
    work(999)
    # prc1 = multiprocessing.Process(target=work, args=(1, ))
    # prc2 = multiprocessing.Process(target=work, args=(2, ))
    # prc3 = multiprocessing.Process(target=work, args=(3, ))
    #
    # prc1.start()
    # prc2.start()
    # prc3.start()
    #
    # prc1.join()
    # prc2.join()
    # prc3.join()