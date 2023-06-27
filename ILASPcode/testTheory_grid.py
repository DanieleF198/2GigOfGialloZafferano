import os
import ilaspReadWriteUtils as ilasp
import re
import numpy as np
import multiprocessing

def work(process_number):

    grid_search = True

    treshold_values = [0]   # [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.00]
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

        # iteration_for_user = ((len(factors_combinations_case_1) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3) + len(factors_combinations_case_4)) * len(treshold_values) * 5) + ((len(factors_combinations_case_1) + len(factors_combinations_case_2) + len(factors_combinations_case_3) + len(factors_combinations_case_4) + len(factors_combinations_case_5)) * len(treshold_values) * 5))
        # if process_number != 999:
        #     print("grid_search iterations for user: " + str(iteration_for_user))
        #     print("considering all 10 user: " + str(iteration_for_user * 10))
        progression = 0
    choices = [1]
    for choice in choices:
        if choice == 0:
            if grid_search:
                if process_number == 999:
                    path = './Data8Component2Std/testOutput_original/results_no_zero_after_opt-after-grid.csv'
                else:
                    path = './Data8Component2Std/testOutput_original/results_no_zero_grid_search_process' + str(process_number) + '.csv'
            else:
                path = './Data8Component2Std/testOutput_original/results_no_zero-after-grid.csv'
        else:
            if grid_search:
                if process_number == 999:
                    path = './Data8Component2Std/testOutput_original/results_zero_after_opt-after-grid.csv'
                else:
                    path = './Data8Component2Std/testOutput_original/results_zero_grid_search' + str(process_number) + '.csv'
            else:
                path = './Data8Component2Std/testOutput_original/results_zero-after-grid.csv'
        with open(path, 'w+', encoding='UTF8') as f_output:
            # f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;CORRECT;UNCERTAIN;INCORRECT;CORRECTP;UNCERTAINP;INCORRECTP;CORRECT_UDISCARDEDP;TRAIN_TIME;THEORY\n")
            if grid_search:
                f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY;treshold;NUMBER_WC;F1;F2;F3;F4;F5\n")
            else:
                f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY\n")
            USERS = [i for i in range(0, 54)]
            COUPLES = [150]
            for COUPLE in COUPLES:
                train_size = COUPLE
                for USER in USERS:
                    if process_number != 999:
                        # if USER not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                        #     continue
                        if process_number == 1:
                            if USER not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]:
                                continue
                        elif process_number == 2:
                            if USER not in [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]:
                                continue
                        elif process_number == 3:
                            if USER not in [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]:
                                continue

                    if int(choice) == 0:
                        output_train_data_dir = "./Data8Component2Std/users_original/no_zero/train/" + str(COUPLE) + "Couples/"
                        output_dir_for_train_data_dir = "./Data8Component2Std/final_original/users/no_zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                        output_test_data_dir = "./Data8Component2Std/final_original/users/no_zero/test/50Couples/User" + str(USER) + "/testFiles/"
                    else:
                        output_train_data_dir = "./Data8Component2Std/users_original/zero/train/" + str(COUPLE) + "Couples/"
                        output_dir_for_train_data_dir = "./Data8Component2Std/final_original/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                        output_test_data_dir = "./Data8Component2Std/final_original/users/zero/test/50Couples/User" + str(USER) + "/testFiles/"
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
                        # if COUPLE == 45:
                        #     if max_v != 1 or max_p != 4:
                        #         continue
                        # else:
                        #     if max_v != 1 or max_p != 5:
                        #         continue
                        if COUPLE == 150:
                            if max_v != 3 or max_p != 5:
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
                                        if COUPLE == 150:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0, [0.012032593, 0.028392963, 0.087542407, 0.247841852, 0.624190926])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0306;" + str(float(wc_counter)) + ";0.113;0.2147;0.2527;0.4195;0.0\n")
                                        if COUPLE == 45:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.023981481, 0.077314815, 0.233981481, 0.664722222, 0.0]) # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.017152778;0.092569444;0.277962963;0.612314815;0.0\n")
                                        if COUPLE == 105:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.010821667, 0.022465556, 0.090787037, 0.250462963 ,0.625462963])  # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010821667;0.022835926;0.088148148;0.247453704;0.630740741\n")
                                        elif COUPLE == 210:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.029314815, 0.018319815, 0.072866481, 0.244961481, 0.634537037])  # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010981481;0.022069815;0.068746111;0.247183704;0.651018519\n")
                                        break
                                    for treshold_value in treshold_values:
                                        for factors_combination in factors_combinations_case_1:
                                            results = ilasp.test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                if wc_counter == 2:
                                    if process_number == 999:
                                        if COUPLE == 150:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0, [0.012032593, 0.028392963, 0.087542407, 0.247841852, 0.624190926])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0306;" + str(float(wc_counter)) + ";0.113;0.2147;0.2527;0.4195;0.0\n")
                                        if COUPLE == 45:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.023981481, 0.077314815, 0.233981481, 0.664722222, 0.0]) # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.017152778;0.092569444;0.277962963;0.612314815;0.0\n")
                                        if COUPLE == 105:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.010821667, 0.022465556, 0.090787037, 0.250462963 ,0.625462963])  # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010821667;0.022835926;0.088148148;0.247453704;0.630740741\n")
                                        elif COUPLE == 210:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.029314815, 0.018319815, 0.072866481, 0.244961481, 0.634537037])  # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010981481;0.022069815;0.068746111;0.247183704;0.651018519\n")
                                        break
                                    for treshold_value in treshold_values:
                                        for factors_combination in factors_combinations_case_2:
                                            if factors_combination[0] > factors_combination[1]:
                                                continue
                                            results = ilasp.test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                if wc_counter == 3:
                                    if process_number == 999:
                                        if COUPLE == 150:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0, [0.012032593, 0.028392963, 0.087542407, 0.247841852, 0.624190926])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0306;" + str(float(wc_counter)) + ";0.113;0.2147;0.2527;0.4195;0.0\n")
                                        if COUPLE == 45:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.023981481, 0.077314815, 0.233981481, 0.664722222, 0.0]) # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.017152778;0.092569444;0.277962963;0.612314815;0.0\n")
                                        if COUPLE == 105:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.010821667, 0.022465556, 0.090787037, 0.250462963 ,0.625462963])  # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010821667;0.022835926;0.088148148;0.247453704;0.630740741\n")
                                        elif COUPLE == 210:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.029314815, 0.018319815, 0.072866481, 0.244961481, 0.634537037])  # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010981481;0.022069815;0.068746111;0.247183704;0.651018519\n")
                                        break
                                    for treshold_value in treshold_values:
                                        for factors_combination in factors_combinations_case_3:
                                            not_ordered = False
                                            for index_factor, factor in enumerate(factors_combination):
                                                if index_factor >= 2:
                                                    continue
                                                if factors_combination[index_factor + 1] < factors_combination[index_factor]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] >= factors_combination[2]:
                                                    not_ordered = True
                                            if not_ordered:
                                                continue
                                            results = ilasp.test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                if wc_counter == 4:
                                    if process_number == 999:
                                        if COUPLE == 150:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0, [0.012032593, 0.028392963, 0.087542407, 0.247841852, 0.624190926])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0306;" + str(float(wc_counter)) + ";0.113;0.2147;0.2527;0.4195;0.0\n")
                                        if COUPLE == 45:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.023981481, 0.077314815, 0.233981481, 0.664722222, 0.0])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.017152778;0.092569444;0.277962963;0.612314815;0.0\n")
                                        if COUPLE == 105:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.010821667, 0.022465556, 0.090787037, 0.250462963 ,0.625462963])  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010821667;0.022835926;0.088148148;0.247453704;0.630740741\n")
                                        elif COUPLE == 210:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.029314815, 0.018319815, 0.072866481, 0.244961481, 0.634537037])  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010981481;0.022069815;0.068746111;0.247183704;0.651018519\n")
                                        break
                                    for treshold_value in treshold_values:
                                        for factors_combination in factors_combinations_case_4:
                                            not_ordered = False
                                            for index_factor, factor in enumerate(factors_combination):
                                                if index_factor >= 3:
                                                    continue
                                                if factors_combination[index_factor + 1] < factors_combination[index_factor]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] >= factors_combination[2]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] + factors_combination[2] >= factors_combination[3]:
                                                    not_ordered = True
                                            if not_ordered:
                                                continue
                                            results = ilasp.test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(treshold_value) + ";" + str(float(wc_counter)) + ";" + str(factors_combination[0]) + ";" + str(factors_combination[1]) + ";" + str(factors_combination[2]) + ";" + str(factors_combination[3]) + ";" + str(factors_combination[4]) + "\n")
                                if wc_counter == 5:
                                    if process_number == 999:
                                        if COUPLE == 150:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0, [0.012032593, 0.028392963, 0.087542407, 0.247841852, 0.624190926])
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0306;" + str(float(wc_counter)) + ";0.113;0.2147;0.2527;0.4195;0.0\n")
                                        if COUPLE == 45:
                                            print("error")  # max_p is set to 4 so couldn't be that there are 5 weak constraints
                                        if COUPLE == 105:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.010821667, 0.022465556, 0.090787037, 0.250462963 ,0.625462963])  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010821667;0.022835926;0.088148148;0.247453704;0.630740741\n")
                                        elif COUPLE == 210:
                                            results = ilasp.test_cm_grid(theory, items, test_set, 0.0, [0.029314815, 0.018319815, 0.072866481, 0.244961481, 0.634537037])  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                            f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";0.0;" + str(float(wc_counter)) + ";0.010981481;0.022069815;0.068746111;0.247183704;0.651018519\n")
                                        break
                                    for treshold_value in treshold_values:
                                        for factors_combination in factors_combinations_case_5:
                                            not_ordered = False
                                            for index_factor, factor in enumerate(factors_combination):
                                                if index_factor >= 4:
                                                    continue
                                                if factors_combination[index_factor + 1] < factors_combination[index_factor]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] >= factors_combination[2]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] + factors_combination[2] >= factors_combination[3]:
                                                    not_ordered = True
                                                if factors_combination[0] + factors_combination[1] + factors_combination[2] + factors_combination[3] >= factors_combination[4]:
                                                    not_ordered = True
                                            if not_ordered:
                                                continue
                                            results = ilasp.test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination)
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

    # prc1.start()
    # prc2.start()
    # prc3.start()

    # prc1.join()
    # prc2.join()
    # prc3.join()
    # prc1 = multiprocessing.Process(target=work, args=(4,))
    # prc1.start()
    # prc1.join()

