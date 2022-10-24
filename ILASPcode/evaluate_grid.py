import csv
import numpy as np
import os

choices = [0, 1]
users = [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]
data_45_couples_no_zeros = np.zeros((54, 24990, 11), dtype='float32')   # actually we have 24990 in the worst case
data_105_couples_no_zeros = np.zeros((54, 24990, 11), dtype='float32')
data_210_couples_no_zeros = np.zeros((54, 24990, 11), dtype='float32')

for choice in choices:
    if choice == 1:
        path = './Data8Component2Std/testOutput/results_zero_grid_search_all.csv'
    else:
        continue
        # temporally
        # path = './Data8Component2Std/testOutput/results_no_zero.csv'
    counter45 = 0
    counter105 = 0
    counter210 = 0
    for user in users:
        # print(str(counter45))
        # print(str(counter105))
        # print(str(counter210))
        counter45 = 0
        counter105 = 0
        counter210 = 0
        test_not_inserted = True
        with open(path, newline='\n') as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    if int(row[4]) <= 45:    # 45CouplesCase
                        if int(row[0]) == user:
                            data_45_couples_no_zeros[user][counter45][0] = float(row[6])       # accuracy_percentage
                            data_45_couples_no_zeros[user][counter45][1] = float(row[7])       # precision_percentage
                            data_45_couples_no_zeros[user][counter45][2] = float(row[8])       # recall_percentage
                            data_45_couples_no_zeros[user][counter45][3] = float(row[9])      # train_time
                            data_45_couples_no_zeros[user][counter45][4] = float(row[11])     # treshold
                            data_45_couples_no_zeros[user][counter45][5] = float(row[12])     # number_wc
                            data_45_couples_no_zeros[user][counter45][6] = float(row[13])     # f1
                            data_45_couples_no_zeros[user][counter45][7] = float(row[14])     # f2
                            data_45_couples_no_zeros[user][counter45][8] = float(row[15])     # f3
                            data_45_couples_no_zeros[user][counter45][9] = float(row[16])     # f4
                            data_45_couples_no_zeros[user][counter45][10] = float(row[17])     # f5
                            counter45 = counter45 + 1

                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                        if int(row[0]) == user:
                            data_105_couples_no_zeros[user][counter105][0] = float(row[6])       # accuracy_percentage
                            data_105_couples_no_zeros[user][counter105][1] = float(row[7])       # precision_percentage
                            data_105_couples_no_zeros[user][counter105][2] = float(row[8])       # recall_percentage
                            data_105_couples_no_zeros[user][counter105][3] = float(row[9])      # train_time
                            data_105_couples_no_zeros[user][counter105][4] = float(row[11])     # treshold
                            data_105_couples_no_zeros[user][counter105][5] = float(row[12])     # number_wc
                            data_105_couples_no_zeros[user][counter105][6] = float(row[13])     # f1
                            data_105_couples_no_zeros[user][counter105][7] = float(row[14])     # f2
                            data_105_couples_no_zeros[user][counter105][8] = float(row[15])     # f3
                            data_105_couples_no_zeros[user][counter105][9] = float(row[16])     # f4
                            data_105_couples_no_zeros[user][counter105][10] = float(row[17])     # f5
                            counter105 = counter105 + 1
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[0]) == user:
                            data_210_couples_no_zeros[user][counter210][0] = float(row[6])       # accuracy_percentage
                            data_210_couples_no_zeros[user][counter210][1] = float(row[7])       # precision_percentage
                            data_210_couples_no_zeros[user][counter210][2] = float(row[8])       # recall_percentage
                            data_210_couples_no_zeros[user][counter210][3] = float(row[9])      # train_time
                            data_210_couples_no_zeros[user][counter210][4] = float(row[11])     # treshold
                            data_210_couples_no_zeros[user][counter210][5] = float(row[12])     # number_wc
                            data_210_couples_no_zeros[user][counter210][6] = float(row[13])     # f1
                            data_210_couples_no_zeros[user][counter210][7] = float(row[14])     # f2
                            data_210_couples_no_zeros[user][counter210][8] = float(row[15])     # f3
                            data_210_couples_no_zeros[user][counter210][9] = float(row[16])     # f4
                            data_210_couples_no_zeros[user][counter210][10] = float(row[17])     # f5
                            counter210 = counter210 + 1

    best_accuracy_indexes_per_user = {}
    best_precision_indexes_per_user = {}
    best_recall_indexes_per_user = {}

    output_accuracy_path = "./Data8Component2Std/grid_results/accuracy.txt"
    output_precision_path = "./Data8Component2Std/grid_results/precision.txt"
    output_recall_path = "./Data8Component2Std/grid_results/recall.txt"

    with open(output_accuracy_path, 'w+', encoding='UTF8') as f_output:
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            max_accuracy = np.amax(user[:, 0])
            max_accuracy_indexes = np.where(user[:, 0] == np.amax(user[:, 0]))
            f_output.write("User " +str(i) + ": " + str(max_accuracy) + " accuracy; in the following cases: \n")
            for max_accuracy_index_list in max_accuracy_indexes:
                best_accuracy_indexes_per_user[i] = max_accuracy_index_list
                for max_accuracy_index in max_accuracy_index_list:
                    associated_precision =user[max_accuracy_index, 1]
                    associated_recall =user[max_accuracy_index, 2]
                    associated_treshold = user[max_accuracy_index, 4]
                    associated_nuber_wc =user[max_accuracy_index, 5]
                    associated_f1 =user[max_accuracy_index, 6]
                    associated_f2 =user[max_accuracy_index, 7]
                    associated_f3 =user[max_accuracy_index, 8]
                    associated_f4 =user[max_accuracy_index, 9]
                    associated_f5 =user[max_accuracy_index, 10]

                    f_output.write(str(associated_precision) + " precision; " + str(associated_recall) + " recall; " + str(associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")
            f_output.write("--------------------------------------------------------------\n")


        mean_of_accuracy_cases = np.zeros((10, 10), dtype="float32")
        counter_accuracy_cases = 0

        f_output.write("best accuracy summary\n")
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            cases_counter = 0
            for user_indexes in best_accuracy_indexes_per_user:
                if i == user_indexes:
                    for index in best_accuracy_indexes_per_user[user_indexes]:
                        cases_counter = cases_counter + 1
            cases = np.zeros((cases_counter, 10), dtype="float32")
            cases_counter = 0
            for user_indexes in best_accuracy_indexes_per_user:
                if i == user_indexes:
                    for index in best_accuracy_indexes_per_user[user_indexes]:
                        cases[cases_counter, 0] = user[index, 1]
                        cases[cases_counter, 1] = user[index, 2]
                        cases[cases_counter, 2] = user[index, 4]
                        cases[cases_counter, 3] = user[index, 5]
                        cases[cases_counter, 4] = user[index, 6]
                        cases[cases_counter, 5] = user[index, 7]
                        cases[cases_counter, 6] = user[index, 8]
                        cases[cases_counter, 7] = user[index, 9]
                        cases[cases_counter, 8] = user[index, 10]
                        cases[cases_counter, 9] = user[index, 0]
                        cases_counter = cases_counter + 1
            associated_precision = np.mean(cases[:, 0])
            associated_recall = np.mean(cases[:, 1])
            associated_treshold = np.mean(cases[:, 2])
            associated_nuber_wc = np.mean(cases[:, 3])
            associated_f1 = np.mean(cases[:, 4])
            associated_f2 = np.mean(cases[:, 5])
            associated_f3 = np.mean(cases[:, 6])
            associated_f4 = np.mean(cases[:, 7])
            associated_f5 = np.mean(cases[:, 8])
            associated_accuracy = np.mean(cases[:, 9])
            mean_of_accuracy_cases[counter_accuracy_cases, 0] = associated_precision
            mean_of_accuracy_cases[counter_accuracy_cases, 1] = associated_recall
            mean_of_accuracy_cases[counter_accuracy_cases, 2] = associated_treshold
            mean_of_accuracy_cases[counter_accuracy_cases, 3] = associated_nuber_wc
            mean_of_accuracy_cases[counter_accuracy_cases, 4] = associated_f1
            mean_of_accuracy_cases[counter_accuracy_cases, 5] = associated_f2
            mean_of_accuracy_cases[counter_accuracy_cases, 6] = associated_f3
            mean_of_accuracy_cases[counter_accuracy_cases, 7] = associated_f4
            mean_of_accuracy_cases[counter_accuracy_cases, 8] = associated_f5
            mean_of_accuracy_cases[counter_accuracy_cases, 9] = associated_accuracy
            counter_accuracy_cases = counter_accuracy_cases + 1
            f_output.write("User " + str(i) + ", with " + str(associated_accuracy) + " as best accuracy, has average best case: " + str(associated_precision) + " precision; " + str(associated_recall) + " recall; " + str(associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

        f_output.write("\n")
        f_output.write("˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅\n")
        f_output.write("\n")
        associated_precision = np.mean(mean_of_accuracy_cases[:, 0])
        associated_recall = np.mean(mean_of_accuracy_cases[:, 1])
        associated_treshold = np.mean(mean_of_accuracy_cases[:, 2])
        associated_nuber_wc = np.mean(mean_of_accuracy_cases[:, 3])
        associated_f1 = np.mean(mean_of_accuracy_cases[:, 4])
        associated_f2 = np.mean(mean_of_accuracy_cases[:, 5])
        associated_f3 = np.mean(mean_of_accuracy_cases[:, 6])
        associated_f4 = np.mean(mean_of_accuracy_cases[:, 7])
        associated_f5 = np.mean(mean_of_accuracy_cases[:, 8])
        associated_accuracy = np.mean(mean_of_accuracy_cases[:, 9])
        f_output.write("average of the average cases of all users: \n")
        f_output.write(str(associated_accuracy) + " accuracy; " + str(associated_precision) + " precision; " + str(associated_recall) + " recall; " + str(associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

    with open(output_precision_path, 'w+', encoding='UTF8') as f_output:
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            max_precision = np.amax(user[:, 1])
            max_precision_indexes = np.where(user[:, 1] == np.amax(user[:, 1]))
            f_output.write("User " + str(i) + ": " + str(max_precision) + " precision; in the following cases: \n")
            for max_precision_index_list in max_precision_indexes:
                best_precision_indexes_per_user[i] = max_precision_index_list
                for max_precision_index in max_precision_index_list:
                    associated_accuracy = user[max_precision_index, 0]
                    associated_recall = user[max_precision_index, 2]
                    associated_treshold = user[max_precision_index, 4]
                    associated_nuber_wc = user[max_precision_index, 5]
                    associated_f1 = user[max_precision_index, 6]
                    associated_f2 = user[max_precision_index, 7]
                    associated_f3 = user[max_precision_index, 8]
                    associated_f4 = user[max_precision_index, 9]
                    associated_f5 = user[max_precision_index, 10]

                    f_output.write(
                        str(associated_accuracy) + " accuracy; " + str(associated_recall) + " recall; " + str(
                            associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(
                            associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(
                            associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")
            f_output.write("--------------------------------------------------------------\n")

        mean_of_precision_cases = np.zeros((10, 10), dtype="float32")
        counter_precision_cases = 0

        f_output.write("best precision summary\n")
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            cases_counter = 0
            for user_indexes in best_precision_indexes_per_user:
                if i == user_indexes:
                    for index in best_precision_indexes_per_user[user_indexes]:
                        cases_counter = cases_counter + 1
            cases = np.zeros((cases_counter, 10), dtype="float32")
            cases_counter = 0
            for user_indexes in best_precision_indexes_per_user:
                if i == user_indexes:
                    for index in best_precision_indexes_per_user[user_indexes]:
                        cases[cases_counter, 0] = user[index, 0]
                        cases[cases_counter, 1] = user[index, 2]
                        cases[cases_counter, 2] = user[index, 4]
                        cases[cases_counter, 3] = user[index, 5]
                        cases[cases_counter, 4] = user[index, 6]
                        cases[cases_counter, 5] = user[index, 7]
                        cases[cases_counter, 6] = user[index, 8]
                        cases[cases_counter, 7] = user[index, 9]
                        cases[cases_counter, 8] = user[index, 10]
                        cases[cases_counter, 9] = user[index, 1]
                        cases_counter = cases_counter + 1
            associated_accuracy = np.mean(cases[:, 0])
            associated_recall = np.mean(cases[:, 1])
            associated_treshold = np.mean(cases[:, 2])
            associated_nuber_wc = np.mean(cases[:, 3])
            associated_f1 = np.mean(cases[:, 4])
            associated_f2 = np.mean(cases[:, 5])
            associated_f3 = np.mean(cases[:, 6])
            associated_f4 = np.mean(cases[:, 7])
            associated_f5 = np.mean(cases[:, 8])
            associated_precision = np.mean(cases[:, 9])
            mean_of_precision_cases[counter_precision_cases, 0] = associated_accuracy
            mean_of_precision_cases[counter_precision_cases, 1] = associated_recall
            mean_of_precision_cases[counter_precision_cases, 2] = associated_treshold
            mean_of_precision_cases[counter_precision_cases, 3] = associated_nuber_wc
            mean_of_precision_cases[counter_precision_cases, 4] = associated_f1
            mean_of_precision_cases[counter_precision_cases, 5] = associated_f2
            mean_of_precision_cases[counter_precision_cases, 6] = associated_f3
            mean_of_precision_cases[counter_precision_cases, 7] = associated_f4
            mean_of_precision_cases[counter_precision_cases, 8] = associated_f5
            mean_of_precision_cases[counter_precision_cases, 9] = associated_precision
            counter_precision_cases = counter_precision_cases + 1
            f_output.write("User " + str(i) + ", with " + str(
                associated_precision) + " as best precision, has average best case: " + str(
                associated_accuracy) + " accuracy; " + str(associated_recall) + " recall; " + str(
                associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(
                associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(
                associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

        f_output.write("\n")
        f_output.write("˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅\n")
        f_output.write("\n")
        associated_accuracy = np.mean(mean_of_precision_cases[:, 0])
        associated_recall = np.mean(mean_of_precision_cases[:, 1])
        associated_treshold = np.mean(mean_of_precision_cases[:, 2])
        associated_nuber_wc = np.mean(mean_of_precision_cases[:, 3])
        associated_f1 = np.mean(mean_of_precision_cases[:, 4])
        associated_f2 = np.mean(mean_of_precision_cases[:, 5])
        associated_f3 = np.mean(mean_of_precision_cases[:, 6])
        associated_f4 = np.mean(mean_of_precision_cases[:, 7])
        associated_f5 = np.mean(mean_of_precision_cases[:, 8])
        associated_precision = np.mean(mean_of_precision_cases[:, 9])
        f_output.write("average of the average cases of all users: \n")
        f_output.write(str(associated_precision) + " precision; " + str(associated_accuracy) + " accuracy; " + str(
            associated_recall) + " recall; " + str(associated_treshold) + " treshold; " + str(
            associated_nuber_wc) + "wc; [" + str(associated_f1) + ", " + str(associated_f2) + ", " + str(
            associated_f3) + ", " + str(associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

    with open(output_recall_path, 'w+', encoding='UTF8') as f_output:
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            max_recall = np.amax(user[:, 2])
            max_recall_indexes = np.where(user[:, 2] == np.amax(user[:, 2]))
            f_output.write("User " + str(i) + ": " + str(max_recall) + " recall; in the following cases: \n")
            for max_recall_index_list in max_recall_indexes:
                best_recall_indexes_per_user[i] = max_recall_index_list
                for max_recall_index in max_recall_index_list:
                    associated_accuracy = user[max_recall_index, 0]
                    associated_precision = user[max_recall_index, 1]
                    associated_treshold = user[max_recall_index, 4]
                    associated_nuber_wc = user[max_recall_index, 5]
                    associated_f1 = user[max_recall_index, 6]
                    associated_f2 = user[max_recall_index, 7]
                    associated_f3 = user[max_recall_index, 8]
                    associated_f4 = user[max_recall_index, 9]
                    associated_f5 = user[max_recall_index, 10]

                    f_output.write(
                        str(associated_accuracy) + " accuracy; " + str(associated_precision) + " precision; " + str(
                            associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(
                            associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(
                            associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")
            f_output.write("--------------------------------------------------------------\n")

        mean_of_recall_cases = np.zeros((10, 10), dtype="float32")
        counter_recall_cases = 0

        f_output.write("best recall summary\n")
        for i, user in enumerate(data_210_couples_no_zeros):
            if i not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                continue
            cases_counter = 0
            for user_indexes in best_recall_indexes_per_user:
                if i == user_indexes:
                    for index in best_recall_indexes_per_user[user_indexes]:
                        cases_counter = cases_counter + 1
            cases = np.zeros((cases_counter, 10), dtype="float32")
            cases_counter = 0
            for user_indexes in best_recall_indexes_per_user:
                if i == user_indexes:
                    for index in best_recall_indexes_per_user[user_indexes]:
                        cases[cases_counter, 0] = user[index, 0]
                        cases[cases_counter, 1] = user[index, 1]
                        cases[cases_counter, 2] = user[index, 4]
                        cases[cases_counter, 3] = user[index, 5]
                        cases[cases_counter, 4] = user[index, 6]
                        cases[cases_counter, 5] = user[index, 7]
                        cases[cases_counter, 6] = user[index, 8]
                        cases[cases_counter, 7] = user[index, 9]
                        cases[cases_counter, 8] = user[index, 10]
                        cases[cases_counter, 9] = user[index, 2]
                        cases_counter = cases_counter + 1
            associated_accuracy = np.mean(cases[:, 0])
            associated_precision = np.mean(cases[:, 1])
            associated_treshold = np.mean(cases[:, 2])
            associated_nuber_wc = np.mean(cases[:, 3])
            associated_f1 = np.mean(cases[:, 4])
            associated_f2 = np.mean(cases[:, 5])
            associated_f3 = np.mean(cases[:, 6])
            associated_f4 = np.mean(cases[:, 7])
            associated_f5 = np.mean(cases[:, 8])
            associated_recall = np.mean(cases[:, 9])
            mean_of_recall_cases[counter_recall_cases, 0] = associated_accuracy
            mean_of_recall_cases[counter_recall_cases, 1] = associated_precision
            mean_of_recall_cases[counter_recall_cases, 2] = associated_treshold
            mean_of_recall_cases[counter_recall_cases, 3] = associated_nuber_wc
            mean_of_recall_cases[counter_recall_cases, 4] = associated_f1
            mean_of_recall_cases[counter_recall_cases, 5] = associated_f2
            mean_of_recall_cases[counter_recall_cases, 6] = associated_f3
            mean_of_recall_cases[counter_recall_cases, 7] = associated_f4
            mean_of_recall_cases[counter_recall_cases, 8] = associated_f5
            mean_of_recall_cases[counter_recall_cases, 9] = associated_recall
            counter_recall_cases = counter_recall_cases + 1
            f_output.write("User " + str(i) + ", with " + str(
                associated_recall) + " as best recall, has average best case: " + str(
                associated_accuracy) + " accuracy; " + str(associated_precision) + " precision; " + str(
                associated_treshold) + " treshold; " + str(associated_nuber_wc) + "wc; [" + str(
                associated_f1) + ", " + str(associated_f2) + ", " + str(associated_f3) + ", " + str(
                associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

        f_output.write("\n")
        f_output.write("˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅˄˅\n")
        f_output.write("\n")
        associated_accuracy = np.mean(mean_of_recall_cases[:, 0])
        associated_precision = np.mean(mean_of_recall_cases[:, 1])
        associated_treshold = np.mean(mean_of_recall_cases[:, 2])
        associated_nuber_wc = np.mean(mean_of_recall_cases[:, 3])
        associated_f1 = np.mean(mean_of_recall_cases[:, 4])
        associated_f2 = np.mean(mean_of_recall_cases[:, 5])
        associated_f3 = np.mean(mean_of_recall_cases[:, 6])
        associated_f4 = np.mean(mean_of_recall_cases[:, 7])
        associated_f5 = np.mean(mean_of_recall_cases[:, 8])
        associated_recall = np.mean(mean_of_recall_cases[:, 9])
        f_output.write("average of the average cases of all users: \n")
        f_output.write(str(associated_recall) + " recall; " + str(associated_accuracy) + " accuracy; " + str(
            associated_precision) + " precision; " + str(associated_treshold) + " treshold; " + str(
            associated_nuber_wc) + "wc; [" + str(associated_f1) + ", " + str(associated_f2) + ", " + str(
            associated_f3) + ", " + str(associated_f4) + ", " + str(associated_f5) + "] as factor vector\n")

    # best_performance_indexes = {}
    #
    # for user_accuracy_indexes in best_accuracy_indexes_per_user.keys():
    #     for user_precision_indexes in best_precision_indexes_per_user.keys():
    #         for user_recall_indexes in best_recall_indexes_per_user.keys():
    #             if user_accuracy_indexes == user_precision_indexes == user_recall_indexes:
    #                 indexes_to_collect = []
    #                 for accuracy_index in best_accuracy_indexes_per_user[user_accuracy_indexes]:
    #                     for precision_index in best_precision_indexes_per_user[user_precision_indexes]:
    #                         for recall_index in best_recall_indexes_per_user[user_recall_indexes]:
    #                             if accuracy_index == precision_index == recall_index:
    #                                 indexes_to_collect.append(accuracy_index)
    #                 best_performance_indexes[user_accuracy_indexes] = indexes_to_collect