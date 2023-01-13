import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

choices = [0, 1]
users = [i for i in range(0, 54)]
data_45_couples_no_zeros = np.zeros((54, 1, 4), dtype='float32')
data_105_couples_no_zeros = np.zeros((54, 1, 4), dtype='float32')
data_210_couples_no_zeros = np.zeros((54, 1, 4), dtype='float32')
list_theory_45_couple = []
list_theory_105_couple = []
list_theory_210_couple = []
for choice in choices:
    if choice == 1:
        path = './Data8Component2Std/testOutput/results_zero_founded_parameters.csv'
    else:
        continue
        # temporally
        # path = './Data8Component2Std/testOutput/results_no_zero.csv'
    for user in users:
        parameter_base = 0
        test_not_inserted = True
        with open(path, newline='\n') as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    if int(row[4]) <= 45:    # 45CouplesCase
                        if int(row[1]) == 1 and int(row[2]) == 4:
                            if int(row[0]) == user:
                                data_45_couples_no_zeros[user][0][0] = float(row[6])       # accuracy_percentage
                                data_45_couples_no_zeros[user][0][1] = float(row[7])       # precision_percentage
                                data_45_couples_no_zeros[user][0][2] = float(row[8])       # recall_percentage
                                data_45_couples_no_zeros[user][0][3] = float(row[9])      # train_time
                                list_theory_45_couple.append(row[10])                                  # theory
                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                        if int(row[1]) == 1 and int(row[2]) == 5:
                            if int(row[0]) == user:
                                data_105_couples_no_zeros[user][0][0] = float(row[6])       # accuracy_percentage
                                data_105_couples_no_zeros[user][0][1] = float(row[7])       # precision_percentage
                                data_105_couples_no_zeros[user][0][2] = float(row[8])       # recall_percentage
                                data_105_couples_no_zeros[user][0][3] = float(row[9])      # train_time
                                list_theory_105_couple.append(row[10])                                  # theory
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[1]) == 1 and int(row[2]) == 5:
                            if int(row[0]) == user:
                                data_210_couples_no_zeros[user][0][0] = float(row[6])       # accuracy_percentage
                                data_210_couples_no_zeros[user][0][1] = float(row[7])       # precision_percentage
                                data_210_couples_no_zeros[user][0][2] = float(row[8])       # recall_percentage
                                data_210_couples_no_zeros[user][0][3] = float(row[9])      # train_time
                                list_theory_210_couple.append(row[10])                                  # theory

    user_counter = 0
    insert_counter = 0
    final_accuracy_percentages = np.zeros((54, 3))
    final_precision_percentages = np.zeros((54, 3))
    final_recall_percentages = np.zeros((54, 3))
    final_training_times = np.zeros((54, 3))
    for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_no_zeros, data_105_couples_no_zeros, data_210_couples_no_zeros):
        accuracy_percentages_45 = user_matrix_45[:, 0]
        precision_percentages_45 = user_matrix_45[:, 1]
        recall_percentages_45 = user_matrix_45[:, 2]
        accuracy_percentages_105 = user_matrix_105[:, 0]
        precision_percentages_105 = user_matrix_105[:, 1]
        recall_percentages_105 = user_matrix_105[:, 2]
        accuracy_percentages_210 = user_matrix_210[:, 0]
        precision_percentages_210 = user_matrix_210[:, 1]
        recall_percentages_210 = user_matrix_210[:, 2]
        accuracy_percentages_temp = np.concatenate((accuracy_percentages_45, accuracy_percentages_105))
        precision_percentages_temp = np.concatenate((precision_percentages_45, precision_percentages_105))
        recall_percentages_temp = np.concatenate((recall_percentages_45, recall_percentages_105))
        accuracy_percentages = np.concatenate((accuracy_percentages_temp, accuracy_percentages_210))
        precision_percentages = np.concatenate((precision_percentages_temp, precision_percentages_210))
        recall_percentages = np.concatenate((recall_percentages_temp, recall_percentages_210))
        train_times_45 = user_matrix_45[:, 3]
        train_times_105 = user_matrix_105[:, 3]
        train_times_210 = user_matrix_210[:, 3]
        train_times_temp = np.concatenate((train_times_45, train_times_105))
        train_times = np.concatenate((train_times_temp, train_times_210))
        final_accuracy_percentages[insert_counter] = accuracy_percentages
        final_precision_percentages[insert_counter] = precision_percentages
        final_recall_percentages[insert_counter] = recall_percentages
        final_training_times[insert_counter] = train_times
        insert_counter += 1
        user_counter += 1

    datasets = ["Dataset_45_couples", "Dataset_105_couples", "Dataset_210_couples"]
    parameters = ["maxv= " + str(i) + "; maxp=" + str(j) for i in range(1, 6) for j in range(1, 6)]
    parameters_for_graph = [str(i) + ";" + str(j) for i in range(1, 6) for j in range(1, 6)]
    print("considered user id: all")
    # print("considered user id: all")
    print("test size: 105")
    print("")
    for dataset_counter in range(0,3):
        if dataset_counter == 0:
            mean_of_accuracy = np.mean(final_accuracy_percentages[:, 0])
            mean_of_precision = np.mean(final_precision_percentages[:, 0])
            mean_of_recall = np.mean(final_recall_percentages[:, 0])
            mean_of_training_time = np.mean(final_training_times[:, 0])
            print("On Dataset_45_couples with max_v=1;max_p=4 mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
            print("")
        elif dataset_counter == 1:
            mean_of_accuracy = np.mean(final_accuracy_percentages[:, 1])
            mean_of_precision = np.mean(final_precision_percentages[:, 1])
            mean_of_recall = np.mean(final_recall_percentages[:, 1])
            mean_of_training_time = np.mean(final_training_times[:, 1])
            print("On Dataset_105_couples with max_v=1;max_p=5 mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
            print("")
        else:
            mean_of_accuracy = np.mean(final_accuracy_percentages[:, 2])
            mean_of_precision = np.mean(final_precision_percentages[:, 2])
            mean_of_recall = np.mean(final_recall_percentages[:, 2])
            mean_of_training_time = np.mean(final_training_times[:, 2])
            print("On Dataset_190_couples with max_v=1;max_p=5 mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + ";  mean of training time " + str(mean_of_training_time))
            print("")
