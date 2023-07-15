import csv
import numpy as np
import matplotlib.pyplot as plt

choices = [0, 1]
users = [i for i in range(0, 48)]
data_45_couples_no_zeros = np.zeros((48, 5, 4), dtype='float32')
data_105_couples_no_zeros = np.zeros((48, 5, 4), dtype='float32')
data_210_couples_no_zeros = np.zeros((48, 5, 4), dtype='float32')
list_theory_45_couple = []
list_theory_105_couple = []
list_theory_210_couple = []
for choice in choices:
    if choice == 1:
        path = './Data8Component2Std/testOutput_original20/results_zero.csv'
    else:
        continue
        # temporally
        # path = './Data8Component2Std/testOutput/results_no_zero(variant).csv'
    for user in users:
        test_not_inserted = True
        with open(path, newline='\n') as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    if int(row[4]) <= 45:    # 45CouplesCase
                        if int(row[0]) == user:
                            data_45_couples_no_zeros[user][int(row[1])-1][0] = float(row[6])       # accuracy_percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][1] = float(row[7])       # precision_percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][2] = float(row[8])       # recall_percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][3] = float(row[9])      # train_time
                            list_theory_45_couple.append(row[10])                                  # theory
                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                        if int(row[0]) == user:
                            data_105_couples_no_zeros[user][int(row[1])-1][0] = float(row[6])       # accuracy_percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][1] = float(row[7])       # precision_percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][2] = float(row[8])       # recall_percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][3] = float(row[9])      # train_time
                            list_theory_105_couple.append(row[10])                                  # theory
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[0]) == user:
                            data_210_couples_no_zeros[user][int(row[1])-1][0] = float(row[6])       # accuracy_percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][1] = float(row[7])       # precision_percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][2] = float(row[8])       # recall_percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][3] = float(row[9])      # train_time
                            list_theory_210_couple.append(row[10])                                  # theory

    user_counter = 0
    insert_counter = 0
    final_accuracy_percentages = np.zeros((10, 15))
    final_precision_percentages = np.zeros((10, 15))
    final_recall_percentages = np.zeros((10, 15))
    final_training_times = np.zeros((10, 15))
    for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_no_zeros, data_105_couples_no_zeros, data_210_couples_no_zeros):
        if user_counter not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
            user_counter += 1
            continue
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

    final_accuracy_percentages_T = final_accuracy_percentages.T
    final_precision_percentages_T = final_precision_percentages.T
    final_recall_percentages_T = final_recall_percentages.T
    final_training_times_T = final_training_times.T
    datasets = ["Dataset_45_couples", "Dataset_105_couples", "Dataset_210_couples"]
    parameters = ["maxv=maxp=" + str(i) for i in range(1,6)]
    print("considered user id: 15, 3, 32, 7, 36, 4, 20, 29, 14, 11")
    # print("considered user id: all")
    print("test size: 50")
    print("")
    mean_of_accuracy_of_all_45 = []
    mean_of_precision_of_all_45 = []
    mean_of_recall_of_all_45 = []
    mean_of_training_time_of_all_45 = []
    mean_of_accuracy_of_all_105 = []
    mean_of_precision_of_all_105 = []
    mean_of_recall_of_all_105 = []
    mean_of_training_time_of_all_105 = []
    mean_of_accuracy_of_all_210 = []
    mean_of_precision_of_all_210 = []
    mean_of_recall_of_all_210 = []
    mean_of_training_time_of_all_210 = []
    for dataset_counter in range(0,3):
        if dataset_counter == 0:
            continue
        #     for parameter_counter in range(0, 5):
        #         mean_of_accuracy = np.mean(final_accuracy_percentages_T[parameter_counter])
        #         mean_of_precision = np.mean(final_precision_percentages_T[parameter_counter])
        #         mean_of_recall = np.mean(final_recall_percentages_T[parameter_counter])
        #         mean_of_training_time = np.mean(final_training_times_T[parameter_counter])
        #         mean_of_accuracy_of_all_45.append(np.mean(final_accuracy_percentages_T[parameter_counter]))
        #         mean_of_precision_of_all_45.append(np.mean(final_precision_percentages_T[parameter_counter]))
        #         mean_of_recall_of_all_45.append(np.mean(final_recall_percentages_T[parameter_counter]))
        #         mean_of_training_time_of_all_45.append(np.mean(final_training_times_T[parameter_counter]))
        #         print("On Dataset_45_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
        elif dataset_counter == 1:
            continue
        #     for parameter_counter in range(0, 5):
        #         mean_of_accuracy = np.mean(final_accuracy_percentages_T[5+parameter_counter])
        #         mean_of_precision = np.mean(final_precision_percentages_T[5+parameter_counter])
        #         mean_of_recall = np.mean(final_recall_percentages_T[5+parameter_counter])
        #         mean_of_training_time = np.mean(final_training_times_T[5+parameter_counter])
        #         mean_of_accuracy_of_all_105.append(np.mean(final_accuracy_percentages_T[5 + parameter_counter]))
        #         mean_of_precision_of_all_105.append(np.mean(final_precision_percentages_T[5 + parameter_counter]))
        #         mean_of_recall_of_all_105.append(np.mean(final_recall_percentages_T[5 + parameter_counter]))
        #         mean_of_training_time_of_all_105.append(np.mean(final_training_times_T[5 + parameter_counter]))
        #         print("On Dataset_105_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))

        else:
            for parameter_counter in range(0, 5):
                mean_of_accuracy = np.mean(final_accuracy_percentages_T[10 + parameter_counter])
                mean_of_precision = np.mean(final_precision_percentages_T[10 + parameter_counter])
                mean_of_recall = np.mean(final_recall_percentages_T[10 + parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[10 + parameter_counter])
                mean_of_accuracy_of_all_210.append(np.mean(final_accuracy_percentages_T[10 + parameter_counter]))
                mean_of_precision_of_all_210.append(np.mean(final_precision_percentages_T[10 + parameter_counter]))
                mean_of_recall_of_all_210.append(np.mean(final_recall_percentages_T[10 + parameter_counter]))
                mean_of_training_time_of_all_210.append(np.mean(final_training_times_T[10 + parameter_counter]))
                print("On Dataset_190_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + ";  mean of training time " + str(mean_of_training_time))

    wc_number_case45 = []
    wc_number_case105 = []
    wc_number_case210 = []

    for theory in list_theory_45_couple:
        wc_number_case45.append(len(theory.split(":~")))

    for theory in list_theory_105_couple:
        wc_number_case105.append(len(theory.split(":~")))

    for theory in list_theory_210_couple:
        wc_number_case210.append(len(theory.split(":~")))

    mean_wc_numer_case45 = np.mean(wc_number_case45)
    mean_wc_numer_case105 = np.mean(wc_number_case105)
    mean_wc_numer_case210 = np.mean(wc_number_case210)


    # print("")
    # print("On dataset with 45 couple")
    # print("Train size = 45")
    # print("accuracy = " + str(np.mean(mean_of_accuracy_of_all_45)))
    # print("precision = " + str(np.mean(mean_of_precision_of_all_45)))
    # print("recall = " + str(np.mean(mean_of_recall_of_all_45)))
    # print("training time = " + str(np.mean(mean_of_training_time_of_all_45)))
    # print("")
    # print("On dataset with 105 couple")
    # print("Train size = 105")
    # print("accuracy = " + str(np.mean(mean_of_accuracy_of_all_105)))
    # print("precision = " + str(np.mean(mean_of_precision_of_all_105)))
    # print("recall = " + str(np.mean(mean_of_recall_of_all_105)))
    # print("training time = " + str(np.mean(mean_of_training_time_of_all_105)))
    print("")
    print("On dataset with 190 couple")
    print("Train size = 190")
    print("accuracy = " + str(np.mean(mean_of_accuracy_of_all_210)))
    print("precision = " + str(np.mean(mean_of_precision_of_all_210)))
    print("recall = " + str(np.mean(mean_of_recall_of_all_210)))
    print("training time = " + str(np.mean(mean_of_training_time_of_all_210)))
    print("mean number of wc = " + str(mean_wc_numer_case210))
