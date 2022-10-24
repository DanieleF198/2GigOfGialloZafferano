import csv
import numpy as np
import matplotlib.pyplot as plt

choices = [0, 1]
users = [i for i in range(0, 48)]

for choice in choices:
    if choice == 0:
        continue
        path = './Data8Component2Std/testOutput/results_no_zero.csv'
    else:
        # continue    # temporally
        path = './Data8Component2Std/testOutput/results_zero.csv'
    data_45_couples_no_zeros = np.zeros((48, 5, 7), dtype='float32')
    data_105_couples_no_zeros = np.zeros((48, 5, 7), dtype='float32')
    data_210_couples_no_zeros = np.zeros((48, 5, 7), dtype='float32')
    for user in users:
        with open(path, newline='\n') as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    if int(row[4]) <= 45:    # 45CouplesCase
                        if int(row[0]) == user:
                            data_45_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_45_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            data_45_couples_no_zeros[user][int(row[1]) - 1][5] = float(row[4])     # train_size
                            data_45_couples_no_zeros[user][int(row[1]) - 1][6] = float(row[5])     # test_size
                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                        if int(row[0]) == user:
                            data_105_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_105_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            data_105_couples_no_zeros[user][int(row[1]) - 1][5] = float(row[4])     # train_size
                            data_105_couples_no_zeros[user][int(row[1]) - 1][6] = float(row[5])     # test_size
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[0]) == user:
                            data_210_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_210_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            data_210_couples_no_zeros[user][int(row[1]) - 1][5] = float(row[4])     # train_size
                            data_210_couples_no_zeros[user][int(row[1]) - 1][6] = float(row[5])     # test_size

    user_counter = 0
    insert_counter = 0
    final_correct_percentages = np.zeros((48, 15))
    final_uncertain_percentages = np.zeros((48, 15))
    final_incorrect_percentages = np.zeros((48, 15))
    final_correct_discarded_percentages = np.zeros((48, 15))
    final_training_times = np.zeros((48, 15))
    final_train_size = np.zeros((48, 3))
    final_test_size = np.zeros(48)
    for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_no_zeros, data_105_couples_no_zeros, data_210_couples_no_zeros):
        if user_counter not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
            user_counter += 1
            continue
        correct_percentages_45 = user_matrix_45[:, 0]
        uncertain_percentages_45 = user_matrix_45[:, 1]
        incorrect_percentages_45 = user_matrix_45[:, 2]
        correct_discarded_percentages_45 = user_matrix_45[:, 3]
        correct_percentages_105 = user_matrix_105[:, 0]
        uncertain_percentages_105 = user_matrix_105[:, 1]
        incorrect_percentages_105 = user_matrix_105[:, 2]
        correct_discarded_percentages_105 = user_matrix_105[:, 3]
        correct_percentages_210 = user_matrix_210[:, 0]
        uncertain_percentages_210 = user_matrix_210[:, 1]
        incorrect_percentages_210 = user_matrix_210[:, 2]
        correct_discarded_percentages_210 = user_matrix_210[:, 3]
        correct_percentages_temp = np.concatenate((correct_percentages_45, correct_percentages_105))
        uncertain_percentages_temp = np.concatenate((uncertain_percentages_45, uncertain_percentages_105))
        incorrect_percentages_temp = np.concatenate((incorrect_percentages_45, incorrect_percentages_105))
        correct_discarded_percentages_temp = np.concatenate((correct_discarded_percentages_45, correct_discarded_percentages_105))
        correct_percentages = np.concatenate((correct_percentages_temp, correct_percentages_210))
        uncertain_percentages = np.concatenate((uncertain_percentages_temp, uncertain_percentages_210))
        incorrect_percentages = np.concatenate((incorrect_percentages_temp, incorrect_percentages_210))
        correct_discarded_percentages = np.concatenate((correct_discarded_percentages_temp, correct_discarded_percentages_210))

        train_times_45 = user_matrix_45[:, 4]
        train_times_105 = user_matrix_105[:, 4]
        train_times_210 = user_matrix_210[:, 4]
        train_times_temp = np.concatenate((train_times_45, train_times_105))
        train_times = np.concatenate((train_times_temp, train_times_210))

        final_correct_percentages[insert_counter] = correct_percentages
        final_uncertain_percentages[insert_counter] = uncertain_percentages
        final_incorrect_percentages[insert_counter] = incorrect_percentages
        final_correct_discarded_percentages[insert_counter] = correct_discarded_percentages
        final_training_times[insert_counter] = train_times
        final_train_size[insert_counter] = (user_matrix_45[0, 5], user_matrix_105[0, 5], user_matrix_210[0, 5])
        final_test_size[insert_counter] = user_matrix_45[0, 6]

        insert_counter += 1
        user_counter += 1

    final_correct_percentages = final_correct_percentages[0:10]     # because are 10 users
    final_uncertain_percentages = final_uncertain_percentages[0:10]
    final_incorrect_percentages = final_incorrect_percentages[0:10]
    final_correct_discarded_percentages = final_correct_discarded_percentages[0:10]
    final_training_times = final_training_times[0:10]
    final_train_size = final_train_size[0:10]
    final_test_size = final_test_size[0:10]
    final_correct_percentages_T = final_correct_percentages.T
    final_uncertain_percentages_T = final_uncertain_percentages.T
    final_incorrect_percentages_T = final_incorrect_percentages.T
    final_correct_discarded_percentages_T = final_correct_discarded_percentages.T
    final_training_times_T = final_training_times.T
    final_train_size_T = final_train_size.T
    datasets = ["Dataset_45_couples", "Dataset_105_couples", "Dataset_190_couples"]
    parameters = ["maxv=maxp=" + str(i) for i in range(1, 6)]
    list_correct_45 = []
    list_uncertain_45 = []
    list_incorrect_45 = []
    list_correct_discarded_45 = []
    list_training_time_45 = []
    list_correct_105 = []
    list_uncertain_105 = []
    list_incorrect_105 = []
    list_correct_discarded_105 = []
    list_training_time_105 = []
    list_correct_210 = []
    list_uncertain_210 = []
    list_incorrect_210 = []
    list_correct_discarded_210 = []
    list_training_time_210 = []
    mean_of_train_size_45 = np.mean(final_train_size_T[0])
    mean_of_train_size_105 = np.mean(final_train_size_T[1])
    mean_of_train_size_210 = np.mean(final_train_size_T[2])
    print("considered user id: 15, 3, 32, 7, 36, 4, 20, 29, 14, 11")
    mean_of_test_size = np.mean(final_test_size[0])
    print("mean of test size: " + str(mean_of_test_size))
    print("mean of train sizes: " + str(round(mean_of_train_size_45)) + " - " + str(round(mean_of_train_size_105)) + " - " + str(round(mean_of_train_size_210)))
    print("")
    for dataset_counter in range(0, 3):
        if dataset_counter == 0:
            for parameter_counter in range(0, 5):
                mean_of_correct = np.mean(final_correct_percentages_T[parameter_counter])
                mean_of_uncertain = np.mean(final_uncertain_percentages_T[parameter_counter])
                mean_of_incorrect = np.mean(final_incorrect_percentages_T[parameter_counter])
                mean_of_correct_discarded = np.mean(final_correct_discarded_percentages_T[parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[parameter_counter])
                list_correct_45.append(mean_of_correct)
                list_uncertain_45.append(mean_of_uncertain)
                list_incorrect_45.append(mean_of_incorrect)
                list_correct_discarded_45.append(mean_of_correct_discarded)
                list_training_time_45.append(mean_of_training_time)
                print("On Dataset_45_couples with " + parameters[parameter_counter] + "; mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
        elif dataset_counter == 1:
            print("")
            for parameter_counter in range(0, 5):
                mean_of_correct = np.mean(final_correct_percentages_T[5+parameter_counter])
                mean_of_uncertain = np.mean(final_uncertain_percentages_T[5+parameter_counter])
                mean_of_incorrect = np.mean(final_incorrect_percentages_T[5+parameter_counter])
                mean_of_correct_discarded = np.mean(final_correct_discarded_percentages_T[5+parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[5+parameter_counter])
                list_correct_105.append(mean_of_correct)
                list_uncertain_105.append(mean_of_uncertain)
                list_incorrect_105.append(mean_of_incorrect)
                list_correct_discarded_105.append(mean_of_correct_discarded)
                list_training_time_105.append(mean_of_training_time)
                print("On Dataset_105_couples with " + parameters[parameter_counter] + "; mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
        else:
            print("")
            for parameter_counter in range(0, 5):
                final_correct_percentages_T[final_correct_percentages_T == 0] = np.nan
                final_uncertain_percentages_T[final_uncertain_percentages_T == 0] = np.nan
                final_incorrect_percentages_T[final_incorrect_percentages_T == 0] = np.nan
                final_correct_discarded_percentages_T[final_correct_discarded_percentages_T == 0] = np.nan
                final_training_times_T[final_training_times_T == 0] = np.nan
                mean_of_correct = np.nanmean(final_correct_percentages_T[10 + parameter_counter])
                mean_of_uncertain = np.nanmean(final_uncertain_percentages_T[10 + parameter_counter])
                mean_of_incorrect = np.nanmean(final_incorrect_percentages_T[10 + parameter_counter])
                mean_of_correct_discarded = np.nanmean(final_correct_discarded_percentages_T[10 + parameter_counter])
                mean_of_training_time = np.nanmean(final_training_times_T[10 + parameter_counter])
                list_correct_210.append(mean_of_correct)
                list_uncertain_210.append(mean_of_uncertain)
                list_incorrect_210.append(mean_of_incorrect)
                list_correct_discarded_210.append(mean_of_correct_discarded)
                list_training_time_210.append(mean_of_training_time)
                print("On Dataset_190_couples with " + parameters[parameter_counter] + "; mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
    print("")
    print("mean of means: ")
    print("On dataset with 45 couples:")
    print("train size = " + str(round(mean_of_train_size_45)))
    print("correct = " + str(np.mean(list_correct_45)))
    print("uncertain = " + str(np.mean(list_uncertain_45)))
    print("incorrect = " + str(np.mean(list_incorrect_45)))
    print("correct discarded = " + str(np.mean(list_correct_discarded_45)))
    print("training time = " + str(np.mean(list_training_time_45)))
    print("")
    print("On dataset with 105 couples:")
    print("train size = " + str(round(mean_of_train_size_105)))
    print("correct = " + str(np.mean(list_correct_105)))
    print("uncertain = " + str(np.mean(list_uncertain_105)))
    print("incorrect = " + str(np.mean(list_incorrect_105)))
    print("correct discarded = " + str(np.mean(list_correct_discarded_105)))
    print("training time = " + str(np.mean(list_training_time_105)))
    print("")
    list_correct_210_no_zeros = list(filter(lambda x: x != 0, list_correct_210))
    list_uncertain_210_no_zeros = list(filter(lambda x: x != 0, list_uncertain_210))
    list_incorrect_210_no_zeros = list(filter(lambda x: x != 0, list_incorrect_210))
    list_correct_discarded_210_no_zeros = list(filter(lambda x: x != 0, list_correct_discarded_210))
    list_training_time_210_no_zeros = list(filter(lambda x: x != 0, list_training_time_210))
    print("On dataset with 190 couples:")
    print("train size = " + str(round(mean_of_train_size_210)))
    print("correct = " + str(np.nanmean(list_correct_210_no_zeros)))
    print("uncertain = " + str(np.nanmean(list_uncertain_210_no_zeros)))
    print("incorrect = " + str(np.nanmean(list_incorrect_210_no_zeros)))
    print("correct discarded = " + str(np.nanmean(list_correct_discarded_210_no_zeros)))
    print("training time = " + str(np.nanmean(list_training_time_210_no_zeros)))
