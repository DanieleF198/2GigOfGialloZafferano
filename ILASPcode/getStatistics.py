import csv
import numpy as np
import matplotlib.pyplot as plt

choices = [0, 1]
users = [i for i in range(0, 48)]
data_45_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')
data_105_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')
data_210_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')
test_size_210_couples_zero = np.zeros(10, dtype='float32')
test_counter = 0

for choice in choices:
    if choice == 1:
        path = './PCAexperiment/testOutput_original20/results_zero.csv'
    else:
        continue
        # temporally
        # path = './Data8Component2Std/testOutput/results_no_zero.csv'
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
                            data_45_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_45_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_45_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                        if int(row[0]) == user:
                            data_105_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_105_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_105_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[0]) == user:
                            data_210_couples_no_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                            data_210_couples_no_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                            data_210_couples_no_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            if test_not_inserted:
                                test_size_210_couples_zero[test_counter] = int(row[5])
                                test_counter = test_counter+1
                                test_not_inserted = False

    user_counter = 0
    insert_counter = 0
    final_correct_percentages = np.zeros((10, 15))
    final_uncertain_percentages = np.zeros((10, 15))
    final_incorrect_percentages = np.zeros((10, 15))
    final_correct_discarded_percentages = np.zeros((10, 15))
    final_training_times = np.zeros((10, 15))
    mean_of_test_size = np.mean(test_size_210_couples_zero)
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
        insert_counter += 1
        user_counter += 1

    final_correct_percentages_T = final_correct_percentages.T
    final_uncertain_percentages_T = final_uncertain_percentages.T
    final_incorrect_percentages_T = final_incorrect_percentages.T
    final_correct_discarded_percentages_T = final_correct_discarded_percentages.T
    final_training_times_T = final_training_times.T
    datasets = ["Dataset_45_couples", "Dataset_105_couples", "Dataset_210_couples"]
    parameters = ["maxv=maxp=" + str(i) for i in range(1,6)]
    print("considered user id: 15, 3, 32, 7, 36, 4, 20, 29, 14, 11")
    # print("considered user id: all")
    print("test size: " + str(mean_of_test_size))
    print("")
    mean_of_correct_of_all_45 = []
    mean_of_uncertain_of_all_45 = []
    mean_of_incorrect_of_all_45 = []
    mean_of_correct_discarded_of_all_45 = []
    mean_of_training_time_of_all_45 = []
    mean_of_correct_of_all_105 = []
    mean_of_uncertain_of_all_105 = []
    mean_of_incorrect_of_all_105 = []
    mean_of_correct_discarded_of_all_105 = []
    mean_of_training_time_of_all_105 = []
    mean_of_correct_of_all_210 = []
    mean_of_uncertain_of_all_210 = []
    mean_of_incorrect_of_all_210 = []
    mean_of_correct_discarded_of_all_210 = []
    mean_of_training_time_of_all_210 = []
    for dataset_counter in range(0,3):
        if dataset_counter == 0:
            # continue
            for parameter_counter in range(0,5):
                mean_of_correct = np.mean(final_correct_percentages_T[parameter_counter])
                mean_of_uncertain = np.mean(final_uncertain_percentages_T[parameter_counter])
                mean_of_incorrect = np.mean(final_incorrect_percentages_T[parameter_counter])
                mean_of_correct_discarded = np.mean(final_correct_discarded_percentages_T[parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[parameter_counter])
                mean_of_correct_of_all_45.append(np.mean(final_correct_percentages_T[parameter_counter]))
                mean_of_uncertain_of_all_45.append(np.mean(final_uncertain_percentages_T[parameter_counter]))
                mean_of_incorrect_of_all_45.append(np.mean(final_incorrect_percentages_T[parameter_counter]))
                mean_of_correct_discarded_of_all_45.append(np.mean(final_correct_discarded_percentages_T[parameter_counter]))
                mean_of_training_time_of_all_45.append(np.mean(final_training_times_T[parameter_counter]))
                print("On Dataset_45_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
                # print("On Dataset_45_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of training time " + str(mean_of_training_time))
        elif dataset_counter == 1:
            # continue
            for parameter_counter in range(0, 5):
                mean_of_correct = np.mean(final_correct_percentages_T[5+parameter_counter])
                mean_of_uncertain = np.mean(final_uncertain_percentages_T[5+parameter_counter])
                mean_of_incorrect = np.mean(final_incorrect_percentages_T[5+parameter_counter])
                mean_of_correct_discarded = np.mean(final_correct_discarded_percentages_T[5+parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[5+parameter_counter])
                mean_of_correct_of_all_105.append(np.mean(final_correct_percentages_T[5 + parameter_counter]))
                mean_of_uncertain_of_all_105.append(np.mean(final_uncertain_percentages_T[5 + parameter_counter]))
                mean_of_incorrect_of_all_105.append(np.mean(final_incorrect_percentages_T[5 + parameter_counter]))
                mean_of_correct_discarded_of_all_105.append(np.mean(final_correct_discarded_percentages_T[5 + parameter_counter]))
                mean_of_training_time_of_all_105.append(np.mean(final_training_times_T[5 + parameter_counter]))
                print("On Dataset_105_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
                # print("On Dataset_105_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of training time " + str(mean_of_training_time))

        else:
            for parameter_counter in range(0, 5):
                mean_of_correct = np.mean(final_correct_percentages_T[10 + parameter_counter])
                mean_of_uncertain = np.mean(final_uncertain_percentages_T[10 + parameter_counter])
                mean_of_incorrect = np.mean(final_incorrect_percentages_T[10 + parameter_counter])
                mean_of_correct_discarded = np.mean(final_correct_discarded_percentages_T[10 + parameter_counter])
                mean_of_training_time = np.mean(final_training_times_T[10 + parameter_counter])
                mean_of_correct_of_all_210.append(np.mean(final_correct_percentages_T[10 + parameter_counter]))
                mean_of_uncertain_of_all_210.append(np.mean(final_uncertain_percentages_T[10 + parameter_counter]))
                mean_of_incorrect_of_all_210.append(np.mean(final_incorrect_percentages_T[10 + parameter_counter]))
                mean_of_correct_discarded_of_all_210.append(np.mean(final_correct_discarded_percentages_T[10 + parameter_counter]))
                mean_of_training_time_of_all_210.append(np.mean(final_training_times_T[10 + parameter_counter]))
                print("On Dataset_190_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of uncertain " + str(mean_of_uncertain) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of correct discarded " + str(mean_of_correct_discarded) + "; mean of training time " + str(mean_of_training_time))
                # print("On Dataset_190_couples with " + parameters[parameter_counter] + " mean of correct " + str(mean_of_correct) + "; mean of incorrect " + str(mean_of_incorrect) + "; mean of training time " + str(mean_of_training_time))
    print("")
    print("On dataset with 45 couple")
    print("Train size = 45")
    print("correct = " + str(np.mean(mean_of_correct_of_all_45)))
    print("uncertain = " + str(np.mean(mean_of_uncertain_of_all_45)))
    print("incorrect = " + str(np.mean(mean_of_incorrect_of_all_45)))
    print("correct discarded = " + str(np.mean(mean_of_correct_discarded_of_all_45)))
    print("training time = " + str(np.mean(mean_of_training_time_of_all_45)))
    print("")
    print("On dataset with 105 couple")
    print("Train size = 105")
    print("correct = " + str(np.mean(mean_of_correct_of_all_105)))
    print("uncertain = " + str(np.mean(mean_of_uncertain_of_all_105)))
    print("incorrect = " + str(np.mean(mean_of_incorrect_of_all_105)))
    print("correct discarded = " + str(np.mean(mean_of_correct_discarded_of_all_105)))
    print("training time = " + str(np.mean(mean_of_training_time_of_all_105)))
    print("")
    print("On dataset with 190 couple")
    print("Train size = 190")
    print("correct = " + str(np.mean(mean_of_correct_of_all_210)))
    print("uncertain = " + str(np.mean(mean_of_uncertain_of_all_210)))
    print("incorrect = " + str(np.mean(mean_of_incorrect_of_all_210)))
    print("correct discarded = " + str(np.mean(mean_of_correct_discarded_of_all_210)))
    print("training time = " + str(np.mean(mean_of_training_time_of_all_210)))
