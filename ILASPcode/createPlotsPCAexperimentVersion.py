import csv
import numpy as np
import matplotlib.pyplot as plt

choices = [0, 1]
PCAindexes = [5,10,15,20]
scopes = ["", "_original"]
users = [i for i in range(0, 48)]
data_45_couples_zeros = np.zeros((48, 5, 5), dtype='float32')
data_105_couples_zeros = np.zeros((48, 5, 5), dtype='float32')
data_210_couples_zeros = np.zeros((48, 5, 5), dtype='float32')

for choice in choices:
    for scope in scopes:
        for PCAindex in PCAindexes:
            if choice == 1:
                continue
                # path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero.csv'
            else:
                # continue    # temporally
                path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_no_zero.csv'
            for user in users:
                with open(path, newline='\n') as csvFile:
                    reader = csv.reader(csvFile, delimiter=";")
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        else:
                            if int(row[4]) <= 45:    # 45CouplesCase
                                if int(row[0]) == user:
                                    data_45_couples_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                                    data_45_couples_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                                    data_45_couples_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                                    data_45_couples_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                                    data_45_couples_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                                if int(row[0]) == user:
                                    data_105_couples_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                                    data_105_couples_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                                    data_105_couples_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                                    data_105_couples_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                                    data_105_couples_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time
                            elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                                if int(row[0]) == user:
                                    data_210_couples_zeros[user][int(row[1])-1][0] = float(row[9])       # correct percentage
                                    data_210_couples_zeros[user][int(row[1])-1][1] = float(row[10])      # uncertain percentage
                                    data_210_couples_zeros[user][int(row[1])-1][2] = float(row[11])      # incorrect percentage
                                    data_210_couples_zeros[user][int(row[1])-1][3] = float(row[12])      # correct/test_size-uncertain
                                    data_210_couples_zeros[user][int(row[1])-1][4] = float(row[13])      # train_time


            # for i, user_matrix in enumerate(data_45_couples_zeros):
            #     if np.all(user_matrix == 0):
            #         continue
            #     else:
            #         x = [1, 2, 3, 4, 5]
            #         correct_percentages = user_matrix[:, 0]
            #         uncertain_percentages = user_matrix[:, 1]
            #         incorrect_percentages = user_matrix[:, 2]
            #         correct_discarded_percentages = user_matrix[:, 3]
            #         train_times = user_matrix[:, 4]
            #         fig, ax = plt.subplots()
            #         ax.plot(x, correct_percentages, label="correct")
            #         ax.plot(x, uncertain_percentages, label="uncertain")
            #         ax.plot(x, incorrect_percentages, label="incorrect")
            #         ax.plot(x, correct_discarded_percentages, label="correct over not uncertain")
            #         ax.set_xlabel('max_v and max_p with max_v == max_p')
            #         ax.set_ylabel('percentage')
            #         ax.set_title('user' + str(i) + "-45Couples-Zeros")
            #         ax.legend(loc="lower center", ncol=4, prop={'size': 8})
            #         plt.savefig("./PCAexperiment/plots/percentages/45Couple/user" + str(i) +".png", dpi=300)
            #
            #         fig, ax = plt.subplots()
            #         ax.plot(x, train_times, label="training time")
            #         ax.set_xlabel('max_v and max_p with max_v == max_p')
            #         ax.set_ylabel('time in second')
            #         ax.set_title('user' + str(i) + "-45Couples-Zeros")
            #         ax.legend(loc="lower right")
            #         plt.savefig("./PCAexperiment/plots/training_time/45Couple/user" + str(i) +".png", dpi=300)
            #
            # for i, user_matrix in enumerate(data_105_couples_zeros):
            #     if np.all(user_matrix == 0):
            #         continue
            #     else:
            #         x = [1, 2, 3, 4, 5]
            #         correct_percentages = user_matrix[:, 0]
            #         uncertain_percentages = user_matrix[:, 1]
            #         incorrect_percentages = user_matrix[:, 2]
            #         correct_discarded_percentages = user_matrix[:, 3]
            #         train_times = user_matrix[:, 4]
            #         fig, ax = plt.subplots()
            #         ax.plot(x, correct_percentages, label="correct")
            #         ax.plot(x, uncertain_percentages, label="uncertain")
            #         ax.plot(x, incorrect_percentages, label="incorrect")
            #         ax.plot(x, correct_discarded_percentages, label="correct over not uncertain")
            #         ax.set_xlabel('max_v and max_p with max_v == max_p')
            #         ax.set_ylabel('percentage')
            #         ax.set_title('user' + str(i) + "-105Couples-Zeros")
            #         ax.legend(loc="lower center", ncol=4, prop={'size': 8})
            #         plt.savefig("./PCAexperiment/plots/percentages/105Couple/user" + str(i) +".png", dpi=300)
            #
            #         fig, ax = plt.subplots()
            #         ax.plot(x, train_times, label="training time")
            #         ax.set_xlabel('max_v and max_p with max_v == max_p')
            #         ax.set_ylabel('time in second')
            #         ax.set_title('user' + str(i) + "-105Couples-Zeros")
            #         ax.legend(loc="lower right")
            #         plt.savefig("./PCAexperiment/plots/training_time/105Couple/user" + str(i) +".png", dpi=300)

            for i, user_matrix in enumerate(data_210_couples_zeros):
                if np.all(user_matrix == 0):
                    continue
                else:
                    x = [1, 2, 3, 4, 5]
                    correct_percentages = user_matrix[:, 0]
                    uncertain_percentages = user_matrix[:, 1]
                    incorrect_percentages = user_matrix[:, 2]
                    correct_discarded_percentages = user_matrix[:, 3]
                    train_times = user_matrix[:, 4]
                    fig, ax = plt.subplots()
                    ax.plot(x, correct_percentages, label="correct")
                    ax.plot(x, uncertain_percentages, label="uncertain")
                    ax.plot(x, incorrect_percentages, label="incorrect")
                    ax.plot(x, correct_discarded_percentages, label="correct over not uncertain")
                    ax.set_xlabel('max_v and max_p with max_v == max_p')
                    ax.set_ylabel('percentage')
                    if scope == "":
                        ax.set_title('user' + str(i) + "-190Couples-Zeros")
                    else:
                        ax.set_title('user' + str(i) + "-150Couples-Zeros")
                    ax.legend(loc="lower center", ncol=4, prop={'size': 8})
                    if scope == "":
                        plt.savefig("./PCAexperiment/plots" + scope + str(PCAindex) +"/percentages/210Couple/user" + str(i) +".png", dpi=300)
                    else:
                        plt.savefig("./PCAexperiment/plots" + scope + str(PCAindex) +"/percentages/150Couple/user" + str(i) +".png", dpi=300)
                    fig, ax = plt.subplots()
                    ax.plot(x, train_times, label="training time")
                    ax.set_xlabel('max_v and max_p with max_v == max_p')
                    ax.set_ylabel('time in second')
                    if scope == "":
                        ax.set_title('user' + str(i) + "-190Couples-Zeros")
                    else:
                        ax.set_title('user' + str(i) + "-150Couples-Zeros")
                    ax.legend(loc="lower right")
                    if scope == "":
                        plt.savefig("./PCAexperiment/plots" + scope + str(PCAindex) +"/training_time/210Couple/user" + str(i) +".png", dpi=300)
                    else:
                        plt.savefig("./PCAexperiment/plots" + scope + str(PCAindex) +"/training_time/150Couple/user" + str(i) +".png", dpi=300)
            # user_counter = 0
            # for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_zeros, data_105_couples_zeros, data_210_couples_zeros):
            #     if np.all(user_matrix_45 == 0):
            #         user_counter += 1
            #         continue
            #     if np.all(user_matrix_105 == 0):
            #         user_counter += 1
            #         continue
            #     if np.all(user_matrix_210 == 0):
            #         user_counter += 1
            #         continue
            #     x = ["45-1", "45-2", "45-3", "45-4", "45-5", "105-1", "105-2", "105-3", "105-4", "105-5", "190-1", "190-2", "190-3", "190-4", "190-5"]
            #     correct_percentages_45 = user_matrix_45[:, 0]
            #     uncertain_percentages_45 = user_matrix_45[:, 1]
            #     incorrect_percentages_45 = user_matrix_45[:, 2]
            #     correct_discarded_percentages_45 = user_matrix_45[:, 3]
            #     correct_percentages_105 = user_matrix_105[:, 0]
            #     uncertain_percentages_105 = user_matrix_105[:, 1]
            #     incorrect_percentages_105 = user_matrix_105[:, 2]
            #     correct_discarded_percentages_105 = user_matrix_105[:, 3]
            #     correct_percentages_210 = user_matrix_210[:, 0]
            #     uncertain_percentages_210 = user_matrix_210[:, 1]
            #     incorrect_percentages_210 = user_matrix_210[:, 2]
            #     correct_discarded_percentages_210 = user_matrix_210[:, 3]
            #     correct_percentages_temp = np.concatenate((correct_percentages_45, correct_percentages_105))
            #     uncertain_percentages_temp = np.concatenate((uncertain_percentages_45, uncertain_percentages_105))
            #     incorrect_percentages_temp = np.concatenate((incorrect_percentages_45, incorrect_percentages_105))
            #     correct_discarded_percentages_temp = np.concatenate((correct_discarded_percentages_45, correct_discarded_percentages_105))
            #     correct_percentages = np.concatenate((correct_percentages_temp, correct_percentages_210))
            #     uncertain_percentages = np.concatenate((uncertain_percentages_temp, uncertain_percentages_210))
            #     incorrect_percentages = np.concatenate((incorrect_percentages_temp, incorrect_percentages_210))
            #     correct_discarded_percentages = np.concatenate((correct_discarded_percentages_temp, correct_discarded_percentages_210))
            #     fig, ax = plt.subplots()
            #     ax.plot(x, correct_percentages, label="correct")
            #     ax.plot(x, uncertain_percentages, label="uncertain")
            #     ax.plot(x, incorrect_percentages, label="incorrect")
            #     ax.plot(x, correct_discarded_percentages, label="correct over not uncertain")
            #     ax.set_xlabel('45/105/210Couple-(max_v and max_p) with max_v == max_p')
            #     ax.set_ylabel('percentage')
            #     ax.set_title('user' + str(user_counter) + "-45Couples-105Couples-190Couples-Zeros")
            #     ax.legend(loc="lower center", ncol=4, prop={'size': 8})
            #     plt.savefig("./PCAexperiment/plots/percentages/AllCouple/user" + str(user_counter) + ".png", dpi=300)
            #     user_counter += 1
            #
            # user_counter = 0
            # for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_zeros, data_105_couples_zeros, data_210_couples_zeros):
            #     if np.all(user_matrix_45 == 0):
            #         user_counter += 1
            #         continue
            #     if np.all(user_matrix_105 == 0):
            #         user_counter += 1
            #         continue
            #     if np.all(user_matrix_210 == 0):
            #         user_counter += 1
            #         continue
            #     x = ["45-1", "45-2", "45-3", "45-4", "45-5", "105-1", "105-2", "105-3", "105-4", "105-5", "210-1", "210-2", "210-3", "210-4", "210-5"]
            #     train_times_45 = user_matrix_45[:, 4]
            #     train_times_105 = user_matrix_105[:, 4]
            #     train_times_210 = user_matrix_210[:, 4]
            #     train_times_temp = np.concatenate((train_times_45, train_times_105))
            #     train_times = np.concatenate((train_times_temp, train_times_210))
            #     fig, ax = plt.subplots()
            #     ax.plot(x, train_times, label="training time")
            #     ax.set_xlabel('45/105/210Couple-(max_v and max_p) with max_v == max_p')
            #     ax.set_ylabel('time in second')
            #     ax.set_title('user' + str(user_counter) + "-45Couples-105Couples-210Couples-Zeros")
            #     ax.legend(loc="lower right")
            #     plt.savefig("./PCAexperiment/plots/training_time/AllCouple/user" + str(user_counter) + ".png", dpi=300)
            #     user_counter += 1
