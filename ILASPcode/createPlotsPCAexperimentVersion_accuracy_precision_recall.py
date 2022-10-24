import csv
import numpy as np
import matplotlib.pyplot as plt

choices = [0, 1]
PCAindexes = [5,10,15,20]
scopes = ["", "_original"]
users = [i for i in range(0, 48)]
data_45_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')
data_105_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')
data_210_couples_no_zeros = np.zeros((48, 5, 5), dtype='float32')

for choice in choices:
    for scope in scopes:
        for PCAindex in PCAindexes:
            if choice == 1:
                # continue
                path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero.csv'
            else:
                continue    # temporally
                # path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_no_zero.csv'
            for user in users:
                with open(path, newline='\n') as csvFile:
                    reader = csv.reader(csvFile, delimiter=";")
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        else:
                            if int(row[4]) <= 45:  # 45CouplesCase
                                if int(row[0]) == user:
                                    data_45_couples_no_zeros[user][int(row[1]) - 1][0] = float(row[6])  # accuracy_percentage
                                    data_45_couples_no_zeros[user][int(row[1]) - 1][1] = float(row[7])  # precision_percentage
                                    data_45_couples_no_zeros[user][int(row[1]) - 1][2] = float(row[8])  # recall_percentage
                                    data_45_couples_no_zeros[user][int(row[1]) - 1][3] = float(row[9])  # train_time
                            elif (int(row[4]) > 45) and (int(row[4]) <= 105):  # 45CouplesCase
                                if int(row[0]) == user:
                                    data_105_couples_no_zeros[user][int(row[1]) - 1][0] = float(row[6])  # accuracy_percentage
                                    data_105_couples_no_zeros[user][int(row[1]) - 1][1] = float(row[7])  # precision_percentage
                                    data_105_couples_no_zeros[user][int(row[1]) - 1][2] = float(row[8])  # recall_percentage
                                    data_105_couples_no_zeros[user][int(row[1]) - 1][3] = float(row[9])  # train_time
                            elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                                if int(row[0]) == user:
                                    data_210_couples_no_zeros[user][int(row[1]) - 1][0] = float(row[6])  # accuracy_percentage
                                    data_210_couples_no_zeros[user][int(row[1]) - 1][1] = float(row[7])  # precision_percentage
                                    data_210_couples_no_zeros[user][int(row[1]) - 1][2] = float(row[8])  # recall_percentage
                                    data_210_couples_no_zeros[user][int(row[1]) - 1][3] = float(row[9])  # train_time

            for i, user_matrix in enumerate(data_210_couples_no_zeros):
                if np.all(user_matrix == 0):
                    continue
                else:
                    x = [1, 2, 3, 4, 5]
                    accuracy_percentages = user_matrix[:, 0]
                    precision_percentages = user_matrix[:, 1]
                    recall_percentages = user_matrix[:, 2]
                    train_times = user_matrix[:, 3]
                    fig, ax = plt.subplots()
                    ax.plot(x, accuracy_percentages, label="accuracy")
                    ax.plot(x, precision_percentages, label="precision")
                    ax.plot(x, recall_percentages, label="recall")
                    ax.set_xlabel('max_v and max_p with max_v == max_p')
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

