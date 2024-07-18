import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


choices = [0, 1]
users = [i for i in range(0, 54)]
data_45_couples_no_zeros = np.zeros((54, 5, 5, 4), dtype='float32')
data_105_couples_no_zeros = np.zeros((54, 5, 5, 4), dtype='float32')
data_210_couples_no_zeros = np.zeros((54, 5, 5, 4), dtype='float32')

for choice in choices:
    if choice == 1:
        path = './Data8Component2Std/testOutput/results_zero_all_combination.csv'
    else:
        continue    # temporally
        # path = './Data8Component2Std/testOutput/results_no_zero(variant).csv'
    for user in users:
        with open(path, newline='\n') as csvFile:
            reader = csv.reader(csvFile, delimiter=";")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                else:
                    if int(row[4]) <= 45:  # 45CouplesCase
                        if int(row[0]) == user:
                            data_45_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][0] = float(row[6])  # accuracy_percentage
                            data_45_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][1] = float(row[7])  # precision_percentage
                            data_45_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][2] = float(row[8])  # recall_percentage
                            data_45_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][3] = float(row[9])  # train_time
                    elif (int(row[4]) > 45) and (int(row[4]) <= 105):  # 45CouplesCase
                        if int(row[0]) == user:
                            data_105_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][0] = float(row[6])  # accuracy_percentage
                            data_105_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][1] = float(row[7])  # precision_percentage
                            data_105_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][2] = float(row[8])  # recall_percentage
                            data_105_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][3] = float(row[9])  # train_time
                    elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                        if int(row[0]) == user:
                            data_210_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][0] = float(row[6])  # accuracy_percentage
                            data_210_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][1] = float(row[7])  # precision_percentage
                            data_210_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][2] = float(row[8])  # recall_percentage
                            data_210_couples_no_zeros[user][int(row[1]) - 1][int(row[2]) - 1][3] = float(row[9])  # train_time

    # NOTE: dict are useless, but since I have created Them, I left them here for eventual future scopes
    data_45_couples_no_zeros = data_45_couples_no_zeros.reshape((54, 25, 4))
    data_105_couples_no_zeros = data_105_couples_no_zeros.reshape((54, 25, 4))
    data_210_couples_no_zeros = data_210_couples_no_zeros.reshape((54, 25, 4))
    # NOTE: the following line is made AD-HOC (brute-force) for this specific case. The problem is that the time required by user 4 when max_v = 4 and max_p = 5 is out of range, ruining the final plot
    # data_210_couples_no_zeros[4, 19, 3] = 0
    cases_dict_45 = {}
    cases_dict_105 = {}
    cases_dict_210 = {}
    cases_np_45 = np.zeros((25, 4), dtype="float32")
    cases_np_105 = np.zeros((25, 4), dtype="float32")
    cases_np_210 = np.zeros((25, 4), dtype="float32")
    m_v = 1
    m_p = 1
    for case in range(0,  25):
        np_values_45 = np.zeros((10, 4), dtype="float32")
        np_values_105 = np.zeros((10, 4), dtype="float32")
        np_values_210 = np.zeros((10, 4), dtype="float32")
        u_counter = 0
        for user_matrix_45, user_matrix_105, user_matrix_210 in zip(data_45_couples_no_zeros, data_105_couples_no_zeros, data_210_couples_no_zeros):
            if np.all(user_matrix_45 == 0) and np.all(user_matrix_105 == 0) and np.all(user_matrix_210 == 0):
                continue
            else:   # if one is != 0 then all are != 0
                np_values_45[u_counter, :] = user_matrix_45[case]
                np_values_105[u_counter, :] = user_matrix_105[case]
                np_values_210[u_counter, :] = user_matrix_210[case]
                u_counter += 1
        cases_dict_45[str(m_v) + ";" + str(m_p)] = np.mean(np_values_45, axis=0).tolist()
        cases_dict_105[str(m_v) + ";" + str(m_p)] = np.mean(np_values_105, axis=0).tolist()
        cases_dict_210[str(m_v) + ";" + str(m_p)] = np.mean(np_values_210, axis=0).tolist()
        cases_np_45[case] = np.mean(np_values_45, axis=0).tolist()
        cases_np_105[case] = np.mean(np_values_105, axis=0).tolist()
        cases_np_210[case] = np.mean(np_values_210, axis=0).tolist()
        if m_p == 5:
            m_p = 1
            m_v += 1
        else:
            m_p += 1
    # normalization of the results between 0 and 1 in respect to the corresponding dataset using https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    scaler45 = MinMaxScaler()
    scaler105 = MinMaxScaler()
    scaler210 = MinMaxScaler()
    scaled_time_data45 = scaler45.fit_transform(cases_np_45[:, 3].reshape(-1, 1))
    scaled_time_data105 = scaler45.fit_transform(cases_np_105[:, 3].reshape(-1, 1))
    scaled_time_data210 = scaler45.fit_transform(cases_np_210[:, 3].reshape(-1, 1))


    indices = np.arange(25)
    bins = [str(i) + "-" + str(j) for i in range(1, 6) for j in range(1, 6)]
    bar_width = 0.2
    plt.figure(figsize=(20, 10), dpi=300)
    plt.bar(indices, scaled_time_data45.reshape(25), align='edge', width=bar_width, color="tab:blue", label="45 pairs")
    plt.bar(indices + bar_width, scaled_time_data105.reshape(25), align='edge', width=bar_width, color='tab:green', label="105 pairs")
    plt.bar(indices + (bar_width*2), scaled_time_data210.reshape(25), align='edge', width=bar_width, color='tab:orange', label="190 pairs")
    plt.legend(loc='upper left')
    plt.title("average execution time")
    plt.xticks(indices + bar_width + (bar_width/2), bins)
    plt.show()
    plt.clf()

    plt.figure(figsize=(20, 10), dpi=300)
    plt.bar(indices, cases_np_45[:, 0].reshape(25), align='edge', width=bar_width, color="tab:blue", label="45 pairs")
    plt.bar(indices + bar_width, cases_np_105[:, 0].reshape(25), align='edge', width=bar_width, color='tab:green', label="105 pairs")
    plt.bar(indices + (bar_width * 2), cases_np_210[:, 0].reshape(25), align='edge', width=bar_width, color='tab:orange', label="190 pairs")
    plt.legend(loc='upper left')
    plt.title("average accuracy")
    plt.xticks(indices + bar_width + (bar_width / 2), bins)
    plt.show()






