import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

choices = [0, 1]
users = [i for i in range(0, 48)]
data_45_couples_no_zeros = np.zeros((48, 5*5, 4), dtype='float32')  # we have more answer, but at this point we are only interested in the ten users for testing, which are in the first 48
data_105_couples_no_zeros = np.zeros((48, 5*5, 4), dtype='float32')
data_210_couples_no_zeros = np.zeros((48, 5*5, 4), dtype='float32')
data_150_couples_no_zeros = np.zeros((48, 5*5, 4), dtype='float32')
list_theory_45_couple = []
list_theory_105_couple = []
list_theory_210_couple = []
list_theory_150_couple = []

PCAindexes = [5, 10, 15, 20]
scopes = ["", "_original"]

for choice in choices:
    for scope in scopes:
        for PCAindex in PCAindexes:
            if choice == 1:
                path = './PCAexperiment/testOutput' + scope + str(PCAindex) + '/results_zero.csv'
            else:
                continue
                # temporally
                # path = './PCAexperiment/testOutput/results_no_zero(variant).csv'
            for user in users:
                parameter_base = 0
                test_not_inserted = True
                with open(path, newline='\n') as csvFile:
                    reader = csv.reader(csvFile, delimiter=";")
                    temp_base = 9999999
                    for i, row in enumerate(reader):
                        if i == 0:
                            continue
                        else:
                            if scope == "_original":
                                if temp_base == 9999999:
                                    temp_base = int(row[1])
                                elif temp_base != int(row[1]):
                                    temp_base = int(row[1])
                                    parameter_base += 1
                                    if parameter_base == 5:
                                        parameter_base = 0
                                if int(row[0]) == user:
                                    data_150_couples_no_zeros[user][(parameter_base * 5) + (int(row[2])) - 1][0] = float(row[6])  # accuracy_percentage
                                    data_150_couples_no_zeros[user][(parameter_base * 5) + (int(row[2])) - 1][1] = float(row[7])  # precision_percentage
                                    data_150_couples_no_zeros[user][(parameter_base * 5) + (int(row[2])) - 1][2] = float(row[8])  # recall_percentage
                                    data_150_couples_no_zeros[user][(parameter_base * 5) + (int(row[2])) - 1][3] = float(row[9])  # train_time
                                    list_theory_150_couple.append(row[10])  # theory
                            else:
                                if temp_base == 9999999:
                                    temp_base = int(row[1])
                                elif temp_base != int(row[1]):
                                    temp_base = int(row[1])
                                    parameter_base += 1
                                    if parameter_base == 5:
                                        parameter_base = 0
                                if int(row[4]) <= 45:    # 45CouplesCase
                                    if int(row[0]) == user:
                                        data_45_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][0] = float(row[6])       # accuracy_percentage
                                        data_45_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][1] = float(row[7])       # precision_percentage
                                        data_45_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][2] = float(row[8])       # recall_percentage
                                        data_45_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][3] = float(row[9])      # train_time
                                        list_theory_45_couple.append(row[10])                                  # theory
                                elif (int(row[4]) > 45) and (int(row[4]) <= 105):    # 45CouplesCase
                                    if int(row[0]) == user:
                                        data_105_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][0] = float(row[6])       # accuracy_percentage
                                        data_105_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][1] = float(row[7])       # precision_percentage
                                        data_105_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][2] = float(row[8])       # recall_percentage
                                        data_105_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][3] = float(row[9])      # train_time
                                        list_theory_105_couple.append(row[10])                                  # theory
                                elif (int(row[4]) > 105) and (int(row[4]) <= 210):
                                    if int(row[0]) == user:
                                        data_210_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][0] = float(row[6])       # accuracy_percentage
                                        data_210_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][1] = float(row[7])       # precision_percentage
                                        data_210_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][2] = float(row[8])       # recall_percentage
                                        data_210_couples_no_zeros[user][(parameter_base*5) + (int(row[2])) - 1][3] = float(row[9])      # train_time
                                        list_theory_210_couple.append(row[10])                                  # theory


            if scope == "_original":
                user_counter = 0
                insert_counter = 0
                final_accuracy_percentages = np.zeros((10, 25))
                final_precision_percentages = np.zeros((10, 25))
                final_recall_percentages = np.zeros((10, 25))
                final_training_times = np.zeros((10, 25))
                for user_matrix_150 in data_150_couples_no_zeros:
                    if user_counter not in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
                        user_counter += 1
                        continue
                    accuracy_percentages = user_matrix_150[:, 0]
                    precision_percentages = user_matrix_150[:, 1]
                    recall_percentages = user_matrix_150[:, 2]
                    train_times = user_matrix_150[:, 3]
                    final_accuracy_percentages[insert_counter] = accuracy_percentages
                    final_precision_percentages[insert_counter] = precision_percentages
                    final_recall_percentages[insert_counter] = recall_percentages
                    final_training_times[insert_counter] = train_times
                    insert_counter += 1
                    user_counter += 1
            else:
                user_counter = 0
                insert_counter = 0
                final_accuracy_percentages = np.zeros((10, 75))
                final_precision_percentages = np.zeros((10, 75))
                final_recall_percentages = np.zeros((10, 75))
                final_training_times = np.zeros((10, 75))
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
            parameters = ["maxv= " + str(i) + "; maxp=" + str(j) for i in range(1, 6) for j in range(1, 6)]
            parameters_for_graph = [str(i) + ";" + str(j) for i in range(1, 6) for j in range(1, 6)]
            print("direct PCA: " + str(PCAindex) + "; type: " + scope)
            print("considered user id: 15, 3, 32, 7, 36, 4, 20, 29, 14, 11")
            # print("considered user id: all")
            if scope == "_original":
                print("test size: 50")
            else:
                print("test size: 105")
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
            mean_of_accuracy_of_all_150 = []
            mean_of_precision_of_all_150 = []
            mean_of_recall_of_all_150 = []
            mean_of_training_time_of_all_150 = []
            if scope == "_original":
                average_accuracy_percentages_150 = np.zeros((25,), dtype="float32")
                average_precision_percentages_150 = np.zeros((25,), dtype="float32")
                average_recall_percentages_150 = np.zeros((25,), dtype="float32")
                average_execution_time_150 = np.zeros((25,), dtype="float32")
                for parameter_counter in range(0, 25):
                    mean_of_accuracy = np.mean(final_accuracy_percentages_T[parameter_counter])
                    mean_of_precision = np.mean(final_precision_percentages_T[parameter_counter])
                    mean_of_recall = np.mean(final_recall_percentages_T[parameter_counter])
                    mean_of_training_time = np.mean(final_training_times_T[parameter_counter])
                    mean_of_accuracy_of_all_150.append(np.mean(final_accuracy_percentages_T[parameter_counter]))
                    mean_of_precision_of_all_150.append(np.mean(final_precision_percentages_T[parameter_counter]))
                    mean_of_recall_of_all_150.append(np.mean(final_recall_percentages_T[parameter_counter]))
                    mean_of_training_time_of_all_150.append(np.mean(final_training_times_T[parameter_counter]))
                    average_accuracy_percentages_150[parameter_counter] = mean_of_accuracy
                    average_precision_percentages_150[parameter_counter] = mean_of_precision
                    average_recall_percentages_150[parameter_counter] = mean_of_recall
                    average_execution_time_150[parameter_counter] = mean_of_training_time
                    print("On Dataset_150_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
                print("")
                fig, axs = plt.subplots(2, 2, figsize=(25, 15))
                fig.suptitle('dataset 150, average performances and training times (max_v;max_p)')
                axs[0, 0].set_title('average accuracy')
                axs[0, 1].set_title('average precision')
                axs[1, 0].set_title('average recall')
                axs[1, 1].set_title('average execution time')
                sb.barplot(ax=axs[0, 0], y=average_accuracy_percentages_150, x=parameters_for_graph, facecolor=(31 / 255, 119 / 255, 180 / 255, 1.000))
                sb.barplot(ax=axs[0, 1], y=average_precision_percentages_150, x=parameters_for_graph, facecolor=(31 / 255, 119 / 255, 180 / 255, 1.000))
                sb.barplot(ax=axs[1, 0], y=average_recall_percentages_150, x=parameters_for_graph, facecolor=(31 / 255, 119 / 255, 180 / 255, 1.000))
                sb.barplot(ax=axs[1, 1], y=average_execution_time_150, x=parameters_for_graph, facecolor=(31 / 255, 119 / 255, 180 / 255, 1.000))
                plt.savefig("./PCAexperiment/all_parameters_variation_graphs/original/" + str(PCAindex) + "/dataset150.jpg", dpi=300)
            else:
                for dataset_counter in range(0,3):
                    if dataset_counter == 0:
                        average_accuracy_percentages_45 = np.zeros((25,), dtype="float32")
                        average_precision_percentages_45 = np.zeros((25,), dtype="float32")
                        average_recall_percentages_45 = np.zeros((25,), dtype="float32")
                        average_execution_time_45 = np.zeros((25,), dtype="float32")
                        for parameter_counter in range(0, 25):
                            mean_of_accuracy = np.mean(final_accuracy_percentages_T[parameter_counter])
                            mean_of_precision = np.mean(final_precision_percentages_T[parameter_counter])
                            mean_of_recall = np.mean(final_recall_percentages_T[parameter_counter])
                            mean_of_training_time = np.mean(final_training_times_T[parameter_counter])
                            mean_of_accuracy_of_all_45.append(np.mean(final_accuracy_percentages_T[parameter_counter]))
                            mean_of_precision_of_all_45.append(np.mean(final_precision_percentages_T[parameter_counter]))
                            mean_of_recall_of_all_45.append(np.mean(final_recall_percentages_T[parameter_counter]))
                            mean_of_training_time_of_all_45.append(np.mean(final_training_times_T[parameter_counter]))
                            average_accuracy_percentages_45[parameter_counter] = mean_of_accuracy
                            average_precision_percentages_45[parameter_counter] = mean_of_precision
                            average_recall_percentages_45[parameter_counter] = mean_of_recall
                            average_execution_time_45[parameter_counter] = mean_of_training_time
                            print("On Dataset_45_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
                        print("")
                        fig, axs = plt.subplots(2, 2, figsize=(25, 15))
                        fig.suptitle('dataset 45, average performances and training times (max_v;max_p)')
                        axs[0, 0].set_title('average accuracy')
                        axs[0, 1].set_title('average precision')
                        axs[1, 0].set_title('average recall')
                        axs[1, 1].set_title('average execution time')
                        sb.barplot(ax=axs[0, 0], y=average_accuracy_percentages_45, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[0, 1], y=average_precision_percentages_45, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1 ,0], y=average_recall_percentages_45, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1, 1], y=average_execution_time_45, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        plt.savefig("./PCAexperiment/all_parameters_variation_graphs/sampled/" + str(PCAindex) + "/dataset45.jpg", dpi=300)
                    elif dataset_counter == 1:
                        average_accuracy_percentages_105 = np.zeros((25,), dtype="float32")
                        average_precision_percentages_105 = np.zeros((25,), dtype="float32")
                        average_recall_percentages_105 = np.zeros((25,), dtype="float32")
                        average_execution_time_105 = np.zeros((25,), dtype="float32")
                        for parameter_counter in range(0, 25):
                            mean_of_accuracy = np.mean(final_accuracy_percentages_T[25+parameter_counter])
                            mean_of_precision = np.mean(final_precision_percentages_T[25+parameter_counter])
                            mean_of_recall = np.mean(final_recall_percentages_T[25+parameter_counter])
                            mean_of_training_time = np.mean(final_training_times_T[25+parameter_counter])
                            mean_of_accuracy_of_all_105.append(np.mean(final_accuracy_percentages_T[25 + parameter_counter]))
                            mean_of_precision_of_all_105.append(np.mean(final_precision_percentages_T[25 + parameter_counter]))
                            mean_of_recall_of_all_105.append(np.mean(final_recall_percentages_T[25 + parameter_counter]))
                            mean_of_training_time_of_all_105.append(np.mean(final_training_times_T[25 + parameter_counter]))
                            average_accuracy_percentages_105[parameter_counter] = mean_of_accuracy
                            average_precision_percentages_105[parameter_counter] = mean_of_precision
                            average_recall_percentages_105[parameter_counter] = mean_of_recall
                            average_execution_time_105[parameter_counter] = mean_of_training_time
                            print("On Dataset_105_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + "; mean of training time " + str(mean_of_training_time))
                        print("")
                        fig, axs = plt.subplots(2, 2, figsize=(25, 15))
                        fig.suptitle('dataset 105, average performances and training times (max_v;max_p)')
                        axs[0, 0].set_title('average accuracy')
                        axs[0, 1].set_title('average precision')
                        axs[1, 0].set_title('average recall')
                        axs[1, 1].set_title('average execution time')
                        sb.barplot(ax=axs[0, 0], y=average_accuracy_percentages_105, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[0, 1], y=average_precision_percentages_105, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1 ,0], y=average_recall_percentages_105, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1, 1], y=average_execution_time_105, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        plt.savefig("./PCAexperiment/all_parameters_variation_graphs/sampled/" + str(PCAindex) + "/dataset105.jpg", dpi=300)
                    else:
                        average_accuracy_percentages_210 = np.zeros((25,), dtype="float32")
                        average_precision_percentages_210 = np.zeros((25,), dtype="float32")
                        average_recall_percentages_210 = np.zeros((25,), dtype="float32")
                        average_execution_time_210 = np.zeros((25,), dtype="float32")
                        for parameter_counter in range(0, 25):
                            mean_of_accuracy = np.mean(final_accuracy_percentages_T[50 + parameter_counter])
                            mean_of_precision = np.mean(final_precision_percentages_T[50 + parameter_counter])
                            mean_of_recall = np.mean(final_recall_percentages_T[50 + parameter_counter])
                            mean_of_training_time = np.mean(final_training_times_T[50 + parameter_counter])
                            mean_of_accuracy_of_all_210.append(np.mean(final_accuracy_percentages_T[50 + parameter_counter]))
                            mean_of_precision_of_all_210.append(np.mean(final_precision_percentages_T[50 + parameter_counter]))
                            mean_of_recall_of_all_210.append(np.mean(final_recall_percentages_T[50 + parameter_counter]))
                            mean_of_training_time_of_all_210.append(np.mean(final_training_times_T[50 + parameter_counter]))
                            average_accuracy_percentages_210[parameter_counter] = mean_of_accuracy
                            average_precision_percentages_210[parameter_counter] = mean_of_precision
                            average_recall_percentages_210[parameter_counter] = mean_of_recall
                            average_execution_time_210[parameter_counter] = mean_of_training_time
                            print("On Dataset_190_couples with " + parameters[parameter_counter] + " mean of accuracy " + str(mean_of_accuracy) + "; mean of precision " + str(mean_of_precision) + "; mean of recall " + str(mean_of_recall) + ";  mean of training time " + str(mean_of_training_time))
                        print("")
                        fig, axs = plt.subplots(2, 2, figsize=(25, 15))
                        fig.suptitle('dataset 190, average performances and training times (max_v;max_p)')
                        axs[0, 0].set_title('average accuracy')
                        axs[0, 1].set_title('average precision')
                        axs[1, 0].set_title('average recall')
                        axs[1, 1].set_title('average execution time')
                        sb.barplot(ax=axs[0, 0], y=average_accuracy_percentages_210, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[0, 1], y=average_precision_percentages_210, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1 ,0], y=average_recall_percentages_210, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        sb.barplot(ax=axs[1, 1], y=average_execution_time_210, x=parameters_for_graph, facecolor=(31/255, 119/255, 180/255, 1.000))
                        plt.savefig("./PCAexperiment/all_parameters_variation_graphs/sampled/" + str(PCAindex) + "/dataset210.jpg", dpi=300)
                index_max_45 = np.argmax(average_accuracy_percentages_45)
                print("for dataset 45 best accuracy obtained when (max_v;max_p)=" + str(parameters_for_graph[index_max_45]) + " with: accuracy = " + str(average_accuracy_percentages_45[index_max_45]) + "; precision = " + str(average_precision_percentages_45[index_max_45]) + "; recall = " + str(average_recall_percentages_45[index_max_45]) + "; execution time = " + str(average_execution_time_45[index_max_45]))
                index_max_105 = np.argmax(average_accuracy_percentages_105)
                print("for dataset 105 best accuracy obtained when (max_v;max_p)=" + str(parameters_for_graph[index_max_105]) + " with: accuracy = " + str(average_accuracy_percentages_105[index_max_105]) + "; precision = " + str(average_precision_percentages_105[index_max_105]) + "; recall = " + str(average_recall_percentages_105[index_max_105]) + "; execution time = " + str(average_execution_time_105[index_max_105]))
                index_max_210 = np.argmax(average_accuracy_percentages_210)
                print("for dataset 210 best accuracy obtained when (max_v;max_p)=" + str(parameters_for_graph[index_max_210]) + " with: accuracy = " + str(average_accuracy_percentages_210[index_max_210]) + "; precision = " + str(average_precision_percentages_210[index_max_210]) + "; recall = " + str(average_recall_percentages_210[index_max_210]) + "; execution time = " + str(average_execution_time_210[index_max_210]))
            # wc_number_case45 = []
            # wc_number_case105 = []
            # wc_number_case210 = []
            #
            # for theory in list_theory_45_couple:
            #     wc_number_case45.append(len(theory.split(":~")))
            #
            # for theory in list_theory_105_couple:
            #     wc_number_case105.append(len(theory.split(":~")))
            #
            # for theory in list_theory_210_couple:
            #     wc_number_case210.append(len(theory.split(":~")))
            #
            # mean_wc_numer_case45 = np.mean(wc_number_case45)
            # mean_wc_numer_case105 = np.mean(wc_number_case105)
            # mean_wc_numer_case210 = np.mean(wc_number_case210)


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
            # print("")
            # print("On dataset with 190 couple")
            # print("Train size = 190")
            # print("accuracy = " + str(np.mean(mean_of_accuracy_of_all_210)))
            # print("precision = " + str(np.mean(mean_of_precision_of_all_210)))
            # print("recall = " + str(np.mean(mean_of_recall_of_all_210)))
            # print("training time = " + str(np.mean(mean_of_training_time_of_all_210)))
            # print("mean number of wc = " + str(mean_wc_numer_case210))
