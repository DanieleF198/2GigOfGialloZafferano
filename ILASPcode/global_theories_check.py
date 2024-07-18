import numpy as np

sizes = [45, 105, 210]
datas = ["Data8Component2Std", "Data", "Data17Component2Std"]
users = [3, 4, 7, 11, 14, 15, 20, 29, 32, 36]
PCA_approach = ["indirect", "direct"]
PCS = [5, 10, 15, 20]
for approach in PCA_approach:
    if approach == "indirect":
        for data in datas:
            for size in sizes:
                list_empty = []
                for user in users:
                    counter_empty = 0
                    filename_theory = './' + str(data) + '/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                    f_local_output = open(filename_theory)
                    local_output = f_local_output.read()
                    f_local_output.close()
                    theory = ""
                    for line in local_output.split("\n"):
                        if ':~' not in line:
                            continue
                        else:
                            theory += line
                    if theory == "":
                        counter_empty += 1
                list_empty.append(counter_empty)
                mean_of_empty = np.mean(list_empty)
                print("[" + str(data) + " ~ indirect PCA] - size = " + str(size) + "; mean of empty theory = " + str(mean_of_empty))
    else:
        print("------------------------------------------------------------------------------------")
        for PC in PCS:
            for size in sizes:
                list_empty = []
                for user in users:
                    counter_empty = 0
                    filename_theory = './PCAexperiment/final' + str(PC) + '/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-p(5)-max_p(5).txt'
                    f_local_output = open(filename_theory)
                    local_output = f_local_output.read()
                    f_local_output.close()
                    theory = ""
                    for line in local_output.split("\n"):
                        if ':~' not in line:
                            continue
                        else:
                            theory += line
                    if theory == "":
                        counter_empty += 1
                list_empty.append(counter_empty)
                mean_of_empty = np.mean(list_empty)
                print("[" + str(PC) + "PC ~ direct PCA] - size = " + str(size) + "; mean of empty theory = " + str(mean_of_empty))