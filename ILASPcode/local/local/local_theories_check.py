import numpy as np

sizes = [45, 105, 190]
stds = [1, 0.1, 0.01, 0.001]
datas = ["Data8Component2Std", "Data", "Data17Component2Std"]
users = [3, 4, 7, 11, 14, 15, 20, 29, 32, 36]
for data in datas:
    for size in sizes:
        for std in stds:
            if size == 190 and std == 1:
                continue
            if data != "Data8Component2Std":
                if size != 105:
                    continue
                if std != 0.1:
                    continue
            list_empty = []
            equal_symbol = []
            greater_symbol = []
            lesser_symbol = []
            complementary_equal_symbol = []
            complementary_greater_symbol = []
            complementary_lesser_symbol = []
            for user in users:
                counter_equal_symbol = 0
                counter_greater_symbol = 0
                counter_lesser_symbol = 0
                counter_empty = 0
                counter_complementary_equal_symbol = 0
                counter_complementary_greater_symbol = 0
                counter_complementary_lesser_symbol = 0
                f_couples = "./" + str(data) + "/sampled-recipes-zero/Train" + str(size) + "_gauss/std-" + str(std) + "/couple" + str(user) + ".txt"

                fCouples = open(f_couples)
                dataCouples = fCouples.read()
                fCouples.close()

                linesOfCouples = dataCouples.split('\n')
                couples = np.zeros((len(linesOfCouples), 2), dtype='float32')
                for i, line in enumerate(linesOfCouples):
                    if line == '':
                        continue
                    values = [x for x in line.split(';')[:]]
                    for j, value in enumerate(values):
                        if value == '':
                            continue
                        couples[i, j] = value

                couples = couples[:-1]

                for couple in couples:
                    filename_train_file = './' + str(data) + '/final/users/zero/train/' + str(size) + 'Couples_gauss_std' + str(std) + '/User' + str(user) + '/trainFiles/Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las'
                    filename_theory = './' + str(data) + '/final/users/zero/train/' + str(size) + 'Couples_gauss_std' + str(std) + '/User' + str(user) + '/outputTrain/Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt'
                    f_local_train = open(filename_train_file)
                    local_train = f_local_train.read()
                    f_local_train.close()
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
                        for line in local_train.split("\n"):
                            if "=" in line:
                                counter_equal_symbol += 1
                            elif ">" in line:
                                counter_greater_symbol += 1
                            elif "<" in line:
                                counter_lesser_symbol += 1
                    else:
                        for line in local_train.split("\n"):
                            if "=" in line:
                                counter_complementary_equal_symbol += 1
                            elif ">" in line:
                                counter_complementary_greater_symbol += 1
                            elif "<" in line:
                                counter_complementary_lesser_symbol += 1

                list_empty.append(counter_empty)
                equal_symbol.append(counter_equal_symbol)
                greater_symbol.append(counter_greater_symbol)
                lesser_symbol.append(counter_lesser_symbol)
                complementary_equal_symbol.append(counter_complementary_equal_symbol)
                complementary_greater_symbol.append(counter_complementary_greater_symbol)
                complementary_lesser_symbol.append(counter_complementary_lesser_symbol)


            mean_of_empty = np.mean(list_empty)
            mean_of_equal = np.mean(equal_symbol)
            mean_of_greater = np.mean(greater_symbol)
            mean_of_lesser = np.mean(lesser_symbol)
            d_means = mean_of_equal + mean_of_greater + mean_of_lesser
            mean_of_complementary_equal = np.mean(complementary_equal_symbol)
            mean_of_complementary_greater = np.mean(complementary_greater_symbol)
            mean_of_complementary_lesser = np.mean(complementary_lesser_symbol)
            d_mean_complementary = mean_of_complementary_equal + mean_of_complementary_greater + mean_of_complementary_lesser
            # print("")
            # print("---------------------------------------------------------------------------------------------------------------------------")
            # print("")
            print("[" + str(data) + "] - size = " + str(size) + "; std = " + str(std) + "; empty theories = " + str(mean_of_empty) + "%; mean of [>, =, <] = [" + str(np.round((mean_of_greater/d_means)*100, 2)) + "%, " + str(np.round((mean_of_equal/d_means)*100, 2)) + "%, " + str(str(np.round((mean_of_lesser/d_means)*100, 2))) + "%] (VS [" + str(np.round((mean_of_complementary_greater/d_mean_complementary)*100, 2)) + "%, " + str(np.round((mean_of_complementary_equal/d_mean_complementary)*100, 2)) + "%, " + str(np.round((mean_of_complementary_lesser/d_mean_complementary)*100, 2)) + "%] when theory is not empty)")
            # print("")
            # print("---------------------------------------------------------------------------------------------------------------------------")
            # print("")
