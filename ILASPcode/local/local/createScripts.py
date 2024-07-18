import os
import numpy as np

max_v_list = [1, 2, 3, 4, 5]
max_p_list = [1, 2, 3, 4, 5]
list_of_user = [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]    # 0, 10, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 31, 41, 42, 43, 44
TrainCouples = [45, 105, 190]
stds = [1, 0.1, 0.01, 0.001]
no_zero = False

for USER in list_of_user:
    for TrainCouple in TrainCouples:
        for std in stds:
            if no_zero:
                NNoutput_dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "_gauss/std-" + str(std)
            else:
                NNoutput_dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "_gauss/std-" + str(std)
            f_couples = os.path.join(NNoutput_dir, 'couple' + str(USER) + '.txt')

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


            # no_zero_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"
            # zero_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"
            #
            # if no_zero:
            #     fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
            #     for k, couple in enumerate(couples):
            #         for max_v in max_v_list:
            #             for max_p in max_p_list:
            #                 if max_v == 1 and max_p == 5:
            #                         fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTrain/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
            #                 else:
            #                     continue
            #     fileToCreate.flush()
            #     fileToCreate.close()
            #
            # else:
            #     fileToCreate = open(zero_data_dir + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
            #     for k, couple in enumerate(couples):
            #         for max_v in max_v_list:
            #             for max_p in max_p_list:
            #                 if max_v == 1 and max_p == 5:
            #                         fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTrain/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
            #                 else:
            #                     continue
            #     fileToCreate.flush()
            #     fileToCreate.close()

            no_zero_data_dir_test = "./Data8Component2Std/final/users/no_zero/test/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"
            zero_data_dir_test = "./Data8Component2Std/final/users/zero/test/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"

            if no_zero:
                fileToCreate = open(no_zero_data_dir_test + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
                for k, couple in enumerate(couples):
                    for max_v in max_v_list:
                        for max_p in max_p_list:
                            if max_v == 1 and max_p == 5:
                                    fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/testFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTest/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
                            else:
                                continue
                fileToCreate.flush()
                fileToCreate.close()

            else:
                fileToCreate = open(zero_data_dir_test + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
                for k, couple in enumerate(couples):
                    for max_v in max_v_list:
                        for max_p in max_p_list:
                            if max_v == 1 and max_p == 5:
                                    fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/testFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTest/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
                            else:
                                continue
                fileToCreate.flush()
                fileToCreate.close()