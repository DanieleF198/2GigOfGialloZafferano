import os
import sys
import numpy as np
import re
no_zero = False
import ilaspReadWriteUtils as ilasp


TrainCouples = [45, 105, 190]
stds = [0.1]
transferTestStr = ""
for u_counter, u in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
    for TrainCouple in TrainCouples:
        for std in stds:
            if no_zero:
                NNoutput_dir = "Data/sampled-recipes-no-zero/Train" + str(TrainCouple) + "_gauss/std-" + str(std)
            else:
                NNoutput_dir = "Data/sampled-recipes-zero/Train" + str(TrainCouple) + "_gauss/std-" + str(std)
            dir_labels_test = os.path.join(NNoutput_dir, 'user_prediction/test/')

            zero_dir_test = "Data/final/users/zero/test/" + transferTestStr + "/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"
            no_zero_dir_test = "Data/final/users/no_zero/test/" + transferTestStr + "/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"

            output_dir_for_zero_test_data_dir = "Data/final/users/zero/test/" + transferTestStr + "/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"
            output_dir_for_no_zero_test_data_dir = "Data/final/users/no_zero/test/" + transferTestStr + "/" + str(TrainCouple) + "Couples_gauss_std" + str(std) + "/"

            f_couples = os.path.join(NNoutput_dir, 'couple' + str(u) + '.txt')

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

            couple_label_test = np.zeros((10, 100, 1), dtype="int32")
            for k, couple in enumerate(couples):
                f_label_test_file = os.path.join(dir_labels_test, 'user' + str(u) + "_Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + '.txt')
                fLabelTestFile = open(f_label_test_file)
                dataLabelTest = fLabelTestFile.read()
                fLabelTestFile.close()
                linesOfLabelTest = dataLabelTest.split('\n')
                for i, line in enumerate(linesOfLabelTest):
                    if line == '':
                        continue
                    if i > 0:  # for error, in sampler I've written the same label 45/105/210 times, now I've corrected, but just to be sure I add this check
                        break
                    values = [x for x in line.split('\n')[:]]
                    for j, value in enumerate(values):
                        if value == '':
                            continue
                        couple_label_test[u_counter, k, i] = value

            items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(1)-max_p(5).las")
            language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(1)-max_p(5).las")

            for k, couple in enumerate(couples):
                dir_user_output_zero_test = zero_dir_test + "User" + str(u) + "/testFiles/"
                dir_output_zero_test = output_dir_for_zero_test_data_dir + "User" + str(u) + "/outputTest/"
                f_user_output_zero_test = os.path.join(dir_user_output_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
                f_user_output_file_zero_test = os.path.join(dir_output_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')

                dir_user_output_no_zero_test = no_zero_dir_test + "User" + str(u) + "/testFiles/"
                dir_output_no_zero_test = output_dir_for_no_zero_test_data_dir + "User" + str(u) + "/outputTest/"
                f_user_output_no_zero_test = os.path.join(dir_user_output_no_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
                f_user_output_file_no_zero_test = os.path.join(dir_output_no_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')

                test_output_file = open(f_user_output_zero_test, "w+")

                test_output_file.write(ilasp.itemsToPos(items) + "\n")
                test_output_file.write(language_bias)

                if couple_label_test[u_counter, k, 0] == 1:
                    test_output_file.write("#brave_ordering(o0@1,item" + str(int(couple[0])) + ",item" + str(int(couple[1])) + ",<).")
                elif couple_label_test[u_counter, k, 0] == -1:
                    test_output_file.write("#brave_ordering(o0@1,item" + str(int(couple[0])) + ",item" + str(int(couple[1])) + ",>).")
                else:
                    test_output_file.write("#brave_ordering(o0@1,item" + str(int(couple[0])) + ",item" + str(int(couple[1])) + ",=).")
                test_output_file.flush()
                test_output_file.close()

                test_output_file_txt = open(f_user_output_file_zero_test, "w+")
                test_output_file_txt.flush()
                test_output_file_txt.close()
