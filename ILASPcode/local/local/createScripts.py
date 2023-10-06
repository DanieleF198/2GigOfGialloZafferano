import os
import numpy as np

max_v_list = [1, 2, 3, 4, 5]
max_p_list = [1, 2, 3, 4, 5]
list_of_user = [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]
no_zero = False
if no_zero:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-no-zero/Train45/"
else:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-zero/Train45/"


f_couples = os.path.join(NNoutput_dir, 'couple.txt')

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

no_zero_data_dir = "./Data8Component2Std/final/users/no_zero/train/45Couples/"
zero_data_dir = "./Data8Component2Std/final/users/zero/train/45Couples/"

if no_zero:
    for USER in list_of_user:
        fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
        for k, couple in enumerate(couples):
            for max_v in max_v_list:
                for max_p in max_p_list:
                    if max_v == 1 and max_p == 5:
                            fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTrain/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
                    else:
                        continue
        fileToCreate.flush()
        fileToCreate.close()

else:
    for USER in list_of_user:
        fileToCreate = open(zero_data_dir + "script_ilasp_for_test_user" + str(USER) + ".sh", "w+")
        for k, couple in enumerate(couples):
            for max_v in max_v_list:
                for max_p in max_p_list:
                    if max_v == 1 and max_p == 5:
                            fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".las > ./User" + str(USER) + "/outputTrain/Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + "-max_v=" + str(max_v) + "-max_p=" + str(max_p) + ".txt;\n")
                    else:
                        continue
        fileToCreate.flush()
        fileToCreate.close()