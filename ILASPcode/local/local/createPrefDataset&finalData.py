import os
import sys
import numpy as np
import re
no_zero = True
if no_zero:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-no-zero/Train45"
else:
    NNoutput_dir = "Data8Component2Std/sampled-recipes-zero/Train45"

zero_dir_train = "Data8Component2Std/final/users/zero/train/45Couples/"
no_zero_dir_train = "./Data8Component2Std/final/users/no_zero/train/45Couples/"
zero_dir_test = "Data8Component2Std/final/users/zero/test/45CouplesForTrain45/"
no_zero_dir_test = "Data8Component2Std/final/users/no_zero/test/45CouplesForTrain45/"

output_dir_for_zero_train_data_dir = "./Data8Component2Std/final/users/zero/train/45Couples/"
output_dir_for_no_zero_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/45Couples/"
output_dir_for_zero_test_data_dir = "Data8Component2Std/final/users/zero/test/45CouplesForTrain45/"
output_dir_for_no_zero_test_data_dir = "Data8Component2Std/final/users/no_zero/test/45CouplesForTrain45/"

dir_labels_train = os.path.join(NNoutput_dir, 'user_prediction/train/')
dir_labels_test = os.path.join(NNoutput_dir, 'user_prediction/test/')
dir_distances_train = os.path.join(NNoutput_dir, 'distances/')
dir_las_files = os.path.join(NNoutput_dir, 'las_files/')
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

distances = np.zeros((len(couples), 45), dtype='float32')

for k, couple in enumerate(couples):
    f_distance = os.path.join(dir_distances_train, 'recipes_distances' + str(int(couple[0])) + '-' + str(int(couple[1])) + ".txt")
    fDistances = open(f_distance)
    dataDistances = fDistances.read()
    fDistances.close()

    linesOfDistances = dataDistances.split('\n')
    for i, line in enumerate(linesOfDistances):
        if line == '':
            continue
        values = [x for x in line.split('\n')[:]]
        for j, value in enumerate(values):
            if value == '':
                continue
            distances[k, i] = value

dict_of_las_files = {}

for k, couple in enumerate(couples):
    f_las_files = os.path.join(dir_las_files, 'recipes_sampled_' + str(int(couple[0])) + '-' + str(int(couple[1])) + ".las")
    fLasFiles = open(f_las_files)
    dataLasFiles = fLasFiles.read()
    fLasFiles.close()
    regex_string1 = "sampled" + str(int(couple[0])) + "-"
    regex_string2 = "sampled" + str(int(couple[1])) + "-"
    replace_string1 = "sampled" + str(int(couple[0])) + "s"
    replace_string2 = "sampled" + str(int(couple[1])) + "s"
    dataLasFiles = re.sub(regex_string1, replace_string1, dataLasFiles)
    dataLasFiles = re.sub(regex_string2, replace_string2, dataLasFiles)
    dict_of_las_files[str(int(couple[0])) + '-' + str(int(couple[1]))] = dataLasFiles

couple_label_train = np.zeros((10, 45, 45), dtype="int32")
couple_label_test = np.zeros((10, 45, 1), dtype="int32")
for u_counter, u in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
    for k, couple in enumerate(couples):
        f_label_train_file = os.path.join(dir_labels_train, 'user' + str(u) + "_Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + '.txt')
        fLabelTrainFile = open(f_label_train_file)
        dataLabelTrain = fLabelTrainFile.read()
        fLabelTrainFile.close()
        linesOfLabelTrain = dataLabelTrain.split('\n')
        for i, line in enumerate(linesOfLabelTrain):
            if line == '':
                continue
            values = [x for x in line.split('\n')[:]]
            for j, value in enumerate(values):
                if value == '':
                    continue
                couple_label_train[u_counter, k, i] = value

for u_counter, u in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
    for k, couple in enumerate(couples):
        f_label_test_file = os.path.join(dir_labels_test, 'user' + str(u) + "_Couple" + str(int(couple[0])) + '-' + str(int(couple[1])) + '.txt')
        fLabelTestFile = open(f_label_test_file)
        dataLabelTest = fLabelTestFile.read()
        fLabelTestFile.close()
        linesOfLabelTest = dataLabelTest.split('\n')
        for i, line in enumerate(linesOfLabelTest):
            if line == '':
                continue
            if i > 0:   # for error, in sampler I've written the same label 45/105/210 times, now I've corrected, but just to be sure I add this check
                break
            values = [x for x in line.split('\n')[:]]
            for j, value in enumerate(values):
                if value == '':
                    continue
                couple_label_test[u_counter, k, i] = value

macro_ingredients_dictionary = {0: "cereali",
                                1: "latticini",
                                2: "uova",
                                3: "farinacei",
                                4: "frutta",
                                5: "erbe_spezie_e_condimenti",
                                6: "carne",
                                7: "funghi_e_tartufi",
                                8: "pasta",
                                9: "pesce",
                                10: "dolcificanti",
                                11: "verdure_e_ortaggi"}


preparation_dictionary = {0: "bollitura",
                          1: "rosolatura",
                          2: "frittura",
                          3: "marinatura",
                          4: "mantecatura",
                          5: "forno",
                          6: "cottura_a_fiamma",
                          7: "stufato"}

choices = [0]
for u_counter, u in enumerate([15, 3, 32, 7, 36, 4, 20, 29, 14, 11]):
    for k, couple in enumerate(couples):
        dir_user_output_zero_train = zero_dir_train + "User" + str(u) + "/trainFiles/"
        dir_output_zero_train = output_dir_for_zero_train_data_dir + "User" + str(u) + "/outputTrain/"
        f_user_output_zero_train = os.path.join(dir_user_output_zero_train, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
        f_user_output_file_zero_train = os.path.join(dir_output_zero_train, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')

        dir_user_output_no_zero_train = no_zero_dir_train + "User" + str(u) + "/trainFiles/"
        dir_output_no_zero_train = output_dir_for_no_zero_train_data_dir + "User" + str(u) + "/outputTrain/"
        f_user_output_no_zero_train = os.path.join(dir_user_output_no_zero_train, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
        f_user_output_file_no_zero_train = os.path.join(dir_output_no_zero_train, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')

        dir_user_output_zero_test = zero_dir_test + "User" + str(u) + "/testFiles/"
        dir_output_zero_test = output_dir_for_zero_test_data_dir + "User" + str(u) + "/outputTest/"
        f_user_output_zero_test = os.path.join(dir_user_output_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
        f_user_output_file_zero_test = os.path.join(dir_output_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')

        dir_user_output_no_zero_test = no_zero_dir_test + "User" + str(u) + "/testFiles/"
        dir_output_no_zero_test = output_dir_for_no_zero_test_data_dir + "User" + str(u) + "/outputTest/"
        f_user_output_no_zero_test = os.path.join(dir_user_output_no_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.las')
        f_user_output_file_no_zero_test = os.path.join(dir_output_no_zero_test, 'Couple' + str(int(couple[0])) + '-' + str(int(couple[1])) + '-max_v=1-max_p=5.txt')


        if no_zero:
            f = open(f_user_output_no_zero_train, 'w+')
            sys.stdout = open(f_user_output_no_zero_train, 'w')
            print(dict_of_las_files[str(int(couple[0])) + '-' + str(int(couple[1]))])
            print("")
            print("#maxv(1).")
            print("#maxp(5).")
            print("")
            print("#modeo(1, value(const(val), var(val))).")
            print("#modeo(1, category(const(mg)), (positive)).")
            print("#weight(val).")
            print("#weight(1).")
            print("#weight(-1).")
            print("#constant(val, category).")
            print("#constant(val, cost).")
            print("#constant(val, difficulty).")
            print("#constant(val, prepTime).")
            print("#constant(mg, 1).")
            print("#constant(mg, 2).")
            print("#constant(mg, 3).")
            print("#constant(mg, 4).")
            print("#constant(mg, 5).")
            for key in macro_ingredients_dictionary.keys():
                print("#constant(val, " + macro_ingredients_dictionary[key] + ").")
            for key in preparation_dictionary.keys():
                print("#constant(val, " + preparation_dictionary[key] + ").")
            print("")
            counter = 0
            for j, sub_couple in enumerate(range(0, 45)):
                if couple_label_train[u_counter, k, j] == 1:
                    print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", <).")
                    counter = counter + 1
                elif couple_label_train[u_counter, k, j] == -1:
                    print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", >).")
                    counter = counter + 1
                else:
                    continue
            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_file_no_zero_train, 'w+')
            sys.stdout = open(f_user_output_file_no_zero_train, 'w')
            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_no_zero_test, 'w+')
            sys.stdout = open(f_user_output_no_zero_test, 'w')
            print(dict_of_las_files[str(int(couple[0])) + '-' + str(int(couple[1]))])
            print("")
            print("#maxv(1).")
            print("#maxp(5).")
            print("")
            print("#modeo(1, value(const(val), var(val))).")
            print("#modeo(1, category(const(mg)), (positive)).")
            print("#weight(val).")
            print("#weight(1).")
            print("#weight(-1).")
            print("#constant(val, category).")
            print("#constant(val, cost).")
            print("#constant(val, difficulty).")
            print("#constant(val, prepTime).")
            print("#constant(mg, 1).")
            print("#constant(mg, 2).")
            print("#constant(mg, 3).")
            print("#constant(mg, 4).")
            print("#constant(mg, 5).")
            for key in macro_ingredients_dictionary.keys():
                print("#constant(val, " + macro_ingredients_dictionary[key] + ").")
            for key in preparation_dictionary.keys():
                print("#constant(val, " + preparation_dictionary[key] + ").")
            print("")
            counter = 0
            if couple_label_test[u_counter, k, 0] == 1:
                print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", <).")
                counter = counter + 1
            elif couple_label_test[u_counter, k, 0] == -1:
                print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", >).")
                counter = counter + 1
            else:
                continue

            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_file_no_zero_test, 'w+')
            sys.stdout = open(f_user_output_file_no_zero_test, 'w')
            sys.stdout = sys.__stdout__
            f.close()
        else:
            f = open(f_user_output_zero_train, 'w+')
            sys.stdout = open(f_user_output_zero_train, 'w')
            print(dict_of_las_files[str(int(couple[0])) + '-' + str(int(couple[1]))])
            print("")
            print("#maxv(1).")
            print("#maxp(5).")
            print("")
            print("#modeo(1, value(const(val), var(val))).")
            print("#modeo(1, category(const(mg)), (positive)).")
            print("#weight(val).")
            print("#weight(1).")
            print("#weight(-1).")
            print("#constant(val, category).")
            print("#constant(val, cost).")
            print("#constant(val, difficulty).")
            print("#constant(val, prepTime).")
            print("#constant(mg, 1).")
            print("#constant(mg, 2).")
            print("#constant(mg, 3).")
            print("#constant(mg, 4).")
            print("#constant(mg, 5).")
            for key in macro_ingredients_dictionary.keys():
                print("#constant(val, " + macro_ingredients_dictionary[key] + ").")
            for key in preparation_dictionary.keys():
                print("#constant(val, " + preparation_dictionary[key] + ").")
            print("")
            counter = 0
            for j, sub_couple in enumerate(range(0, 45)):
                if couple_label_train[u_counter, k, j] == 1:
                    print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", <).")
                    counter = counter + 1
                elif couple_label_train[u_counter, k, j] == -1:
                    print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", >).")
                    counter = counter + 1
                else:
                    print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", =).")
                    counter = counter + 1
            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_file_zero_train, 'w+')
            sys.stdout = open(f_user_output_file_zero_train, 'w')
            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_zero_test, 'w+')
            sys.stdout = open(f_user_output_zero_test, 'w')
            print(dict_of_las_files[str(int(couple[0])) + '-' + str(int(couple[1]))])
            print("")
            print("#maxv(1).")
            print("#maxp(5).")
            print("")
            print("#modeo(1, value(const(val), var(val))).")
            print("#modeo(1, category(const(mg)), (positive)).")
            print("#weight(val).")
            print("#weight(1).")
            print("#weight(-1).")
            print("#constant(val, category).")
            print("#constant(val, cost).")
            print("#constant(val, difficulty).")
            print("#constant(val, prepTime).")
            print("#constant(mg, 1).")
            print("#constant(mg, 2).")
            print("#constant(mg, 3).")
            print("#constant(mg, 4).")
            print("#constant(mg, 5).")
            for key in macro_ingredients_dictionary.keys():
                print("#constant(val, " + macro_ingredients_dictionary[key] + ").")
            for key in preparation_dictionary.keys():
                print("#constant(val, " + preparation_dictionary[key] + ").")
            print("")
            counter = 0
            if couple_label_test[u_counter, k, 0] == 1:
                print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", <).")
                counter = counter + 1
            elif couple_label_test[u_counter, k, 0] == -1:
                print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", >).")
                counter = counter + 1
            else:
                print("#brave_ordering(o" + str(counter + 1) + "@" + str(int(distances[k, j])) + ", sampled" + str(int(couple[0])) + "s" + str(j) + ", sampled" + str(int(couple[1])) + "s" + str(j) + ", =).")
                counter = counter + 1

            sys.stdout = sys.__stdout__
            f.close()

            f = open(f_user_output_file_zero_test, 'w+')
            sys.stdout = open(f_user_output_file_zero_test, 'w')
            sys.stdout = sys.__stdout__
            f.close()


