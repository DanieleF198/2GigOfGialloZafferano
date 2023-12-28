import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pref(L, O):
    PrefMat = np.zeros((len(L), len(L)))
    LockMat = np.zeros((len(L), len(L)))
    for a in L:
        for b in L:
            aindex = np.where(L == a)[0][0]
            bindex = np.where(L == b)[0][0]
            for o in O:
                if a in o and b in o:
                    if np.where(o == a)[0][0] < np.where(o == b)[0][0]:
                        if LockMat[aindex, bindex] == 0:
                            LockMat[aindex, bindex] = 1
                            PrefMat[aindex, bindex] = 1
                            LockMat[bindex, aindex] = 1
                            PrefMat[bindex, aindex] = -1
                        elif PrefMat[aindex, bindex] == -1:
                            LockMat[aindex, bindex] = 1
                            PrefMat[aindex, bindex] = 0
                            LockMat[bindex, aindex] = 1
                            PrefMat[bindex, aindex] = 0
    change = 1
    iteratore = 2
    while change == 1:
        change = 0
        for a in L:
            for b in L:
                for c in L:
                    aindex = np.where(L == a)[0][0]
                    bindex = np.where(L == b)[0][0]
                    cindex = np.where(L == c)[0][0]
                    if PrefMat[aindex, bindex] > 0 and PrefMat[bindex, cindex] > 0:
                        if LockMat[aindex, cindex] == 0:
                            PrefMat[aindex, cindex] = 1
                            PrefMat[cindex, aindex] = -1
                            LockMat[aindex, cindex] = iteratore
                            LockMat[cindex, aindex] = iteratore
                            change = 1
                        elif LockMat[aindex, cindex] == iteratore and PrefMat[aindex, cindex] == -1:
                            PrefMat[aindex, cindex] = 0
                            PrefMat[cindex, aindex] = 0
                            LockMat[aindex, cindex] = iteratore
                            LockMat[cindex, aindex] = iteratore
                            change = 1
        iteratore = iteratore + 1

    return PrefMat


data_dir = "./dataset_100/separated_text_data/"
fnames = os.path.join(data_dir, 'names.txt')
fn = open(fnames)
dataN = fn.read()
fn.close()

linesOfN = dataN.split('\n')
food_data_names = ["" for x in range(101)]
for i, line in enumerate(linesOfN):
    food_data_names[i] = line

food_names = ""
for element in food_data_names:
    food_names += element + ";"

food_names = food_names.split(';')
food_names = food_names[:101]

food_columns = ["food" + str(i) for i in range(1, 11)]
first_ordered_subset = pd.DataFrame(columns=["Survey_ID", *food_columns])
second_ordered_subset = pd.DataFrame(columns=["Survey_ID", *food_columns])
third_ordered_subset = pd.DataFrame(columns=["Survey_ID", *food_columns])
for dirname, _, filenames in os.walk('./Answers_dataset/'):
    for index, filename in enumerate(filenames):
        path = os.path.join(dirname, filename)
        considered_survey = pd.read_csv(path, delimiter=";")
        for i, answer in enumerate(considered_survey.iterrows()):
            answer_number = answer[1][0]
            first_ordered_foods = [x.split(';') for x in answer[1][81].split('\n')]
            second_ordered_foods = [x.split(';') for x in answer[1][84].split('\n')]
            third_ordered_foods = [x.split(';') for x in answer[1][87].split('\n')]
            first_ordered_foods = first_ordered_foods[0]
            second_ordered_foods = second_ordered_foods[0]
            third_ordered_foods = third_ordered_foods[0]
            del first_ordered_foods[-1]
            del second_ordered_foods[-1]
            del third_ordered_foods[-1]
            first_ordered_foods.insert(0, answer[1][1])
            second_ordered_foods.insert(0, answer[1][1])
            third_ordered_foods.insert(0, answer[1][1])
            temp_first = pd.DataFrame([first_ordered_foods], columns=["Survey_ID", *food_columns])
            temp_second = pd.DataFrame([second_ordered_foods], columns=["Survey_ID", *food_columns])
            temp_third = pd.DataFrame([third_ordered_foods], columns=["Survey_ID", *food_columns])
            first_ordered_subset = first_ordered_subset.append(temp_first)
            second_ordered_subset = second_ordered_subset.append(temp_second)
            third_ordered_subset = third_ordered_subset.append(temp_third)


# COMMENTED TO NOT OVERWRITE

# new_filename1 = "Ordinamento1.csv"
# new_filename2 = "Ordinamento2.csv"
# new_filename3 = "Ordinamento3.csv"
# new_dirname = "./Ordinamenti/"
# new_path1 = os.path.join(new_dirname, new_filename1)
# new_path2 = os.path.join(new_dirname, new_filename2)
# new_path3 = os.path.join(new_dirname, new_filename3)
# first_ordered_subset.to_csv(new_path1, index=False, header=False)
# second_ordered_subset.to_csv(new_path2, index=False, header=False)
# third_ordered_subset.to_csv(new_path3, index=False, header=False)

arrayIntOrd1 = np.zeros((answer_number+1, 10), dtype="int32")
arrayIntOrd2 = np.zeros((answer_number+1, 10), dtype="int32")
arrayIntOrd3 = np.zeros((answer_number+1, 10), dtype="int32")
idSet = np.zeros(answer_number+1, dtype="int32")

for i, row in enumerate(first_ordered_subset.iterrows()):
    for j, element in enumerate(row[1]):
        if j == 0:
            idSet[i] = element
        else:
            element_name = element.replace(" ", "-")
            while "--" in element_name:
                element_name = element_name.replace("--", "-")
            if element_name == "Arancini":
                element_name = "Arancine"
            element_name = element_name.lower()
            for k, name in enumerate(food_names):
                name = name.lower()
                if element_name == name:
                    arrayIntOrd1[i, j-1] = k
                elif ("baccal" in name) and ("-in-umido" in name):  # for some strange reason has problem with "à"
                    if ("baccal" in element_name) and ("-in-umido" in element_name):
                        arrayIntOrd1[i, j - 1] = k
                elif ("canederli-alla-tirolese-(Kn" in name) and ("del" in name):   # for some strange reason has problem with "ö"
                    if ("canederli-alla-tirolese-(kn" in element_name) and ("del" in element_name):
                        arrayIntOrd1[i, j - 1] = k

for i, row in enumerate(second_ordered_subset.iterrows()):
    for j, element in enumerate(row[1]):
        if j == 0:
            if idSet[i] != element:
                print("some error occurred in line 139 or near")
                exit()
        else:
            element_name = element.replace(" ", "-")
            while "--" in element_name:
                element_name = element_name.replace("--", "-")
            if element_name == "Arancini":
                element_name = "Arancine"
            element_name = element_name.lower()
            for k, name in enumerate(food_names):
                name = name.lower()
                if element_name == name:
                    arrayIntOrd2[i, j-1] = k
                elif ("baccal" in name) and ("-in-umido" in name):
                    if ("baccal" in element_name) and ("-in-umido" in element_name):
                        arrayIntOrd2[i, j - 1] = k
                elif ("canederli-alla-tirolese-(kn" in name) and ("del" in name):
                    if ("canederli-alla-tirolese-(kn" in element_name) and ("del" in element_name):
                        arrayIntOrd2[i, j - 1] = k

for i, row in enumerate(third_ordered_subset.iterrows()):
    for j, element in enumerate(row[1]):
        if j == 0:
            if idSet[i] != element:
                print("some error occurred in line 155 or near")
                exit()
        else:
            element_name = element.replace(" ", "-")
            while "--" in element_name:
                element_name = element_name.replace("--", "-")
            if element_name == "Arancini":
                element_name = "Arancine"
            element_name = element_name.lower()
            for k, name in enumerate(food_names):
                name = name.lower()
                if element_name == name:
                    arrayIntOrd3[i, j-1] = k
                elif ("baccal" in name) and ("-in-umido" in name):
                    if ("baccal" in element_name) and ("-in-umido" in element_name):
                        arrayIntOrd3[i, j - 1] = k
                elif ("canederli-alla-tirolese-(kn" in name) and ("del" in name):
                    if ("canederli-alla-tirolese-(kn" in element_name) and ("del" in element_name):
                        arrayIntOrd3[i, j - 1] = k

ordTotale = np.concatenate((arrayIntOrd1, arrayIntOrd2, arrayIntOrd3), axis=1)
ricette = np.zeros((answer_number+1, 21), dtype="int32")
for i, row in enumerate(ordTotale):
    counter = 0
    for element in row:
        if not (element in ricette[i]):
            ricette[i, counter] = element
            counter = counter + 1

# ricette = np.sort(ricette)
print(ricette)

# COMMENTED TO NOT OVERWRITE

# output_coppie = "./Ordinamenti/dataset_coppie.txt"
# f = open(output_coppie)
# sys.stdout = open(output_coppie, "a")
# for row in ricette:
#         print_output = ""
#         for i, element in enumerate(row):
#             if(i < 20):
#                 j = i + 1
#                 for j in range(j, len(row)):
#                     ricettaX = row[i]
#                     ricettaY =row[j]
#                     print_output += str(ricettaX) + "," + str(ricettaY) + ";"
#         print(print_output)
#
#
# sys.stdout = sys.__stdout__
# f.close()

np.set_printoptions(linewidth=np.inf)
output_file = "./Ordinamenti/output-file.txt"
output_file2 = "./dataset-risposte/occ2.txt"
for n in range(0, answer_number+1):
    a = pref(ricette[n], [arrayIntOrd1[n]])
    matrixToPlot = np.zeros((len(a), len(a)))
    plt.figure(num=None, figsize=(12, 12), dpi=100)
    for i, row in enumerate(a):
        matrixToPlot[i, :] = row
    plt.matshow(matrixToPlot)
    plt.savefig("./matrixes/matrix" + str(n), dpi=300)

    f = open(output_file)
    sys.stdout = open(output_file, "a")
    finalOutput = ""
    for i, line in enumerate(a):
        j = i + 1
        line = line[j:]
        linea = str(line).replace('.', '')
        linea = linea.replace('[]', '')
        linea = linea.replace('[ ', '')
        linea = linea.replace('[-', '-')
        linea = linea.replace('[', '')
        linea = linea.replace(']', ' ')
        linea = linea.replace('  ', ' ')
        finalOutput = finalOutput + linea
    finalOutput = finalOutput[:-1]
    print(finalOutput)
    sys.stdout = sys.__stdout__
    f.close()
