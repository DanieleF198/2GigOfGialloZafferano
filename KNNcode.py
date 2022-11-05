import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_decision_regions
import re
from sklearn.model_selection import KFold
import warnings
import operator
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


#reading data from external files
data_dir = "./dataset_100/separated_text_data/"
fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fNames = os.path.join(data_dir, 'names.txt')
fCouple = os.path.join('./Ordinamenti/dataset_coppie.txt')
fLabels = os.path.join('./Ordinamenti/output-file.txt')

fS = open(fScalar)
dataS = fS.read()
fS.close()
fC = open(fCategories)
dataC = fC.read()
fC.close()
fI = open(fIngredients)
dataI = fI.read()
fI.close()
fP = open(fPreparation)
dataP = fP.read()
fP.close()
fN = open(fNames)
food_data_names = fN.read()
fN.close()
fCC = open(fCouple)
dataCC = fCC.read()
fCC.close()
fL = open(fLabels)
dataL = fL.read()
fL.close()

linesOfS = dataS.split('\n')
food_data_scalars = np.zeros((len(linesOfS), 1), dtype='float32')
for i, line in enumerate(linesOfS):
    values = [x for x in line.split(' ')[1:]]
    food_data_scalars[i, :] = values

linesOfC = dataC.split('\n')
food_data_categories = np.zeros((len(linesOfC), 3), dtype='float32')
for i, line in enumerate(linesOfC):
    values = [x for x in line.split(' ')[1:]]
    food_data_categories[i, :] = values

linesOfI = dataI.split('\n')
food_data_ingredients = np.zeros((len(linesOfI), 36), dtype='float32')
for i, line in enumerate(linesOfI):
    values = [x for x in line.split(' ')[1:]]
    food_data_ingredients[i, :] = values

linesOfP = dataP.split('\n')
food_data_preparation = np.zeros((len(linesOfP), 8), dtype='float32')
for i, line in enumerate(linesOfP):
    values = [x for x in line.split(' ')[1:]]
    food_data_preparation[i, :] = values

linesOfCC = dataCC.split('\n')
food_couple = np.zeros((48, 210, 2), dtype='int32')
for i, line in enumerate(linesOfCC):
    if i < 48:
        values = [x for x in line.split(';')[:]]
        values = values[:210]
        for j, value in enumerate(values):
            twoValues = value.split(',')
            first = int(twoValues[0])
            second = int(twoValues[1])
            food_couple[i, j, 0] = first
            food_couple[i, j, 1] = second
            # food_couple[i, j+210, 0] = second
            # food_couple[i, j+210, 1] = first

linesOfL = dataL.split('\n')
couple_label = np.zeros((48, 210), dtype='int32')
for i, line in enumerate(linesOfL):
    if i < 48:
        values = [x for x in line.split(' ')[:]]
        for j, value in enumerate(values):
            veryValue = int(value)
            if veryValue == 1:
                couple_label[i, j] = 1
                # couple_label[i, j + 210] = -1
            elif veryValue == -1:
                couple_label[i, j] = -1
                # couple_label[i, j + 210] = 1
            else:
                couple_label[i, j] = 0
                # couple_label[i, j+210] = 0

# manipulate categorical data
category1HE = food_data_categories[:, 0]
food_data_categories = np.delete(food_data_categories, 0, 1)
category1HE = np.reshape(category1HE, (101, 1))
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(category1HE)
finalCategory = enc.transform(category1HE).toarray()

# remove ingredients never used
food_data_ingredients = np.delete(food_data_ingredients, [1, 16, 23, 26, 32], axis=1)

# normalize scaled ingredients and scaled preparation

normalized_food_data_ingredients = np.zeros((101, 31), dtype="float32")
normalized_food_data_preparation = np.zeros((101, 8), dtype="float32")

for i, row in enumerate(food_data_ingredients):
    sum_of_elements = 0
    for element in row:
        sum_of_elements = sum_of_elements + element
    for j, element in enumerate(row):
        normalized_food_data_ingredients[i, j] = food_data_ingredients[i, j]/sum_of_elements

for i, row in enumerate(food_data_preparation):
    sum_of_elements = 0
    for element in row:
        sum_of_elements = sum_of_elements + element
    for j, element in enumerate(row):
        normalized_food_data_preparation[i, j] = food_data_preparation[i, j]/sum_of_elements

# scaling preparation time, cost, difficulty, ingredients and preparations
scaler = preprocessing.StandardScaler()
scaled_food_data_finalCategory = scaler.fit_transform(finalCategory)
scaled_food_data_scalars = scaler.fit_transform(food_data_scalars)
scaled_food_data_categories = scaler.fit_transform(food_data_categories)
scaled_food_data_ingredients = scaler.fit_transform(normalized_food_data_ingredients)
scaled_food_data_preparation = scaler.fit_transform(normalized_food_data_preparation)

# concatenate all data in a numpy tensor
all_data = np.concatenate([scaled_food_data_finalCategory, scaled_food_data_categories, scaled_food_data_scalars, scaled_food_data_ingredients, scaled_food_data_preparation], axis=1)

# preparation of labels for pandas dataframe
foodsLabel = ['F' + str(i) for i in range(1, 101)]
categoryLabel = ['CATEGORY' + str(i) for i in range(1, 6)]
ingredientsLabel = ['INGREDIENTS' + str(i) for i in range(1, 32)]
preparationsLabel = ['PREPARATIONS' + str(i) for i in range(1, 9)]

# creating pandas dataframe
final_data = pd.DataFrame(columns=[*categoryLabel, 'COST', 'DIFFICULTY', 'PREPARATION', *ingredientsLabel, *preparationsLabel], index=foodsLabel)

# fill pandas dataframe with data that we concatenated in line 83
for i, food in enumerate(final_data.index):
    final_data.loc[food] = all_data[i]

# instantiate pca object, then fit on final data
# pca = PCA() #(we used this before to understand the number of component to use)
pca = decomposition.PCA(n_components=17)
pca.fit(final_data)
# apply pca on data in pandas_dataframe
pca_data = pca.transform(final_data)

# VERSION F1X | F2X | ... | FnX | F1Y | F2Y | ... | FnY |
food_effective_couple = np.zeros((len(food_couple), 210, 2, 17)) #len(food_couple) answer, each one with 210 couple made of 2 elements that each is rapresented by 17 feature (PCA)
for i, answer in enumerate(food_couple):
    if i < 48:
        for j, couple in enumerate(answer):
            for k, food in enumerate(couple):
                food_effective_couple[i, j, k, :] = pca_data[int(food)]


# # VERSION F1X | F1Y | F2X | F2Y | ... | FnX | FnY |
# food_effective_couple = np.zeros((len(food_couple), 210, 34))
# plt.clf()
# for i, answer in enumerate(food_couple):
#     if i < 125:
#         for j, couple in enumerate(answer):
#             foodX = pca_data[int(couple[0])]
#             foodY = pca_data[int(couple[1])]
#             even = 1
#             odd = 0
#             for feature in foodX:
#                 food_effective_couple[i, j, odd] = feature
#                 odd += 2
#             for feature in foodY:
#                 food_effective_couple[i, j, even] = feature
#                 even += 2

ks = [1, 5, 20, 40]
ps = [1, 2]
weights = ["uniform", "distance"]
leaf_sizes = [10, 30, 50]

# reverting couple to be of the form "A is/is not/equal preferred than B"
for i, user in enumerate(food_effective_couple):
    for j, couple in enumerate(user):
        SFF1 = 0 #sum of feature of food 1
        SFF2 = 0 #sum of feature of food 2
        for k, food in enumerate(couple):
            if k == 0:
                for feature in food:
                    SFF1 += feature
                continue
            else:
                for feature in food:
                    SFF2 += feature
                if SFF1 < SFF2:
                    temp = np.copy(food_effective_couple[i, j, 0])
                    food_effective_couple[i, j, 0] = food_effective_couple[i, j, 1]
                    food_effective_couple[i, j, 1] = temp
                    if couple_label[i, j] != 0:
                        couple_label[i, j] *= -1

food_effective_couple = np.reshape(food_effective_couple, (len(food_couple), 210, 34))

food_effective_couple_zeros = np.zeros((len(food_effective_couple),210,34))
food_effective_couple_no_zeros = np.zeros((len(food_effective_couple),210,34))
couple_label_zeros = np.zeros((len(couple_label), 210))
couple_label_no_zeros = np.zeros((len(couple_label), 210))
counter_of_answer_with_zero = 0
counter_of_answer_without_zero = 0
for i, answer in enumerate(couple_label):
    if i > 48:
        break
    zero_count = 0
    for element in answer:
        if element == 0:
            zero_count += 1
    if zero_count <= 10:
        inner_counter = 0
        for j, couple in enumerate(food_effective_couple[i]):
            if answer[j] == 1:
                food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
                couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = 1
                inner_counter += 1
            elif answer[j] == -1:
                food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
                couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = -1
                inner_counter += 1
            else:
                continue
        counter_of_answer_without_zero += 1
    else:
        food_effective_couple_zeros[counter_of_answer_with_zero] = food_effective_couple[i]
        couple_label_zeros[counter_of_answer_with_zero] = answer
        counter_of_answer_with_zero += 1

food_effective_couple_zeros = food_effective_couple_zeros[0:counter_of_answer_with_zero]
couple_label_zeros = couple_label_zeros[0:counter_of_answer_with_zero]
food_effective_couple_no_zeros = food_effective_couple_no_zeros[0:counter_of_answer_without_zero]
couple_label_no_zeros = couple_label_no_zeros[0:counter_of_answer_without_zero]
half = len(food_effective_couple_zeros)/2
half = round(half)

print("If you want to make cross-validation for hyperparameter press 1 and then enter")
print("If you want to test some variable press 2 and then enter")
print("NOTE: variable can be changed in the code only, take a look near the line 522 to change them")
inputVar = input()
if not inputVar.isdigit():
    while True:
        print("input not accepted. You have to press 1 or 2, and then enter")
        inputVar = input()
        if inputVar.isdigit():
            if int(inputVar) != 1 and int(inputVar) != 2:
                continue
            else:
                break

if int(inputVar) != 1 and int(inputVar) != 2:
    while True:
        print("input not accepted. You have to press 1 or 2, and then enter")
        inputVar = input()
        if not inputVar.isdigit():
            continue
        else:
            if int(inputVar) != 1 and int(inputVar) != 2:
                continue
            else:
                break

if int(inputVar) == 1:

    for user_id, row in enumerate(food_effective_couple_zeros):
        X = food_effective_couple_zeros[user_id]
        Y = couple_label_zeros[user_id]
        final_values_acc = np.zeros((2,2,4,3), dtype='float32') # matrix of values of accuracy
        std_of_final_values_acc = np.zeros((2,2,4,3), dtype='float32')
        final_values_prec = np.zeros((2,2,4,3), dtype='float32') # matrix of values of precision
        std_of_final_values_prec = np.zeros((2,2,4,3), dtype='float32')
        final_values_rec = np.zeros((2,2,4,3), dtype='float32') # matrix of values of recall
        std_of_final_values_rec = np.zeros((2,2,4,3), dtype='float32')
        final_values_f1s = np.zeros((2,2,4,3), dtype='float32') # matrix of values of f1score
        std_of_final_values_f1s = np.zeros((2,2,4,3), dtype='float32')
        for p_index, p in enumerate(ps):
            for weight_index, weight in enumerate(weights):
                for k_index, k in enumerate(ks):
                    for leaf_size_index, leaf_size in enumerate(leaf_sizes):

                        accuracy = []
                        precision = []
                        recall = []
                        f1score = []

                        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                        for train, test in kfold.split(X, Y):
                            x_train, y_train = X[train], Y[train]
                            x_test, y_test = X[test], Y[test]
                            clf = KNeighborsClassifier(n_neighbors=k, leaf_size=leaf_size, weights=weight, p=p)
                            clf.fit(x_train, y_train)
                            predict = clf.predict(x_test)
                            numbers = re.findall('[0-9]\.[0-9][0-9]', classification_report(y_test, predict))
                            if len(numbers) < 16: #case where there is no "zero" label
                                for i, number in enumerate(numbers):
                                    number = float(number)
                                    if i == 6:
                                        accuracy.append(number)
                                    if i == 10:
                                        precision.append(number)
                                    if i == 11:
                                        recall.append(number)
                                    if i == 12:
                                        f1score.append(number)
                            else:
                                for i, number in enumerate(numbers):
                                    number = float(number)
                                    if i == 9:
                                        accuracy.append(number)
                                    if i == 13:
                                        precision.append(number)
                                    if i == 14:
                                        recall.append(number)
                                    if i == 15:
                                        f1score.append(number)

                        mn_acc = np.mean(accuracy)
                        std_acc = np.std(accuracy)
                        mn_prec = np.mean(precision)
                        std_prec = np.std(precision)
                        mn_rec = np.mean(recall)
                        std_rec = np.std(recall)
                        mn_f1s = np.mean(f1score)
                        std_f1s = np.std(f1score)

                        final_values_acc[p_index, weight_index, k_index, leaf_size_index] = mn_acc
                        std_of_final_values_acc[p_index, weight_index, k_index, leaf_size_index] = std_acc
                        final_values_prec[p_index, weight_index, k_index, leaf_size_index] = mn_prec
                        std_of_final_values_prec[p_index, weight_index, k_index, leaf_size_index] = std_prec
                        final_values_rec[p_index, weight_index, k_index, leaf_size_index] = mn_rec
                        std_of_final_values_rec[p_index, weight_index, k_index, leaf_size_index] = std_rec
                        final_values_f1s[p_index, weight_index, k_index, leaf_size_index] = mn_f1s
                        std_of_final_values_f1s[p_index, weight_index, k_index, leaf_size_index] = std_f1s

        better_k = 0
        better_p = 0
        better_weight = ""
        better_leaf_size = 0
        better_mean_acc = 0
        better_std_acc = 0
        better_mean_prec = 0
        better_std_prec = 0
        better_mean_rec = 0
        better_std_rec = 0
        better_mean_f1s = 0
        better_std_f1s = 0

        for p_index, p in enumerate(ps):
            for weight_index, weight in enumerate(weights):
                for k_index, k in enumerate(ks):
                    for leaf_size_index, leaf_size in enumerate(leaf_sizes):
                        if final_values_acc[p_index, weight_index, k_index, leaf_size_index] > better_mean_acc:
                            better_mean_acc = final_values_acc[p_index, weight_index, k_index, leaf_size_index]
                            better_std_acc = std_of_final_values_acc[p_index, weight_index, k_index, leaf_size_index]
                            better_mean_prec = final_values_prec[p_index, weight_index, k_index, leaf_size_index]
                            better_std_prec = std_of_final_values_prec[p_index, weight_index, k_index, leaf_size_index]
                            better_mean_rec = final_values_rec[p_index, weight_index, k_index, leaf_size_index]
                            better_std_rec = std_of_final_values_rec[p_index, weight_index, k_index, leaf_size_index]
                            better_mean_f1s = final_values_f1s[p_index, weight_index, k_index, leaf_size_index]
                            better_std_f1s = std_of_final_values_f1s[p_index, weight_index, k_index, leaf_size_index]
                            better_k = k
                            better_p = p
                            better_weight = weight
                            better_leaf_size = leaf_size

        print("USER" + str(user_id) + "; n_neighbors = " + str(better_k) + "; leaf_size = " + str(better_leaf_size) + "; weight = " + better_weight + "; p = " + str(better_p) + ", mean accuracy: " + str(better_mean_acc) + " (+/- " + str(better_std_acc) + "), mean precision: " + str(better_mean_prec) + " (+/- " + str(better_std_prec) + "),  mean recall: " + str(better_mean_rec) + " (+/- " + str(better_std_rec) + "),  mean f1score: " + str(better_mean_f1s) + " (+/- " + str(better_std_f1s) + "),")

else:
    n_neighbor = 20
    p = 1
    weight = "distance"
    leaf_size = 20
    mean_of_the_mean_acc = 0
    mean_of_the_mean_prec = 0
    mean_of_the_mean_rec = 0
    mean_of_the_mean_f1s = 0
    print("variable: k = " + str(n_neighbor) + ", p = " + str(p) + ", weight = " + weight + "leaf_size = " + str(leaf_size))
    for k, row in enumerate(food_effective_couple_zeros):
        # if k < half:
        #     continue
        X = food_effective_couple_zeros[k]
        Y = couple_label_zeros[k]
        accuracy = []
        precision = []
        recall = []
        f1score = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for train, test in kfold.split(X, Y):
            x_train, y_train = X[train], Y[train]
            x_test, y_test = X[test], Y[test]
            clf = KNeighborsClassifier(n_neighbors=n_neighbor, weights=weight, p=p, leaf_size=leaf_size)
            clf.fit(x_train, y_train)
            predict = clf.predict(x_test)
            numbers = re.findall('[0-9]\.[0-9][0-9]', classification_report(y_test, predict))
            if len(numbers) < 16:  # case where there is no "zero" label
                for i, number in enumerate(numbers):
                    number = float(number)
                    if i == 6:
                        accuracy.append(number)
                    if i == 10:
                        precision.append(number)
                    if i == 11:
                        recall.append(number)
                    if i == 12:
                        f1score.append(number)
            else:
                for i, number in enumerate(numbers):
                    number = float(number)
                    if i == 9:
                        accuracy.append(number)
                    if i == 13:
                        precision.append(number)
                    if i == 14:
                        recall.append(number)
                    if i == 15:
                        f1score.append(number)

        mn_acc = np.mean(accuracy)
        std_acc = np.std(accuracy)
        mn_prec = np.mean(precision)
        std_prec = np.std(precision)
        mn_rec = np.mean(recall)
        std_rec = np.std(recall)
        mn_f1s = np.mean(f1score)
        std_f1s = np.std(f1score)

        mean_of_the_mean_acc += mn_acc
        mean_of_the_mean_prec += mn_prec
        mean_of_the_mean_rec += mn_rec
        mean_of_the_mean_f1s += mn_f1s

        print("UserID: " + str(k) + "; Mean Accuracy: " + str(mn_acc) + " (+/- " + str(std_acc) + "), mean precision: " + str(mn_prec) + " (+/- " + str(std_prec) + "),  mean recall: " + str(mn_rec) + " (+/- " + str(std_rec) + "),  mean f1score: " + str(mn_f1s) + " (+/- " + str(std_f1s) + "),")

    mean_of_the_mean_acc /= 48
    mean_of_the_mean_prec /= 48
    mean_of_the_mean_rec /= 48
    mean_of_the_mean_f1s /= 48

    print("mean of the means accuracy = " + str(mean_of_the_mean_acc))
    print("mean of the means precision = " + str(mean_of_the_mean_prec))
    print("mean of the means recall = " + str(mean_of_the_mean_rec))
    print("mean of the means f1score = " + str(mean_of_the_mean_f1s))
