import math
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras
import random
import matplotlib.pyplot as plt
import logging


logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def find_samples(first_food, second_food, all_food, pca_food, number_of_sample, index_first_food, index_second_food):
    counter = 0
    distances_metric = np.zeros(shape=(4950, 3), dtype='float32')   # see sushi_paper
    for i1 in range(0, 100):
        for i2 in range(i1 + 1, 100):
            temp1 = 0
            temp2 = 0
            for index_element in range(0, 2):
                for f in range(0, len(all_food[0])):
                    if index_element == 0:
                        if f == 0:
                            temp1 += pow((0 if all_food[i1, f] == first_food[f] else 3), 2)
                        temp1 += pow((abs(all_food[i1, f] - first_food[f])), 2)
                    else:
                        if f == 0:
                            temp2 += pow((0 if all_food[i2, f] == second_food[f] else 3), 2)
                        temp2 += pow((abs(all_food[i2, f] - second_food[f])), 2)
                if index_element == 0:
                    temp1 = math.sqrt(temp1)
                else:
                    temp2 = math.sqrt(temp2)
            distances_metric[counter, 0] = i1
            distances_metric[counter, 1] = i2
            distances_metric[counter, 2] = round(temp1 + temp2)
            counter += 1

    ind = np.argsort(distances_metric[:, -1])
    distances_metric_sorted = distances_metric[ind]
    to_return = np.zeros((number_of_sample, len(all_food[0])*2))
    to_return2 = np.zeros((number_of_sample, 34))
    counter = 0
    index = 0
    to_print = ""
    to_print2 = ""
    while True:
        if counter == number_of_sample:
            break
        if distances_metric_sorted[index, 2] == 0:  # we only want sampled data, not original
            index += 1
            continue
        if (int(distances_metric_sorted[index, 0]) == index_first_food) or (int(distances_metric_sorted[index, 1]) == index_second_food):
            index += 1
            continue
        temp_concat = np.concatenate((all_food[int(distances_metric_sorted[index, 0])], all_food[int(distances_metric_sorted[index, 1])]))
        temp_concat_2 = np.concatenate((pca_food[int(distances_metric_sorted[index, 0])], pca_food[int(distances_metric_sorted[index, 1])]))
        to_print = to_print + str(int(distances_metric_sorted[index, 0])) + "," + str(int(distances_metric_sorted[index, 1])) + ";"
        to_print2 = to_print2 + str(distances_metric_sorted[index, 2]) + ";"
        to_return[counter] = temp_concat
        to_return2[counter] = temp_concat_2
        index += 1
        counter += 1
    print(to_print)
    print("--------------------distances from original couple--------------------")
    print(to_print2)
    plt.scatter(pca_food[:, 0], pca_food[:, 1])
    plt.scatter(to_return2[:, 0], to_return2[:, 1])
    plt.scatter(pca_food[index_first_food, 0], pca_food[index_first_food, 1])
    plt.scatter(pca_food[index_second_food, 0], pca_food[index_second_food, 1])
    plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
    plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.clf()
    return to_return, to_return2
    return to_return, to_return2

def find_samples_2(first_food, second_food, all_food, pca_food, number_of_sample, index_first_food, index_second_food):
    counter = 0
    distances_metric = np.zeros(shape=(4950, 3), dtype='float32')   # see sushi_paper
    for i1 in range(0, 100):
        for i2 in range(i1 + 1, 100):
            temp1 = 0
            temp2 = 0
            for index_element in range(0, 2):
                for f in range(0, len(all_food[0])):
                    if index_element == 0:
                        if f == 0:
                            temp1 += pow((0 if all_food[i1, f] == first_food[f] else 3), 2)
                        temp1 += pow((abs(all_food[i1, f] - first_food[f])), 2)
                    else:
                        if f == 0:
                            temp2 += pow((0 if all_food[i2, f] == second_food[f] else 3), 2)
                        temp2 += pow((abs(all_food[i2, f] - second_food[f])), 2)
                if index_element == 0:
                    temp1 = math.sqrt(temp1)
                else:
                    temp2 = math.sqrt(temp2)
            distances_metric[counter, 0] = i1
            distances_metric[counter, 1] = i2
            distances_metric[counter, 2] = round(temp1 + temp2)
            counter += 1

    ind = np.argsort(distances_metric[:, -1])
    distances_metric_sorted = distances_metric[ind]
    to_return = np.zeros((number_of_sample, len(all_food[0])*2))
    to_return2 = np.zeros((number_of_sample, 34))
    counter = 0
    index = 0
    to_print = ""
    to_print2 = ""
    probabilistic_discounter = 10   # this value is used to lowering the probability to pick certain couple. It incremente everytime a couple is picked and resetted otherwise. This variable is used because otherwise this function is almost the same of "find_sample"
    while True:
        if counter == number_of_sample:
            break
        if distances_metric_sorted[index, 2] == 0:  # we don't want the couple we are approximating in the training set
            index += 1
            continue
        if (int(distances_metric_sorted[index, 0]) == index_first_food) or (int(distances_metric_sorted[index, 1]) == index_second_food):   # in the training set we don't want element in common wit the couple we are approximating
            index += 1
            continue
        temp_concat = np.concatenate((all_food[int(distances_metric_sorted[index, 0])], all_food[int(distances_metric_sorted[index, 1])]))
        temp_concat_2 = np.concatenate((pca_food[int(distances_metric_sorted[index, 0])], pca_food[int(distances_metric_sorted[index, 1])]))
        to_print = to_print + str(int(distances_metric_sorted[index, 0])) + "," + str(int(distances_metric_sorted[index, 1])) + ";"
        to_print2 = to_print2 + str(distances_metric_sorted[index, 2]) + ";"
        probability_to_pick = (1-(distances_metric_sorted[index, 2]/distances_metric_sorted[4949, 2]))*100
        choice_to_pick = random.choices([0, 1], weights=(100+probabilistic_discounter-probability_to_pick, probability_to_pick-probabilistic_discounter), k=1)
        if choice_to_pick[0] == 1:
            to_return[counter] = temp_concat
            to_return2[counter] = temp_concat_2
            index += 1
            counter += 1
            probabilistic_discounter += 5
        else:
            print("WOW: " + str(100+probabilistic_discounter-probability_to_pick))
            probabilistic_discounter = 10
            counter += 1
    print(to_print)
    print("--------------------distances from original couple--------------------")
    print(to_print2)
    plt.scatter(pca_food[:, 0], pca_food[:, 1])
    plt.scatter(to_return2[:, 0], to_return2[:, 1])
    plt.scatter(pca_food[index_first_food, 0], pca_food[index_first_food, 1])
    plt.scatter(pca_food[index_second_food, 0], pca_food[index_second_food, 1])
    plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
    plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
    plt.clf()
    return to_return, to_return2


def prediction(user_id, samples_to_predict):
    model = keras.models.load_model('../../../NN_data/models/User' + str(user_id) + '/folder_version')
    predictions = model(samples_to_predict)
    to_print = ""
    for index, predict in enumerate(predictions):
        if index >= 105:
            break
        if (predictions[index, 0] == predictions[index, 1]) and (predictions[index, 0] == predictions[index, 2]):
            to_print = to_print + "0 "
        else:
            maxProbClass = max(predictions[index, 0], predictions[index, 1], predictions[index, 2])
            if predictions[index, 0] == maxProbClass:
                to_print = to_print + "1 "
            elif predictions[index, 1] == maxProbClass:
                to_print = to_print + "0 "
            else:
                to_print = to_print + "-1 "
    to_print = to_print[0:len(to_print) - 1]
    print(to_print)
    del model
    del predictions


def single_prediction(user_id, first_food, second_food):
    model = keras.models.load_model('../../../NN_data/models/User' + str(user_id) + '/folder_version')
    to_predict = np.concatenate((first_food, second_food))
    to_predict_reshaped = to_predict.reshape(1, -1)
    predictions = model.predict(to_predict_reshaped, verbose=0)
    to_print = ""
    for index, predict in enumerate(predictions):
        if index >= 105:
            break
        if (predictions[index, 0] == predictions[index, 1]) and (predictions[index, 0] == predictions[index, 2]):
            to_print = to_print + "0 "
        else:
            maxProbClass = max(predictions[index, 0], predictions[index, 1], predictions[index, 2])
            if predictions[index, 0] == maxProbClass:
                to_print = to_print + "1 "
            elif predictions[index, 1] == maxProbClass:
                to_print = to_print + "0 "
            else:
                to_print = to_print + "-1 "
    to_print = to_print[0:len(to_print) - 1]
    print(to_print)
    del model
    del to_predict
    del to_predict_reshaped
    del predictions



data_dir = "../../../dataset_100/separated_text_data/"

fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fNames = os.path.join(data_dir, 'names.txt')
fCouple = os.path.join('../../../Ordinamenti/dataset_coppie.txt')
fLabels = os.path.join('../../../Ordinamenti/output-file.txt')

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

original_category = food_data_categories[:, 0]
original_category = np.reshape(original_category, (101, 1))

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
        normalized_food_data_ingredients[i, j] = food_data_ingredients[i, j] / sum_of_elements

for i, row in enumerate(food_data_preparation):
    sum_of_elements = 0
    for element in row:
        sum_of_elements = sum_of_elements + element
    for j, element in enumerate(row):
        normalized_food_data_preparation[i, j] = food_data_preparation[i, j] / sum_of_elements

# scaling preparation time, cost, difficulty, ingredients and preparations
scaler = preprocessing.StandardScaler()
scaled_food_data_finalCategory = scaler.fit_transform(finalCategory)
scaled_food_data_scalars = scaler.fit_transform(food_data_scalars)
scaled_food_data_categories = scaler.fit_transform(food_data_categories)
scaled_food_data_ingredients = scaler.fit_transform(normalized_food_data_ingredients)
scaled_food_data_preparation = scaler.fit_transform(normalized_food_data_preparation)

# concatenate all data in a numpy tensor
all_data = np.concatenate([scaled_food_data_finalCategory, scaled_food_data_categories, scaled_food_data_scalars, scaled_food_data_ingredients, scaled_food_data_preparation], axis=1)
all_data_original = np.concatenate([original_category, food_data_categories, food_data_scalars, food_data_ingredients, food_data_preparation], axis=1)

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

per_var = np.round(pca.explained_variance_ratio_*101, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
pca_df = pd.DataFrame(pca_data, columns=labels)

# outliers: 15, 39, 45, 46, 53, 54, 56

while True:
    print("insert the user_id, special value -1 for all user with random couple query")
    user_id_to_pass = input()

    if user_id_to_pass == "-1":
        break

    if not user_id_to_pass.isdigit():
        print("input must be integers")
        continue
    if int(user_id_to_pass) < 0 or int(user_id_to_pass) > 47:
        print("user_id must be 0 < user_id < 47")
        continue

    print("insert the first food_id and the second_food_id")

    first_food_id_to_pass = input()
    second_food_id_to_pass = input()

    if not first_food_id_to_pass.isdigit() or not second_food_id_to_pass.isdigit():
        print("input must be integers")
        continue
    if int(first_food_id_to_pass) < 0 or int(first_food_id_to_pass) > 99:
        print("food_id must be 0 < food_id < 99")
        continue
    if int(second_food_id_to_pass) < 0 or int(second_food_id_to_pass) > 99:
        print("user_id must be 0 < food_id < 47")
        continue

    # X, X_PCA = find_samples(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=45, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass))
    X, X_PCA = find_samples_2(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=45, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass))
    # plt.scatter(pca_data[:, 0], pca_data[:, 1])
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.scatter(pca_data[int(first_food_id_to_pass), 0], pca_data[int(first_food_id_to_pass), 1])
    # plt.scatter(pca_data[int(second_food_id_to_pass), 0], pca_data[int(second_food_id_to_pass), 1])
    # plt.show()
    # plt.clf()

    prediction(user_id=user_id_to_pass, samples_to_predict=X_PCA)

    print("do you want to continue? [y/n]")
    want_to_remember = ""
    want = input()
    if want == "N" or want == "n":
        break
    if want == "Y" or want == "y":
        continue
    while True:
        print("illegal input. Do yo want to continue? [y/n]")
        want = input()
        if want == "N" or want == "n":
            want_to_remember = want
        if want == "Y" or want == "y":
            want_to_remember = want
        if want_to_remember != "":
            break
    want = want_to_remember
    if want == "N" or want == "n":
        break
    if want == "Y" or want == "y":
        continue

# recipes = random.sample(range(0, 100), 2) # first random choice (TRAIN) 9, 47 (considering plot with PC1 and PC2 and cluster with k=3, 9 is near <0, 2> - <yellow cluster, on bound>; while 47 is near <2.2, -1.9> - <blue cluster, near centroid>) which seems a good choice for experiment (not too far but not too close)
# first_food_id_to_pass = recipes[0]
# second_food_id_to_pass = recipes[1]

# first_food_id_to_pass = 9
# second_food_id_to_pass = 47

# for test i picked by hand 28 and 87, because they are near to symmetry with 9 and 47
first_food_id_to_pass = 28
second_food_id_to_pass = 87
print("--------------------couple queried--------------------")
print(str(int(first_food_id_to_pass)) + "," + str(int(second_food_id_to_pass)))
print("--------------------prediction of i-th user's model on queried couple--------------------")
for i in range(0, 48):
    single_prediction(user_id=i, first_food=pca_data[int(first_food_id_to_pass)], second_food=pca_data[int(second_food_id_to_pass)])
print("--------------------generation samples for train--------------------")
# X, X_PCA = find_samples(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=105, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass))
X, X_PCA = find_samples_2(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=105, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass))
print("--------------------generation label for train--------------------")
for i in range(0, 48):
    prediction(user_id=i, samples_to_predict=X_PCA)
