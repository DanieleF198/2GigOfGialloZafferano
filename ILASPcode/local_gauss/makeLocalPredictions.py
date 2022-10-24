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
    return to_return, to_return2


def find_samples_gaussian_noise(first_food, second_food, all_food, pca_food, number_of_sample, index_first_food, index_second_food, pca_model):
    to_return = np.zeros((number_of_sample, len(all_food[0])*2), dtype='float32')
    to_return2 = np.zeros((number_of_sample, len(pca_food[0])*2), dtype='float32')
    number_of_new_food = (int(np.floor(np.sqrt(number_of_sample))) + 1) * 2
    raw_new_food = np.zeros((number_of_new_food, len(all_food[0])), dtype='float32')
    new_food = np.zeros((number_of_new_food, len(all_food[0])), dtype='float32')
    # add noise to original data
    counter_couple = 0
    for index in range(0, int(np.floor(number_of_new_food/2))):
        noise = tf.random.normal(shape=(tf.shape(first_food)), mean=0.0, stddev=0.05, dtype="float32")
        first_food_with_gauss = np.array(tf.add(first_food, noise))
        noise = tf.random.normal(shape=(tf.shape(first_food)), mean=0.0, stddev=0.05, dtype="float32")
        second_food_with_gauss = np.array(tf.add(second_food, noise))
        raw_new_food[counter_couple] = first_food_with_gauss
        counter_couple += 1
        raw_new_food[counter_couple] = second_food_with_gauss
        counter_couple += 1

    # rounding data:
    # for each feature i round with P = 0.5 to superior part or inferior part of the number. Also, for each feature (over the samples) i round superior part only number_of_new_food/2 times (eventually rounded naturally) and inferior part to remained cases

    random_categories = np.zeros((number_of_new_food, 3), dtype='float32')
    random_scalar = np.zeros((number_of_new_food, 1), dtype='float32')
    random_ingredients = np.zeros((number_of_new_food, 31), dtype='float32')
    random_preparations = np.zeros((number_of_new_food, 8), dtype='float32')
    counter_categories = 0
    marks_categories = [False for bool_value in range(3)]
    counter_scalar = 0
    counter_ingredients = 0
    marks_ingredients = [False for bool_value in range(31)]
    counter_preparations = 0
    marks_preparations = [False for bool_value in range(8)]
    while True:
        counter_marks = 0  # if all feature of tensor are done go on, anyway continue
        for mark_categories in marks_categories:
            if mark_categories:
                counter_marks += 1
        if counter_marks == 3:
            break
        for feature_index in range(0, 3):
            while True:
                if np.sum(random_categories[:, feature_index]) >= round(number_of_new_food / 2):
                    marks_categories[feature_index] = True
                    counter_categories = 0
                    break
                else:
                    if random_categories[counter_categories, feature_index] == 0:
                        random_categories[counter_categories, feature_index] += random.randint(0, 1)
                    counter_categories += 1
                    if counter_categories == number_of_new_food - 1:
                        counter_categories = 0

    while True:
        if np.sum(random_scalar) >= round(number_of_new_food / 2):
            break
        else:
            if random_scalar[counter_scalar] == 0:
                random_scalar[counter_scalar] += random.randint(0, 1)
            counter_scalar += 1
            if counter_scalar == number_of_new_food - 1:
                counter_scalar = 0

    while True:
        counter_marks = 0  # if all feature of tensor are done go on, anyway continue
        for mark_ingredients in marks_ingredients:
            if mark_ingredients:
                counter_marks += 1
        if counter_marks == 31:
            break
        for feature_index in range(0, 31):
            while True:
                if np.sum(random_ingredients[:, feature_index]) >= round(number_of_new_food / 2):
                    marks_ingredients[feature_index] = True
                    counter_ingredients = 0
                    break
                else:
                    if random_ingredients[counter_ingredients, feature_index] == 0:
                        random_ingredients[counter_ingredients, feature_index] += random.randint(0, 1)
                    counter_ingredients += 1
                    if counter_ingredients == number_of_new_food - 1:
                        counter_ingredients = 0

    while True:
        counter_marks = 0  # if all feature of tensor are done go on, anyway continue
        for mark_preparations in marks_preparations:
            if mark_preparations:
                counter_marks += 1
        if counter_marks == 8:
            break
        for feature_index in range(0, 8):
            while True:
                if np.sum(random_preparations[:, feature_index]) >= round(number_of_new_food / 2):
                    marks_preparations[feature_index] = True
                    counter_preparations = 0
                    break
                else:
                    if random_preparations[counter_preparations, feature_index] == 0:
                        random_preparations[counter_preparations, feature_index] += random.randint(0, 1)
                    counter_preparations += 1
                    if counter_preparations == number_of_new_food - 1:
                        counter_preparations = 0

    # rounding food

    for noised_row_index, noised_row in enumerate(raw_new_food):
        for noised_element_index, noised_element in enumerate(noised_row):
            # for category it's better to start from the natural codification, add noise, round it and only then convert to 1HE for PCA.
            if 0 <= noised_element_index < 3:
                if random_categories[noised_row_index, noised_element_index] == 0:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index])
                else:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index]) + 1
            elif noised_element_index == 3:
                if random_scalar[noised_row_index, 0] == 0:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index])
                else:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index]) + 1
            elif 4 <= noised_element_index < 35:
                if random_ingredients[noised_row_index, noised_element_index - 4] == 0:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index])
                    if new_food[noised_row_index, noised_element_index] > 5:
                        new_food[noised_row_index, noised_element_index] = 5
                else:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index]) + 1
                    if new_food[noised_row_index, noised_element_index] > 5:
                        new_food[noised_row_index, noised_element_index] = 5
            elif 35 <= noised_element_index < 43:
                if random_preparations[noised_row_index, noised_element_index - 35] == 0:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index])
                    if new_food[noised_row_index, noised_element_index] > 5:
                        new_food[noised_row_index, noised_element_index] = 5
                else:
                    new_food[noised_row_index, noised_element_index] = np.floor(raw_new_food[noised_row_index, noised_element_index]) + 1
                    if new_food[noised_row_index, noised_element_index] > 5:
                        new_food[noised_row_index, noised_element_index] = 5

    new_food[new_food < 0] = 0  # it's possible for ingredients and preparations that, cause of gaussian noise, get negative value rounded to -1, that we don't want

    indexes_of_samples = np.zeros((number_of_sample, 2), dtype='int32')
    insert_counter = 0
    start_first = True  # in order to get duplicated in balanced way before i create couple of type X, a and then a, X alternating (avoid duplicated)
    exit_condition = False
    number_of_change = 0
    while True:
        if exit_condition:
            break
        if start_first:
            for first_index, first_raw_food in enumerate(new_food):
                if first_index < number_of_change:
                    continue
                for second_index, second_raw_food in enumerate(new_food):
                    if (first_index % 2 != 0) or (second_index % 2 == 0) or first_index >= second_index:   # in even i have duplicated of first food, in odd of second
                        continue
                    else:
                        to_return[insert_counter] = np.concatenate((first_raw_food, second_raw_food))
                        indexes_of_samples[insert_counter] = [first_index, second_index]
                        insert_counter += 1
                        if insert_counter >= len(to_return):
                            exit_condition = True
                            break
                start_first = False
                number_of_change += 1
                break
        else:
            for second_index, second_raw_food in enumerate(new_food):
                if second_index < number_of_change:
                    continue
                for first_index, first_raw_food in enumerate(new_food):
                    if (first_index % 2 != 0) or (second_index % 2 == 0) or second_index >= first_index:    # index are switched
                        continue
                    else:
                        to_return[insert_counter] = np.concatenate((second_raw_food, first_raw_food))
                        indexes_of_samples[insert_counter] = [second_index, first_index]
                        insert_counter += 1
                        if insert_counter >= len(to_return):
                            exit_condition = True
                            break
                number_of_change += 1
                start_first = True
                break

    # if len(np.unique(to_return, axis=0)) < len(to_return):  # check if some duplicated arised (debug mode)
    #     quit()

    if len(np.unique(to_return, axis=0)) < len(to_return):  # some duplicate could be generated, in this case i prefer directly restart the program and hope... anyway probability of duplicates should be really low
        quit()
    if len(np.unique(new_food, axis=0)) < len(new_food):  # for next comfort i want all food to be differents (indexing problem)
        quit()
    # standardize data as for original before PCA
    new_food_copy = new_food.copy()
    scaler = preprocessing.StandardScaler()
    enc2 = OneHotEncoder(handle_unknown='ignore')   # manage the category that doesn't appear to have length 5 to categories
    category = new_food_copy[:, 0]
    category = np.append(category, [0, 1, 2, 3, 4])   # for 1HotEncoder
    category = category.reshape(-1, 1)
    categories = new_food_copy[:, 1:3]
    scalars = new_food_copy[:, 3].reshape(-1, 1)
    ingredients = new_food_copy[:, 4:35]
    preparations = new_food_copy[:, 35:]
    enc2.fit(category)
    category = enc2.transform(category).toarray()
    for to_delete in range(0,5):
        category = np.delete(category, len(category)-1, 0)
    for index_ingredient, entry in enumerate(ingredients):
        sum_elements = 0
        for ingredient in entry:
            sum_elements = sum_elements + ingredient
        for index_entry, ingredient in enumerate(entry):
            ingredients[index_ingredient, index_entry] = ingredients[index_ingredient, index_entry] / sum_elements
    for index_preparations, entry in enumerate(preparations):
        sum_elements = 0
        for preparation in entry:
            sum_elements = sum_elements + preparation
        for index_entry, preparation in enumerate(entry):
            preparations[index_preparations, index_entry] = preparations[index_preparations, index_entry] / sum_elements
    scaled_food_data_finalCategory1 = scaler.fit_transform(category)
    scaled_food_data_scalars1 = scaler.fit_transform(scalars)
    scaled_food_data_categories1 = scaler.fit_transform(categories)
    scaled_food_data_ingredients1 = scaler.fit_transform(ingredients)
    scaled_food_data_preparation1 = scaler.fit_transform(preparations)
    new_food_with_gauss_stand = np.concatenate([scaled_food_data_finalCategory1, scaled_food_data_categories1, scaled_food_data_scalars1, scaled_food_data_ingredients1, scaled_food_data_preparation1], axis=1)
    # PCA - I fill to_return2 as i filled to_return (the starting data are the same in the same order but after PCA
    new_food_with_gauss_stand_pca = pca_model.transform(new_food_with_gauss_stand)
    insert_counter = 0
    start_first = True
    exit_condition = False
    number_of_change = 0
    while True:
        if exit_condition:
            break
        if start_first:
            for first_index, first_raw_food in enumerate(new_food_with_gauss_stand_pca):
                if first_index < number_of_change:
                    continue
                for second_index, second_raw_food in enumerate(new_food_with_gauss_stand_pca):
                    if (first_index % 2 != 0) or (second_index % 2 == 0) or first_index >= second_index:  # in even i have duplicated of first food, in odd of second
                        continue
                    else:
                        to_return2[insert_counter] = np.concatenate((first_raw_food, second_raw_food))
                        insert_counter += 1
                        if insert_counter >= len(to_return2):
                            exit_condition = True
                            break
                start_first = False
                number_of_change += 1
                break
        else:
            for second_index, second_raw_food in enumerate(new_food_with_gauss_stand_pca):
                if second_index < number_of_change:
                    continue
                for first_index, first_raw_food in enumerate(new_food_with_gauss_stand_pca):
                    if (first_index % 2 != 0) or (second_index % 2 == 0) or second_index >= first_index:  # index are switched
                        continue
                    else:
                        to_return2[insert_counter] = np.concatenate((second_raw_food, first_raw_food))
                        insert_counter += 1
                        if insert_counter >= len(to_return2):
                            exit_condition = True
                            break
                number_of_change += 1
                start_first = True
                break

    return to_return, to_return2, new_food, indexes_of_samples


def prediction(user_id, samples_to_predict):
    model = keras.models.load_model('../../NN_data/models/User' + str(user_id) + '/folder_version')
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
    model = keras.models.load_model('../../NN_data/models/User' + str(user_id) + '/folder_version')
    to_predict = np.concatenate((first_food, second_food))
    to_predict_reshaped = to_predict.reshape(1, -1)
    predictions = model.predict(to_predict_reshaped)
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



data_dir = "../../dataset_100/separated_text_data/"

fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fNames = os.path.join(data_dir, 'names.txt')
fCouple = os.path.join('../../Ordinamenti/dataset_coppie.txt')
fLabels = os.path.join('../../Ordinamenti/output-file.txt')

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
all_data_original_1HE = np.concatenate([finalCategory, food_data_categories, food_data_scalars, food_data_ingredients, food_data_preparation], axis=1)

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

    X, X_PCA, samples, sample_indexes = find_samples_gaussian_noise(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=190, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass), pca_model=pca)

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

# for test i picked by hand 28 and 87, because they are near to symmetry with 9 and 47 (i picked the same with gaussian noise in order to compare results)
first_food_id_to_pass = 28
second_food_id_to_pass = 87
print("--------------------couple queried--------------------")
print(str(int(first_food_id_to_pass)) + "," + str(int(second_food_id_to_pass)))
print("--------------------prediction of i-th user's model on queried couple--------------------")
for i in range(0, 48):
    single_prediction(user_id=i, first_food=pca_data[int(first_food_id_to_pass)], second_food=pca_data[int(second_food_id_to_pass)])
print("--------------------generation samples for train--------------------")
print("NOTE: the following indexes are about data created with gaussian noise, so to refer to file data.csv in relative folder")
X, X_PCA, samples, sample_indexes = find_samples_gaussian_noise(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=190, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass), pca_model=pca)
to_print = ""
for row in sample_indexes:
    to_print += str(row[0]) + "," + str(row[1]) + ";"
print(to_print)
np.savetxt('./Data8Component2Std/NNoutput/train/210CouplesGaussianNoise/data.csv', samples, delimiter=';')
print("--------------------generation label for train--------------------")
for i in range(0, 48):
    prediction(user_id=i, samples_to_predict=X_PCA)
