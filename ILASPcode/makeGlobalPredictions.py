import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras
import random

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prediction(user_id, x):
    model = keras.models.load_model('../NN_data/models/User' + str(user_id) + '/folder_version')
    predictions = model(x)
    to_print = ""
    for index, predict in enumerate(predictions):
        if index >= 150:
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
    to_print = to_print[0:len(to_print)-1]
    print(to_print)
    del model
    del x
    del predictions


data_dir = "../dataset_100/separated_text_data/"

fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fNames = os.path.join(data_dir, 'names.txt')
fCouple = os.path.join('../Ordinamenti/dataset_coppie.txt')
fLabels = os.path.join('../Ordinamenti/output-file.txt')

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
food_couple = np.zeros((len(linesOfCC), 420, 2), dtype='int32')
for i, line in enumerate(linesOfCC):
    if i < 54:
        values = [x for x in line.split(';')[:]]
        values = values[:210]
        for j, value in enumerate(values):
            twoValues = value.split(',')
            first = int(twoValues[0])
            second = int(twoValues[1])
            food_couple[i, j, 0] = first
            food_couple[i, j, 1] = second
            food_couple[i, j + 210, 0] = second
            food_couple[i, j + 210, 1] = first

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

food_effective_couple = np.zeros((len(food_couple) - 1, 420, 2, 17))
for i, answer in enumerate(food_couple):
    if i < 54:
        for j, couple in enumerate(answer):
            for k, food in enumerate(couple):
                food_effective_couple[i, j, k, :] = pca_data[int(food)]

food_effective_couple = np.reshape(food_effective_couple, (len(food_couple)-1, 420, 34))

n_train = 10
n_test = 15     # 15 food implies 105 couple
recipes = random.sample(range(0, 100), n_train+n_test)
# recipes_train = recipes[0:n_train]
# recipes_test = recipes[n_train:n_train + n_test]
recipes_train = np.array([86, 88, 53, 29, 8, 26, 71, 81, 4, 11])
recipes_test = np.array([84, 15, 16, 18, 24, 50, 23, 77, 64, 21, 73, 48, 94, 67, 43])

recipes_train_couple = np.zeros((round((n_train*(n_train-1))/2), 2))
recipes_test_couple = np.zeros((round((n_test*(n_test-1))/2), 2))

train_dataset = np.zeros((round((n_train*(n_train-1))/2), 2, 17))
test_dataset = np.zeros((round((n_test*(n_test-1))/2), 2, 17))

counter = 0
for i, recipe in enumerate(recipes_train):
    for j, recipe2 in enumerate(recipes_train):
        if i == j:
            continue
        if i > j:
            continue
        recipes_train_couple[counter, 0] = recipe
        recipes_train_couple[counter, 1] = recipe2
        counter = counter + 1

counter = 0
for i, recipe in enumerate(recipes_test):
    for j, recipe2 in enumerate(recipes_test):
        if i == j:
            continue
        if i > j:
            continue
        recipes_test_couple[counter, 0] = recipe
        recipes_test_couple[counter, 1] = recipe2
        counter = counter + 1

for i, recipe_couple in enumerate(recipes_train_couple):
    for j, recipe in enumerate(recipe_couple):
        train_dataset[i, j, :] = pca_data[int(recipe)]

for i, recipe_couple in enumerate(recipes_test_couple):
    for j, recipe in enumerate(recipe_couple):
        test_dataset[i, j, :] = pca_data[int(recipe)]

train_dataset = np.reshape(train_dataset, ((round((n_train*(n_train-1))/2)), 34))
test_dataset = np.reshape(test_dataset, ((round((n_test*(n_test-1))/2)), 34))

while True:
    print("insert the user_id, pass special value -1 for create train and test for ILASP experiments")
    user_id_to_pass = input()

    if user_id_to_pass == "-1":
        break
    if not user_id_to_pass.isdigit():
        print("input must be integers")
        continue
    if int(user_id_to_pass) < 0 or int(user_id_to_pass) > 47:
        print("user_id must be 0 < user_id < 47")
        continue

    prediction(user_id=user_id_to_pass, x=food_effective_couple[int(user_id_to_pass), :])

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
print("--------------------generation samples for train--------------------")
to_print = ""
for couple in recipes_train_couple:
    to_print = to_print + str(int(couple[0])) + "," + str(int(couple[1])) + ";"
print(to_print)
print("--------------------generation label for train--------------------")
for i in range(48, 54):
    prediction(user_id=i, x=train_dataset)
# print("--------------------generation samples for test--------------------")
# to_print = ""
# for couple in recipes_test_couple:
#     to_print = to_print + str(int(couple[0])) + "," + str(int(couple[1])) + ";"
# print(to_print)
# print("--------------------generation label for test--------------------")
# for i in range(48, 54):
#     prediction(user_id=i, x=test_dataset)
