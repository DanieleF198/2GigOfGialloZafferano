import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras

data_dir = "../dataset_100/separated_text_data/"

fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fNames = os.path.join(data_dir, 'names.txt')

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

while True:
    ingredients_index = [i for i in range(0, 36)]
    preparations_index = [i for i in range(0, 8)]

    print("--------------------- FOOD 1 ---------------------")
    print("insert category: 1- starter; 2- complete meal; 3- first course; 4- second course; 5- Savory Cake")
    category1 = input()
    print("insert cost: 1- very low; 2- low; 3- medium; 4- high; 5- very high; 0- absent")
    cost1 = input()
    print("insert difficulty: 1- very easy; 2- easy; 3- medium; 4- difficult; 0- absent")
    difficulty1 = input()
    print("insert preparation time:")
    prepTime1 = input()
    ingredients1 = np.zeros(36, dtype="float32")
    for index in ingredients_index:
        print("insert ingredient " + str(index))
        ingredients1[index] = input()
    preparations1 = np.zeros(8, dtype="float32")
    for index in preparations_index:
        print("insert preparation " + str(index))
        preparations1[index] = input()

    print("--------------------- FOOD 2 ---------------------")
    print("insert category: 1- starter; 2- complete meal; 3- first course; 4- second course; 5- Savory Cake")
    category2 = input()
    print("insert cost: 1- very low; 2- low; 3- medium; 4- high; 5- very high; 0- absent")
    cost2 = input()
    print("insert difficulty: 1- very easy; 2- easy; 3- medium; 4- difficult; 0- absent")
    difficulty2 = input()
    print("insert preparation time:")
    prepTime2 = input()
    ingredients2 = np.zeros(36, dtype="float32")
    for index in ingredients_index:
        print("insert ingredient " + str(index))
        ingredients2[index] = input()
    preparations2 = np.zeros(8, dtype="float32")
    for index in preparations_index:
        print("insert preparation " + str(index))
        preparations2[index] = input()

    food_data_scalars2 = np.zeros((2, 1), dtype='float32')
    food_data_scalars2[0, 0] = prepTime1
    food_data_scalars2[1, 0] = prepTime2

    food_data_categories2 = np.zeros((7, 3), dtype='float32')
    food_data_categories2[0, 0] = category1
    food_data_categories2[1, 0] = category2
    # just for 1HE
    food_data_categories2[2, 0] = 1
    food_data_categories2[3, 0] = 2
    food_data_categories2[4, 0] = 3
    food_data_categories2[5, 0] = 4
    food_data_categories2[6, 0] = 5
    food_data_categories2[0, 1] = cost1
    food_data_categories2[1, 1] = cost2
    food_data_categories2[0, 2] = difficulty1
    food_data_categories2[1, 2] = difficulty2

    food_data_ingredients2 = np.zeros((2, 36), dtype='float32')
    food_data_ingredients2[0] = ingredients1
    food_data_ingredients2[1] = ingredients2

    food_data_preparation2 = np.zeros((2, 8), dtype='float32')
    food_data_preparation2[0] = preparations1
    food_data_preparation2[1] = preparations2

    # manipulate categorical data
    category1HE2 = food_data_categories2[:, 0]
    food_data_categories2 = np.delete(food_data_categories2, 0, 1)
    for index_to_del in reversed(range(2, 7)):
        food_data_categories2 = np.delete(food_data_categories2, index_to_del, 0)
    category1HE2 = np.reshape(category1HE2, (7, 1))
    enc2 = OneHotEncoder(handle_unknown='ignore')
    enc2.fit(category1HE2)
    finalCategory2 = enc2.transform(category1HE2).toarray()
    for index_to_del in reversed(range(2, 7)):
        finalCategory2 = np.delete(finalCategory2, index_to_del, 0)

    # remove ingredients never used
    food_data_ingredients2 = np.delete(food_data_ingredients2, [1, 16, 23, 26, 32], axis=1)

    # normalize scaled ingredients and scaled preparation

    normalized_food_data_ingredients2 = np.zeros((2, 31), dtype="float32")
    normalized_food_data_preparation2 = np.zeros((2, 8), dtype="float32")

    for i, row in enumerate(food_data_ingredients2):
        sum_of_elements = 0
        for element in row:
            sum_of_elements = sum_of_elements + element
        for j, element in enumerate(row):
            normalized_food_data_ingredients2[i, j] = food_data_ingredients2[i, j] / sum_of_elements

    for i, row in enumerate(food_data_preparation2):
        sum_of_elements = 0
        for element in row:
            sum_of_elements = sum_of_elements + element
        for j, element in enumerate(row):
            normalized_food_data_preparation2[i, j] = food_data_preparation2[i, j] / sum_of_elements

    # scaling preparation time, cost, difficulty, ingredients and preparations
    scaler = preprocessing.StandardScaler()
    scaled_food_data_finalCategory2 = scaler.fit_transform(finalCategory2)
    scaled_food_data_scalars2 = scaler.fit_transform(food_data_scalars2)
    scaled_food_data_categories2 = scaler.fit_transform(food_data_categories2)
    scaled_food_data_ingredients2 = scaler.fit_transform(normalized_food_data_ingredients2)
    scaled_food_data_preparation2 = scaler.fit_transform(normalized_food_data_preparation2)

    # concatenate all data in a numpy tensor
    all_data2 = np.concatenate([scaled_food_data_finalCategory2, scaled_food_data_categories2, scaled_food_data_scalars2, scaled_food_data_ingredients2, scaled_food_data_preparation2], axis=1)

    # preparation of labels for pandas dataframe
    foodsLabel2 = ['F' + str(i) for i in range(1, 2)]
    categoryLabel2 = ['CATEGORY' + str(i) for i in range(1, 6)]
    ingredientsLabel2 = ['INGREDIENTS' + str(i) for i in range(1, 32)]
    preparationsLabel2 = ['PREPARATIONS' + str(i) for i in range(1, 9)]

    # creating pandas dataframe
    final_data2 = pd.DataFrame(columns=[*categoryLabel2, 'COST', 'DIFFICULTY', 'PREPARATION', *ingredientsLabel2, *preparationsLabel2], index=foodsLabel2)

    # fill pandas dataframe with data that we concatenated in line 83
    for i, food in enumerate(final_data2.index):
        final_data2.loc[food] = all_data2[i]

    very_final_data = np.concatenate((final_data, final_data2), axis=0)

    # instantiate pca object, then fit on final data
    # pca = PCA() #(we used this before to understand the number of component to use)
    pca = decomposition.PCA(n_components=17)
    pca.fit(very_final_data)
    # apply pca on data in pandas_dataframe
    pca_data = pca.transform(very_final_data)

    print("insert User ID")
    user_id = input()

    while True:
        model = keras.models.load_model('../NN_data/models/User' + str(user_id) + '/folder_version')
        to_predict = np.concatenate((pca_data[0], pca_data[1]))
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
        print("would you like to continue with the same couple but different user? 1- yes; 2- no")
        while True:
            answer = input()
            if not answer.isdigit():
                print("please insert 1 for \"yes\", 2 for \"no\"")
                continue
            if int(answer) != 1 and int(answer) != 2:
                print("please insert 1 for \"yes\", 2 for \"no\"")
                continue
            if int(answer) == 1 or int(answer) == 2:
                break
        if int(answer) == 1:
            print("insert User ID")
            user_id = input()
        if int(answer) == 2:
            break
    print("would you like to continue with different couple? 1- yes; 2- no")
    while True:
        answer = input()
        if not answer.isdigit():
            print("please insert 1 for \"yes\", 2 for \"no\"")
            continue
        if int(answer) != 1 and int(answer) != 2:
            print("please insert 1 for \"yes\", 2 for \"no\"")
            continue
        if int(answer) == 1 or int(answer) == 2:
            break
    if int(answer) == 2:
        break
