import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def prediction(user_id, first_food, second_food):
    model = keras.models.load_model('../NN_data/models/User' + str(user_id) + '/folder_version')
    to_predict = np.concatenate((first_food, second_food))
    X = to_predict.reshape(1, -1)
    predictions = model.predict(X)
    print("P(c = 1) = " + str(predictions[0, 0]))
    print("P(c = 0) = " + str(predictions[0, 1]))
    print("P(c = -1) = " + str(predictions[0, 2]))
    del model
    del to_predict
    del X
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

while True:
    print("insert the user_id, the first food_id and the second_food_id")
    user_id_to_pass = input()
    first_food_id_to_pass = input()
    second_food_id_to_pass = input()

    if not user_id_to_pass.isdigit() or not first_food_id_to_pass.isdigit() or not second_food_id_to_pass.isdigit():
        print("input must be integers")
        continue
    if int(user_id_to_pass) < 0 or int(user_id_to_pass) > 47:
        print("user_id must be 0 < user_id < 47")
        continue
    if int(first_food_id_to_pass) < 0 or int(first_food_id_to_pass) > 99:
        print("food_id must be 0 < food_id < 99")
        continue
    if int(second_food_id_to_pass) < 0 or int(second_food_id_to_pass) > 99:
        print("user_id must be 0 < food_id < 47")
        continue

    prediction(user_id=user_id_to_pass, first_food=pca_data[int(first_food_id_to_pass)], second_food=pca_data[int(second_food_id_to_pass)])

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
