import math
import multiprocessing
import os
import pickle
import sys
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from mlxtend.plotting import plot_decision_regions
import re
from sklearn.model_selection import KFold
import warnings
import operator
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers
from keras import optimizers
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tempfile import TemporaryFile
import tensorflow as tf
import time

from datetime import datetime
from sklearn.utils import shuffle

# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

#reading data from external files
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.np_utils import to_categorical
def testing():
    # now = datetime.now()
    # year = now.strftime("%Y")
    # month = now.strftime("%m")
    # day = now.strftime("%d")
    # year = "2022"
    # month = "01"
    # day = "28"
    data_dir = "./dataset_100/separated_text_data/"
    #
    # if not os.path.exists("./NN_data/plots/User18/accuracy/result" + year + month + day):
    #     print("error, create folder")
    #     quit()

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
    food_couple = np.zeros((len(linesOfCC), 420, 2), dtype='int32')
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
                food_couple[i, j + 210, 0] = second
                food_couple[i, j + 210, 1] = first

    linesOfL = dataL.split('\n')
    couple_label = np.zeros((len(linesOfL), 420), dtype='int32')
    for i, line in enumerate(linesOfL):
        if i < 48:
            values = [x for x in line.split(' ')[:]]
            for j, value in enumerate(values):
                veryValue = int(value)
                if veryValue == 1:
                    couple_label[i, j] = 1
                    couple_label[i, j + 210] = 2
                elif veryValue == -1:
                    couple_label[i, j] = 2
                    couple_label[i, j + 210] = 1
                else:
                    couple_label[i, j] = 0
                    couple_label[i, j + 210] = 0

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


    # VERSION F1X | F2X | ... | FnX | F1Y | F2Y | ... | FnY |
    food_effective_couple = np.zeros((len(food_couple)-1, 420, 2, 17))  # len(food_couple) answer, each one with 210 couple made of 2 elements that each is rapresented by 19 feature (PCA)
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

    # # reverting couple to be of the form "A is/is not/equal preferred than B"
    # for i, user in enumerate(food_effective_couple):
    #     for j, couple in enumerate(user):
    #         SFF1 = 0 #sum of feature of food 1
    #         SFF2 = 0 #sum of feature of food 2
    #         for k, food in enumerate(couple):
    #             if k == 0:
    #                 for feature in food:
    #                     SFF1 += feature
    #                 continue
    #             else:
    #                 for feature in food:
    #                     SFF2 += feature
    #                 if SFF1 < SFF2:
    #                     temp = np.copy(food_effective_couple[i, j, 0])
    #                     food_effective_couple[i, j, 0] = food_effective_couple[i, j, 1]
    #                     food_effective_couple[i, j, 1] = temp
    #                     if couple_label[i, j] != 0:
    #                         if couple_label[i, j] == 1:
    #                             couple_label[i, j] = 2
    #                         else:
    #                             couple_label[i, j] = 1
    #                         # couple_label[i, j] *= -1

    food_effective_couple = np.reshape(food_effective_couple, (len(food_couple)-1, 420, 34))

    # food_effective_couple_zeros = np.zeros((len(food_effective_couple),420,34))
    # food_effective_couple_no_zeros = np.zeros((len(food_effective_couple),420,34))
    # couple_label_zeros = np.zeros((len(couple_label), 420))
    # couple_label_no_zeros = np.zeros((len(couple_label), 420))
    # counter_of_answer_with_zero = 0
    # counter_of_answer_without_zero = 0
    # for i, answer in enumerate(couple_label):
    #     if i >= 125:
    #         break
    #     zero_count = 0
    #     for element in answer:
    #         if element == 0:
    #             zero_count += 1
    #     if zero_count <= 10:
    #         inner_counter = 0
    #         for j, couple in enumerate(food_effective_couple[i]):
    #             if answer[j] == 1:
    #                 food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
    #                 couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = 1
    #                 inner_counter += 1
    #             elif answer[j] == 2:
    #                 food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
    #                 couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = 2
    #                 inner_counter += 1
    #             else:
    #                 continue
    #         counter_of_answer_without_zero += 1
    #     else:
    #         food_effective_couple_zeros[counter_of_answer_with_zero] = food_effective_couple[i]
    #         couple_label_zeros[counter_of_answer_with_zero] = answer
    #         counter_of_answer_with_zero += 1

    # food_effective_couple_zeros = food_effective_couple_zeros[0:counter_of_answer_with_zero]
    # couple_label_zeros = couple_label_zeros[0:counter_of_answer_with_zero]
    # food_effective_couple_no_zeros = food_effective_couple_no_zeros[0:counter_of_answer_without_zero]
    # couple_label_no_zeros = couple_label_no_zeros[0:counter_of_answer_without_zero]
    half = len(food_effective_couple) / 2
    half = round(half)

    optimizersValues = ["SGD"] # SGD, Adam
    lrValues = [0.0005]    # [0.01, 0.001, 0.0001, 0.0005, 0.00001]
    activationOfFirstLayers = ["tanh"]  # linear, relu, tanh, sigmoid, leakyRelu
    activationOfSecondLayers = ["relu"]
    activationOfThirdLayers = ["linear"]
    dropout_uses = [True]
    batch_normalization_uses = [True]
    dropout_parameters = [0.1]    #[0.1, 0.2, 0.5]
    nodes = [64]   #[32, 64, 128]
    epoch = 10000

    for k, row in enumerate(food_effective_couple):
        print("User" + str(k))
        X = scaler.fit_transform(food_effective_couple[k])
        Y = couple_label[k]

        counterone= 0
        counterzero = 0
        counterminusone = 0

        Y_enc = np.zeros((len(Y), 3), dtype="float32")
        for i, label in enumerate(Y):
            if label == 1:
                Y_enc[i, :] = [1, 0, 0]
                counterone = counterone + 1
            elif label == 0:
                Y_enc[i, :] = [0, 1, 0]
                counterzero = counterzero + 1
            else:
                Y_enc[i, :] = [0, 0, 1]
                counterminusone = counterminusone + 1

        # print("Class 1 = " + str(counterone) + "; Class 0 = " + str(counterzero) + "; Class -1 = " + str(counterminusone))
        #
        # continue

        for counterOpt, optimizerValue in enumerate(optimizersValues):
            for lrValue in lrValues:
                for counterOfActFLyr, activationOfFirstLayer in enumerate(activationOfFirstLayers):
                    for counterOfActSLyr, activationOfSecondLayer in enumerate(activationOfSecondLayers):
                        for counterOfActTLyr, activationOfThirdLayer in enumerate(activationOfThirdLayers):
                            for numberOfNodes in nodes:
                                for dropout_use in dropout_uses:
                                    for batch_normalization_use in batch_normalization_uses:
                                        for dropout_parameter in dropout_parameters:
                                            if (activationOfFirstLayer == "linear" and (activationOfSecondLayer == "linear" or activationOfThirdLayer == "linear")) or (activationOfSecondLayer == "linear" and (activationOfFirstLayer == "linear" or activationOfThirdLayer == "linear")) or (activationOfThirdLayer == "linear" and (activationOfFirstLayer == "linear" or activationOfSecondLayer == "linear")):
                                                continue
                                            if not dropout_use:     # go directly to last iteration of dropout_parameter because we don't consider it (and so don't waste time)
                                                if dropout_parameter == 0.1 or dropout_parameter == 0.2:
                                                    continue
                                            if k == 2:
                                                continue
                                            if k == 5:
                                                continue
                                            if k == 23:
                                                if lrValue == 0.01 or lrValue == 0.001:
                                                    continue
                                                elif lrValue == 0.0001:
                                                    if numberOfNodes == 32:
                                                        continue
                                                    elif numberOfNodes == 64:
                                                        if dropout_use:
                                                            continue
                                                        else:
                                                            if batch_normalization_use:
                                                                continue

                                            # kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # mod.: 31/01/2022 near 17:00
                                            X_train_temp, X_validation, Y_train_temp, Y_validation = train_test_split(X, Y_enc, shuffle=True, stratify=Y_enc, test_size=0.20, random_state=42)
                                            X_train, X_test, Y_train, Y_test = train_test_split(X_train_temp, Y_train_temp, shuffle=True, stratify=Y_train_temp, test_size=0.15, random_state=42)
                                            accuracy = []
                                            loss = []
                                            precision = []
                                            recall = []
                                            f1score = []

                                            # all_accuracy_history = []
                                            # all_loss_history = []
                                            # all_precision_history = []
                                            # all_recall_history = []
                                            # all_f1score_history = []
                                            # all_val_accuracy_history = []
                                            # all_val_loss_history = []
                                            # all_val_precision_history = []
                                            # all_val_recall_history = []
                                            # all_val_f1score_history = []

                                            # for train, test in kfold.split(X, Y):
                                            # X_train_fold, Y_train_fold = X[train], Y_enc[train]
                                            # X_test, Y_test = X[test], Y_enc[test]
                                            optimizerToEval = "optimizers." + optimizerValue + "(learning_rate=" + str(lrValue) + ")"
                                            optimizer = eval(optimizerToEval)

                                            model = models.Sequential()
                                            if activationOfFirstLayer == "leakyRelu":
                                                model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU(), input_shape=(34,)))
                                            else:
                                                model.add(layers.Dense(numberOfNodes, activation=activationOfFirstLayer, input_shape=(34,)))
                                            if dropout_use:
                                                model.add(layers.Dropout(dropout_parameter))
                                            if activationOfSecondLayer == "leakyRelu":
                                                model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                            else:
                                                model.add(layers.Dense(numberOfNodes, activation=activationOfSecondLayer))
                                            if dropout_use:
                                                model.add(layers.Dropout(dropout_parameter))
                                            if batch_normalization_use:
                                                model.add(layers.BatchNormalization())
                                            if activationOfThirdLayer == "leakyRelu":
                                                model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                            else:
                                                model.add(layers.Dense(numberOfNodes, activation=activationOfThirdLayer))
                                            if dropout_use:
                                                model.add(layers.Dropout(dropout_parameter))
                                            model.add(layers.Dense(3, activation="softmax"))
                                            metricPrecision = tf.keras.metrics.Precision()
                                            metricRecall = tf.keras.metrics.Recall()
                                            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', metricPrecision, metricRecall])
                                            # X_train, X_validation, Y_train, Y_validation = train_test_split(X_train_fold, Y_train_fold, test_size=0.20, random_state=42)
                                            history = model.fit(X_train, Y_train, epochs=epoch, batch_size=32, validation_data=(X_validation, Y_validation), verbose=0, shuffle=True)
                                            evaluation = model.evaluate(X_test, Y_test, verbose=0)
                                            loss.append(evaluation[0])
                                            accuracy.append(evaluation[1])
                                            precision.append(evaluation[2])
                                            recall.append(evaluation[3])
                                            f1score.append(2*(precision[len(precision)-1]*recall[len(precision)-1])/(precision[len(precision)-1]+recall[len(precision)-1]))
                                            tf.keras.models.save_model(model= model, filepath="./NN_data/models/User" + str(k) + "/folder_version", save_format='h5py')
                                            model.save("./NN_data/models/User" + str(k) + "/modelUser" + str(k) + ".h5")

                                            if ("precision" in history.history) and ("recall" in history.history):
                                                accuracy_history = history.history['accuracy']
                                                loss_history = history.history['loss']
                                                val_accuracy_history = history.history['val_accuracy']
                                                val_loss_history = history.history['val_loss']

                                                precision_history = history.history['precision']
                                                recall_history = history.history['recall']
                                                val_precision_history = history.history['val_precision']
                                                val_recall_history = history.history['val_recall']
                                            else:
                                                precisionAndRecallCounter = 1
                                                while True:
                                                    if ("precision_" + str(precisionAndRecallCounter) in history.history) and ("recall_" + str(precisionAndRecallCounter) in history.history):
                                                        accuracy_history = history.history['accuracy']
                                                        loss_history = history.history['loss']
                                                        val_accuracy_history = history.history['val_accuracy']
                                                        val_loss_history = history.history['val_loss']

                                                        precision_history = history.history['precision_' + str(precisionAndRecallCounter)]
                                                        recall_history = history.history['recall_' + str(precisionAndRecallCounter)]
                                                        val_precision_history = history.history['val_precision_' + str(precisionAndRecallCounter)]
                                                        val_recall_history = history.history['val_recall_' + str(precisionAndRecallCounter)]
                                                        break
                                                    else:
                                                        precisionAndRecallCounter += 1

                                            f1score_history = np.zeros((len(precision_history)), dtype='float32')
                                            val_f1score_history = np.zeros((len(val_precision_history)), dtype='float32')
                                            for i in range(epoch):
                                                if precision_history[i] + recall_history[i] == 0:
                                                    f1score_history[i] = 0
                                                else:
                                                    f1score_history[i] = 2 * ((precision_history[i] * recall_history[i]) / (precision_history[i] + recall_history[i]))
                                                if val_precision_history[i] + val_recall_history[i] == 0:
                                                    val_f1score_history[i] = 0
                                                else:
                                                    val_f1score_history[i] = 2 * ((val_precision_history[i] * val_recall_history[i]) / (val_precision_history[i] + val_recall_history[i]))

                                            # all_accuracy_history.append(accuracy_history)
                                            # all_loss_history.append(loss_history)
                                            # all_precision_history.append(precision_history)
                                            # all_recall_history.append(recall_history)
                                            # all_f1score_history.append(f1score_history)
                                            # all_val_accuracy_history.append(val_accuracy_history)
                                            # all_val_loss_history.append(val_loss_history)
                                            # all_val_precision_history.append(val_precision_history)
                                            # all_val_recall_history.append(val_recall_history)
                                            # all_val_f1score_history.append(val_f1score_history)
                                            #
                                            # average_accuracy_history = [np.mean([x[i] for x in all_accuracy_history]) for i in range(epoch)]
                                            # average_loss_history = [np.mean([x[i] for x in all_loss_history]) for i in range(epoch)]
                                            # average_precision_history = [np.mean([x[i] for x in all_precision_history]) for i in range(epoch)]
                                            # average_recall_history = [np.mean([x[i] for x in all_recall_history]) for i in range(epoch)]
                                            # average_f1score_history = [np.mean([x[i] for x in all_f1score_history]) for i in range(epoch)]
                                            #
                                            # average_val_accuracy_history = [np.mean([x[i] for x in all_val_accuracy_history]) for i in range(epoch)]
                                            # average_val_loss_history = [np.mean([x[i] for x in all_val_loss_history]) for i in range(epoch)]
                                            # average_val_precision_history = [np.mean([x[i] for x in all_val_precision_history]) for i in range(epoch)]
                                            # average_val_recall_history = [np.mean([x[i] for x in all_val_recall_history]) for i in range(epoch)]
                                            # average_val_f1score_history = [np.mean([x[i] for x in all_val_f1score_history]) for i in range(epoch)]

                                            plt.clf()
                                            plt.plots(range(1, len(accuracy_history) + 1), accuracy_history, label='accuracy')
                                            plt.plots(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='val_accuracy')
                                            plt.xlabel('Epochs')
                                            plt.ylabel('training and validation accuracy')
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                                else:
                                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            elif batch_normalization_use:
                                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            else:
                                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            plt.legend()
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/accuracy/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                                else:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/accuracy/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            elif batch_normalization_use:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/accuracy/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            else:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/accuracy/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                            plt.clf()
                                            plt.plots(range(1, len(loss_history) + 1), loss_history, label='loss')
                                            plt.plots(range(1, len(val_loss_history) + 1), val_loss_history, label='val_loss')
                                            plt.xlabel('Epochs')
                                            plt.ylabel('training and validation loss')
                                            plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            plt.legend()
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/loss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                                else:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/loss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            elif batch_normalization_use:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/loss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            else:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/loss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                            plt.clf()
                                            plt.plots(range(1, len(precision_history) + 1), precision_history, label='precision')
                                            plt.plots(range(1, len(val_precision_history) + 1), val_precision_history, label='val_precision')
                                            plt.xlabel('Epochs')
                                            plt.ylabel('training and validation precision')
                                            plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            plt.legend()
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/precision/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                                else:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/precision/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            elif batch_normalization_use:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/precision/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            else:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/precision/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                            plt.clf()
                                            plt.plots(range(1, len(recall_history) + 1), recall_history, label='recall')
                                            plt.plots(range(1, len(val_recall_history) + 1), val_recall_history, label='val_recall')
                                            plt.xlabel('Epochs')
                                            plt.ylabel('training and validation recall')
                                            plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            plt.legend()
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/recall/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                                else:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/recall/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            elif batch_normalization_use:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/recall/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            else:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/recall/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                            plt.clf()
                                            plt.plots(range(1, len(f1score_history) + 1), f1score_history, label='f1score')
                                            plt.plots(range(1, len(val_f1score_history) + 1), val_f1score_history, label='val_f1score')
                                            plt.xlabel('Epochs')
                                            plt.ylabel('training and validation f1score')
                                            plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                            plt.legend()
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/f1score/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                                else:
                                                    plt.savefig("./NN_data/plots/User" + str(k) + "/f1score/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            elif batch_normalization_use:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/f1score/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
                                            else:
                                                plt.savefig("./NN_data/plots/User" + str(k) + "/f1score/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)


                                            mn_acc = np.mean(accuracy)
                                            mn_loss = np.mean(loss)
                                            mn_prec = np.mean(precision)
                                            mn_rec = np.mean(recall)
                                            mn_f1s = np.mean(f1score)
                                            if dropout_use:
                                                if batch_normalization_use:
                                                    print("Model: " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ". Accuracy: " + str(mn_acc) + ", loss: " + str(mn_loss) + ", precision: " + str(mn_prec) + ", recall: " + str(mn_rec) + ", f1score: " + str(mn_f1s))
                                                else:
                                                    print("Model: " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-Yes_Drop" + str(dropout_parameter) + "_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ". Accuracy: " + str(mn_acc) + ", loss: " + str(mn_loss) + ", precision: " + str(mn_prec) + ", recall: " + str(mn_rec) + ", f1score: " + str(mn_f1s))
                                            elif batch_normalization_use:
                                                print("Model: " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_Yes_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ". Accuracy: " + str(mn_acc) + ", loss: " + str(mn_loss) + ", precision: " + str(mn_prec) + ", recall: " + str(mn_rec) + ", f1score: " + str(mn_f1s))
                                            else:
                                                print("Model: " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-No_Drop_No_batch-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ". Accuracy: " + str(mn_acc) + ", loss: " + str(mn_loss) + ", precision: " + str(mn_prec) + ", recall: " + str(mn_rec) + ", f1score: " + str(mn_f1s))

    return True


def program(userNumber):
    # if userNumber > 24:
    #     return True
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
    food_couple = np.zeros((len(linesOfCC), 420, 2), dtype='int32')
    for i, line in enumerate(linesOfCC):
        if i <125:
            values = [x for x in line.split(';')[:]]
            values = values[:210]
            for j, value in enumerate(values):
                twoValues = value.split(',')
                first = int(twoValues[0])
                second = int(twoValues[1])
                food_couple[i, j, 0] = first
                food_couple[i, j, 1] = second
                food_couple[i, j+210, 0] = second
                food_couple[i, j+210, 1] = first

    linesOfL = dataL.split('\n')
    couple_label = np.zeros((len(linesOfL), 420), dtype='int32')
    for i, line in enumerate(linesOfL):
        if i < 125:
            values = [x for x in line.split(' ')[:]]
            for j, value in enumerate(values):
                veryValue = int(value)
                if veryValue == 1:
                    couple_label[i, j] = 1
                    couple_label[i, j + 210] = 2
                elif veryValue == -1:
                    couple_label[i, j] = 2
                    couple_label[i, j + 210] = 1
                else:
                    couple_label[i, j] = 0
                    couple_label[i, j+210] = 0

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


    # VERSION F1X | F2X | ... | FnX | F1Y | F2Y | ... | FnY |
    food_effective_couple = np.zeros((len(food_couple)-1, 420, 2, 17))  # len(food_couple) answer, each one with 210 couple made of 2 elements that each is rapresented by 19 feature (PCA)
    for i, answer in enumerate(food_couple):
        if i < 125:
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

    # # reverting couple to be of the form "A is/is not/equal preferred than B"
    # for i, user in enumerate(food_effective_couple):
    #     for j, couple in enumerate(user):
    #         SFF1 = 0 #sum of feature of food 1
    #         SFF2 = 0 #sum of feature of food 2
    #         for k, food in enumerate(couple):
    #             if k == 0:
    #                 for feature in food:
    #                     SFF1 += feature
    #                 continue
    #             else:
    #                 for feature in food:
    #                     SFF2 += feature
    #                 if SFF1 < SFF2:
    #                     temp = np.copy(food_effective_couple[i, j, 0])
    #                     food_effective_couple[i, j, 0] = food_effective_couple[i, j, 1]
    #                     food_effective_couple[i, j, 1] = temp
    #                     if couple_label[i, j] != 0:
    #                         if couple_label[i, j] == 1:
    #                             couple_label[i, j] = 2
    #                         else:
    #                             couple_label[i, j] = 1
    #                         # couple_label[i, j] *= -1

    food_effective_couple = np.reshape(food_effective_couple, (len(food_couple) - 1, 420, 34))

    # food_effective_couple_zeros = np.zeros((len(food_effective_couple),420,34))
    # food_effective_couple_no_zeros = np.zeros((len(food_effective_couple),420,34))
    # couple_label_zeros = np.zeros((len(couple_label), 420))
    # couple_label_no_zeros = np.zeros((len(couple_label), 420))
    # counter_of_answer_with_zero = 0
    # counter_of_answer_without_zero = 0
    # for i, answer in enumerate(couple_label):
    #     if i >= 125:
    #         break
    #     zero_count = 0
    #     for element in answer:
    #         if element == 0:
    #             zero_count += 1
    #     if zero_count <= 10:
    #         inner_counter = 0
    #         for j, couple in enumerate(food_effective_couple[i]):
    #             if answer[j] == 1:
    #                 food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
    #                 couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = 1
    #                 inner_counter += 1
    #             elif answer[j] == 2:
    #                 food_effective_couple_no_zeros[counter_of_answer_without_zero, inner_counter] = couple
    #                 couple_label_no_zeros[counter_of_answer_without_zero, inner_counter] = 2
    #                 inner_counter += 1
    #             else:
    #                 continue
    #         counter_of_answer_without_zero += 1
    #     else:
    #         food_effective_couple_zeros[counter_of_answer_with_zero] = food_effective_couple[i]
    #         couple_label_zeros[counter_of_answer_with_zero] = answer
    #         counter_of_answer_with_zero += 1

    # food_effective_couple_zeros = food_effective_couple_zeros[0:counter_of_answer_with_zero]
    # couple_label_zeros = couple_label_zeros[0:counter_of_answer_with_zero]
    # food_effective_couple_no_zeros = food_effective_couple_no_zeros[0:counter_of_answer_without_zero]
    # couple_label_no_zeros = couple_label_no_zeros[0:counter_of_answer_without_zero]
    half = len(food_effective_couple)/2
    half = round(half)

    optimizersValues = ["SGD"]
    lrValues = [0.0005]
    activationOfFirstLayers = ["relu", "tanh"]
    activationOfSecondLayers = ["relu", "tanh"]
    activationOfThirdLayers = ["relu", "tanh"]
    nodes = [64]
    epoch = 10000


    for k, row in enumerate(food_effective_couple):
        # if k != userNumber:
        #     continue
        X = scaler.fit_transform(food_effective_couple[k])
        Y = couple_label[k]

        Y_enc = np.zeros((len(Y), 3), dtype="float32")
        for i, label in enumerate(Y):
            if label == 1:
                Y_enc[i, :] = [1, 0, 0]
            elif label == 0:
                Y_enc[i, :] = [0, 1, 0]
            else:
                Y_enc[i, :] = [0, 0, 1]

        X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y_enc, shuffle=True, stratify=Y_enc, test_size=0.20, random_state=42)

        #the temp are used in case we need to validate also with gaussian noise
        X_train_temp = X_train
        X_validation_temp = X_validation
        Y_train_temp = Y_train
        Y_validation_temp = Y_validation

        #count the number of element of each class to define how to do data augmentation

        train_classes = [0, 0, 0]
        validation_classes = [0, 0, 0]

        for index in range(0, 3):
            for index_two, tuple in enumerate(Y_train):
                train_classes[index] += Y_train[index_two, index]
            for index_two, tuple in enumerate(Y_validation):
                validation_classes[index] += Y_validation[index_two, index]

        # on first 62 users the number of those with more than:
        #    10 zeros => 59
        #    20 zeros => 53
        #    30 zeros => 48
        #    40 zeros => 39 *
        #    50 zeros => 23
        #    60 zeros => 12
        #    70 zeros => 9
        #    80 zeros => 6
        #    90 zeros => 3
        #   100 zeros => 3
        #   110 zeros => 3
        #   120 zeros => 2
        #   130 zeros => 2
        #   140 zeros => 0
        if (train_classes[1]+validation_classes[1]) <= 40:
            # case where it's needed to evaluate model with and without gaussian noise (if this not the case the if body won't be executed and only evaluate model without noise)

            # in this case, cause we're considerate all matrix of preferences, the number of element of class 1 it's almost* the same of the number of element of class -1, so we only have to augmentate the element of class 0
            # "almost" and not "equal" casue we divide in validation and training, and the number of element of class 1 (or -1) could be more than another class by one element.

            max_of_element_for_class_train = max(train_classes)
            max_of_element_for_class_validation = max(validation_classes)

            difference_with_zero_train = max_of_element_for_class_train - train_classes[1]
            difference_with_zero_validation = max_of_element_for_class_validation - validation_classes[1]

            number_of_iteration_train = round(difference_with_zero_train / train_classes[1])
            number_of_iteration_validation = round(difference_with_zero_validation / validation_classes[1])

            # decompose the dataset, augmentate the element of class 0 and rebuild the dataset:

            # decompose

            couple_wich_preferred_train = np.zeros((int(train_classes[0]), 34), dtype='float32')
            couple_wich_indifferent_train = np.zeros((number_of_iteration_train + 1, int(train_classes[1]), 34), dtype='float32')
            couple_wich_not_preferred_train = np.zeros((int(train_classes[2]), 34), dtype='float32')

            couple_wich_preferred_validation = np.zeros((int(validation_classes[0]), 34), dtype='float32')
            couple_wich_indifferent_validation = np.zeros((number_of_iteration_validation + 1, int(validation_classes[1]), 34), dtype='float32')
            couple_wich_not_preferred_validation = np.zeros((int(validation_classes[2]), 34), dtype='float32')

            counter_of_preferred_train = 0
            counter_of_indifferent_train = 0
            counter_of_not_preferred_train = 0
            counter_of_preferred_validation = 0
            counter_of_indifferent_validation = 0
            counter_of_not_preferred_validation = 0

            for i, tuple in enumerate(Y_train_temp):
                for j, label in enumerate(tuple):
                    if label == 1 and j == 0:
                        couple_wich_preferred_train[counter_of_preferred_train] = X_train_temp[i]
                        counter_of_preferred_train += 1
                    elif label == 1 and j == 1:
                        couple_wich_indifferent_train[0, counter_of_indifferent_train] = X_train_temp[i]
                        counter_of_indifferent_train += 1
                    elif label == 1 and j == 2:
                        couple_wich_not_preferred_train[counter_of_not_preferred_train] = X_train_temp[i]
                        counter_of_not_preferred_train += 1

            for i, tuple in enumerate(Y_validation_temp):
                for j, label in enumerate(tuple):
                    if label == 1 and j == 0:
                        couple_wich_preferred_validation[counter_of_preferred_validation] = X_validation_temp[i]
                        counter_of_preferred_validation += 1
                    elif label == 1 and j == 1:
                        couple_wich_indifferent_validation[0, counter_of_indifferent_validation] = X_validation_temp[i]
                        counter_of_indifferent_validation += 1
                    elif label == 1 and j == 2:
                        couple_wich_not_preferred_validation[counter_of_not_preferred_validation] = X_validation_temp[i]
                        counter_of_not_preferred_validation += 1

            # augmented

            for i in range(1, number_of_iteration_train + 1):
                noise = tf.random.normal(shape=(tf.shape(couple_wich_indifferent_train[0])), mean=0.0, stddev=0.03, dtype="float32")
                couple_wich_indifferent_train[i] = np.array(tf.add(couple_wich_indifferent_train[0], noise))
            couple_wich_indifferent_train = np.reshape(couple_wich_indifferent_train, (len(couple_wich_indifferent_train) * (int(train_classes[1])), 34))

            for i in range(1, number_of_iteration_validation + 1):
                noise = tf.random.normal(shape=(tf.shape(couple_wich_indifferent_validation[0])), mean=0.0, stddev=0.03, dtype="float32")
                couple_wich_indifferent_validation[i] = np.array(tf.add(couple_wich_indifferent_validation[0], noise))
            couple_wich_indifferent_validation = np.reshape(couple_wich_indifferent_validation, (len(couple_wich_indifferent_validation) * (int(validation_classes[1])), 34))

            # rebuild

            sumForTrainShape = (counter_of_preferred_train + counter_of_indifferent_train + counter_of_not_preferred_train)
            sumForValidationShape = (counter_of_preferred_validation + counter_of_indifferent_validation + counter_of_not_preferred_validation)

            X_train_not_shuffled = np.zeros((sumForTrainShape, 34), dtype='float32')
            X_validation_not_shuffled = np.zeros((sumForValidationShape, 34), dtype='float32')

            Y_train_not_shuffled = np.zeros((sumForTrainShape, 3), dtype='float32')
            Y_validation_not_shuffled = np.zeros((sumForValidationShape, 3), dtype='float32')

            second_counter_of_preferred_train = 0
            second_counter_of_indifferent_train = 0
            second_counter_of_not_preferred_train = 0
            second_counter_of_preferred_validation = 0
            second_counter_of_indifferent_validation = 0
            second_counter_of_not_preferred_validation = 0

            for counter in range(0, counter_of_preferred_train):
                X_train_not_shuffled[counter] = couple_wich_preferred_train[second_counter_of_preferred_train]
                second_counter_of_preferred_train += 1
                Y_train_not_shuffled[counter] = [1, 0, 0]
            for counter in range(counter_of_preferred_train, counter_of_preferred_train + counter_of_indifferent_train):
                X_train_not_shuffled[counter] = couple_wich_indifferent_train[second_counter_of_indifferent_train]
                second_counter_of_indifferent_train += 1
                Y_train_not_shuffled[counter] = [0, 1, 0]
            for counter in range(counter_of_preferred_train + counter_of_indifferent_train, counter_of_preferred_train + counter_of_indifferent_train + counter_of_not_preferred_train):
                X_train_not_shuffled[counter] = couple_wich_not_preferred_train[second_counter_of_not_preferred_train]
                second_counter_of_not_preferred_train += 1
                Y_train_not_shuffled[counter] = [0, 0, 1]

            for counter in range(0, counter_of_preferred_validation):
                X_validation_not_shuffled[counter] = couple_wich_preferred_validation[second_counter_of_preferred_validation]
                second_counter_of_preferred_validation += 1
                Y_validation_not_shuffled[counter] = [1, 0, 0]
            for counter in range(counter_of_preferred_validation, counter_of_preferred_validation + counter_of_indifferent_validation):
                X_validation_not_shuffled[counter] = couple_wich_indifferent_validation[second_counter_of_indifferent_validation]
                second_counter_of_indifferent_validation += 1
                Y_validation_not_shuffled[counter] = [0, 1, 0]
            for counter in range(counter_of_preferred_validation + counter_of_indifferent_validation, counter_of_preferred_validation + counter_of_indifferent_validation + counter_of_not_preferred_validation):
                X_validation_not_shuffled[counter] = couple_wich_not_preferred_validation[second_counter_of_not_preferred_validation]
                second_counter_of_not_preferred_validation += 1
                Y_validation_not_shuffled[counter] = [0, 0, 1]

            X_train_for_gauss, Y_train_for_gauss = shuffle(X_train_not_shuffled, Y_train_not_shuffled)
            X_validation_for_gauss, Y_validation_for_gauss = shuffle(X_validation_not_shuffled, Y_validation_not_shuffled)

            for counterOpt, optimizerValue in enumerate(optimizersValues):
                for lrValue in lrValues:
                    for counterOfActFLyr, activationOfFirstLayer in enumerate(activationOfFirstLayers):
                        for counterOfActSLyr, activationOfSecondLayer in enumerate(activationOfSecondLayers):
                            for counterOfActTLyr, activationOfThirdLayer in enumerate(activationOfThirdLayers):
                                for numberOfNodes in nodes:

                                    optimizerToEval = "optimizers." + optimizerValue + "(learning_rate=" + str(lrValue) + ")"
                                    optimizer = eval(optimizerToEval)

                                    model = models.Sequential()
                                    if activationOfFirstLayer == "leakyRelu":
                                        model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU(), input_shape=(34,)))
                                    else:
                                        model.add(layers.Dense(numberOfNodes, activation=activationOfFirstLayer, input_shape=(34,)))
                                    model.add(layers.Dropout(0.1))
                                    if activationOfSecondLayer == "leakyRelu":
                                        model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                    else:
                                        model.add(layers.Dense(numberOfNodes, activation=activationOfSecondLayer))
                                    model.add(layers.Dropout(0.1))
                                    model.add(layers.BatchNormalization())
                                    if activationOfThirdLayer == "leakyRelu":
                                        model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                    else:
                                        model.add(layers.Dense(numberOfNodes, activation=activationOfThirdLayer))
                                    model.add(layers.Dropout(0.1))
                                    model.add(layers.Dense(3, activation="softmax"))
                                    metricPrecision = tf.keras.metrics.Precision()
                                    metricRecall = tf.keras.metrics.Recall()
                                    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', metricPrecision, metricRecall])

                                    history = model.fit(X_train_for_gauss, Y_train_for_gauss, epochs=epoch, batch_size=32, validation_data=(X_validation_for_gauss, Y_validation_for_gauss), verbose=0, shuffle=True)

                                    with open('./finalExperiment/UsersForValidation/TrainHistory/User' + str(k) + '/10000EpochFinalExperimentsGauss/' + activationOfFirstLayer + '-' + activationOfSecondLayer + '-' + activationOfThirdLayer + '-softmax_LR-' + str(lrValue) + '_OPT-' + optimizerValue + '_Nodes-' + str(numberOfNodes) + '.npy', 'wb') as fileNumpy:
                                        np.save(fileNumpy, history.history)

                                    if ("precision" in history.history) and ("recall" in history.history):
                                        accuracy_history = history.history['accuracy']
                                        loss_history = history.history['loss']
                                        val_accuracy_history = history.history['val_accuracy']
                                        val_loss_history = history.history['val_loss']

                                        precision_history = history.history['precision']
                                        recall_history = history.history['recall']
                                        val_precision_history = history.history['val_precision']
                                        val_recall_history = history.history['val_recall']
                                    else:
                                        precisionAndRecallCounter = 1
                                        while True:
                                            if ("precision_" + str(precisionAndRecallCounter) in history.history) and ("recall_" + str(precisionAndRecallCounter) in history.history):
                                                accuracy_history = history.history['accuracy']
                                                loss_history = history.history['loss']
                                                val_accuracy_history = history.history['val_accuracy']
                                                val_loss_history = history.history['val_loss']

                                                precision_history = history.history['precision_' + str(precisionAndRecallCounter)]
                                                recall_history = history.history['recall_' + str(precisionAndRecallCounter)]
                                                val_precision_history = history.history['val_precision_' + str(precisionAndRecallCounter)]
                                                val_recall_history = history.history['val_recall_' + str(precisionAndRecallCounter)]
                                                break
                                            else:
                                                precisionAndRecallCounter += 1

                                    f1score_history = [0 for i in range(epoch)]
                                    val_f1score_history = [0 for i in range(epoch)]
                                    for i in range(epoch):
                                        if precision_history[i] + recall_history[i] == 0:
                                            f1score_history[i] = 0
                                        else:
                                            f1score_history[i] = 2 * ((precision_history[i] * recall_history[i]) / (precision_history[i] + recall_history[i]))
                                        if val_precision_history[i] + val_recall_history[i] == 0:
                                            val_f1score_history[i] = 0
                                        else:
                                            val_f1score_history[i] = 2 * ((val_precision_history[i] * val_recall_history[i]) / (val_precision_history[i] + val_recall_history[i]))

                                    plt.clf()
                                    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='accuracy')
                                    plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='val_accuracy')
                                    plt.xlabel('Epochs')
                                    plt.ylabel('training and validation accuracy')
                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                    plt.legend()
                                    plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/accuracy/10000EpochFinalExperimentsGauss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                    plt.clf()
                                    plt.plot(range(1, len(loss_history) + 1), loss_history, label='loss')
                                    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val_loss')
                                    plt.xlabel('Epochs')
                                    plt.ylabel('training and validation loss')
                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                    plt.legend()
                                    plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/loss/10000EpochFinalExperimentsGauss/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                    plt.clf()
                                    plt.plot(range(1, len(precision_history) + 1), precision_history, label='precision')
                                    plt.plot(range(1, len(val_precision_history) + 1), val_precision_history, label='val_precision')
                                    plt.xlabel('Epochs')
                                    plt.ylabel('training and validation precision')
                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                    plt.legend()
                                    plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/precision/10000EpochFinalExperimentsGauss/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                    plt.clf()
                                    plt.plot(range(1, len(recall_history) + 1), recall_history, label='recall')
                                    plt.plot(range(1, len(val_recall_history) + 1), val_recall_history, label='val_recall')
                                    plt.xlabel('Epochs')
                                    plt.ylabel('training and validation recall')
                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                    plt.legend()
                                    plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/recall/10000EpochFinalExperimentsGauss/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                    plt.clf()
                                    plt.plot(range(1, len(f1score_history) + 1), f1score_history, label='f1score')
                                    plt.plot(range(1, len(val_f1score_history) + 1), val_f1score_history, label='val_f1score')
                                    plt.xlabel('Epochs')
                                    plt.ylabel('training and validation f1score')
                                    plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                    plt.legend()
                                    plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/f1score/10000EpochFinalExperimentsGauss/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

        with open('./finalExperiment/UsersForValidation/proportions/User' + str(k) + '/proportions.txt', 'w') as fileTxt:
            fileTxt.write("User" + str(k) + "Training: " + str(len(X_train)) + ", Validation: " + str(len(X_validation)) + "\n")
            fileTxt.write("Training - Class 1 = " + str(train_classes[0]) + "; Class 0 = " + str(train_classes[1]) + "; Class -1 = " + str(train_classes[2]) + "\n")
            fileTxt.write("Validation - Class 1 = " + str(validation_classes[0]) + "; Class 0 = " + str(validation_classes[1]) + "; Class -1 = " + str(validation_classes[2]) + "\n")
            fileTxt.write("\n")

        for counterOpt, optimizerValue in enumerate(optimizersValues):
            for lrValue in lrValues:
                for counterOfActFLyr, activationOfFirstLayer in enumerate(activationOfFirstLayers):
                    for counterOfActSLyr, activationOfSecondLayer in enumerate(activationOfSecondLayers):
                        for counterOfActTLyr, activationOfThirdLayer in enumerate(activationOfThirdLayers):
                            for numberOfNodes in nodes:

                                optimizerToEval = "optimizers." + optimizerValue + "(learning_rate=" + str(lrValue) + ")"
                                optimizer = eval(optimizerToEval)

                                model = models.Sequential()
                                if activationOfFirstLayer == "leakyRelu":
                                    model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU(), input_shape=(34,)))
                                else:
                                    model.add(layers.Dense(numberOfNodes, activation=activationOfFirstLayer, input_shape=(34,)))
                                model.add(layers.Dropout(0.1))
                                if activationOfSecondLayer == "leakyRelu":
                                    model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                else:
                                    model.add(layers.Dense(numberOfNodes, activation=activationOfSecondLayer))
                                model.add(layers.Dropout(0.1))
                                model.add(layers.BatchNormalization())
                                if activationOfThirdLayer == "leakyRelu":
                                    model.add(layers.Dense(numberOfNodes, activation=tf.keras.layers.LeakyReLU()))
                                else:
                                    model.add(layers.Dense(numberOfNodes, activation=activationOfThirdLayer))
                                model.add(layers.Dropout(0.1))
                                model.add(layers.Dense(3, activation="softmax"))
                                metricPrecision = tf.keras.metrics.Precision()
                                metricRecall = tf.keras.metrics.Recall()
                                model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', metricPrecision, metricRecall])
                                history = model.fit(X_train, Y_train, epochs=epoch, batch_size=32, validation_data=(X_validation, Y_validation), verbose=0, shuffle=True)

                                with open('./finalExperiment/UsersForValidation/TrainHistory/User' + str(k) + '/10000EpochFinalExperiments/' + activationOfFirstLayer + '-' + activationOfSecondLayer + '-' + activationOfThirdLayer + '-softmax_LR-' + str(lrValue) + '_OPT-' + optimizerValue + '_Nodes-' + str(numberOfNodes) + '.npy', 'wb') as fileNumpy:
                                    np.save(fileNumpy, history.history)
                                if ("precision" in history.history) and ("recall" in history.history):
                                    accuracy_history = history.history['accuracy']
                                    loss_history = history.history['loss']
                                    val_accuracy_history = history.history['val_accuracy']
                                    val_loss_history = history.history['val_loss']

                                    precision_history = history.history['precision']
                                    recall_history = history.history['recall']
                                    val_precision_history = history.history['val_precision']
                                    val_recall_history = history.history['val_recall']
                                else:
                                    precisionAndRecallCounter = 1
                                    while True:
                                        if ("precision_"+str(precisionAndRecallCounter) in history.history) and ("recall_"+str(precisionAndRecallCounter) in history.history):
                                            accuracy_history = history.history['accuracy']
                                            loss_history = history.history['loss']
                                            val_accuracy_history = history.history['val_accuracy']
                                            val_loss_history = history.history['val_loss']

                                            precision_history = history.history['precision_'+str(precisionAndRecallCounter)]
                                            recall_history = history.history['recall_'+str(precisionAndRecallCounter)]
                                            val_precision_history = history.history['val_precision_'+str(precisionAndRecallCounter)]
                                            val_recall_history = history.history['val_recall_'+str(precisionAndRecallCounter)]
                                            break
                                        else:
                                            precisionAndRecallCounter += 1

                                f1score_history = np.zeros((len(precision_history)), dtype='float32')
                                val_f1score_history = np.zeros((len(val_precision_history)), dtype='float32')
                                for i in range(epoch):
                                    if precision_history[i] + recall_history[i] == 0:
                                        f1score_history[i] = 0
                                    else:
                                        f1score_history[i] = 2 * ((precision_history[i] * recall_history[i]) / (precision_history[i] + recall_history[i]))
                                    if val_precision_history[i] + val_recall_history[i] == 0:
                                        val_f1score_history[i] = 0
                                    else:
                                        val_f1score_history[i] = 2 * ((val_precision_history[i] * val_recall_history[i]) / (val_precision_history[i] + val_recall_history[i]))

                                plt.clf()
                                plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='accuracy')
                                plt.plot(range(1, len(val_accuracy_history) + 1), val_accuracy_history, label='val_accuracy')
                                plt.xlabel('Epochs')
                                plt.ylabel('training and validation accuracy')
                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                plt.legend()
                                plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/accuracy/10000EpochFinalExperiments/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                plt.clf()
                                plt.plot(range(1, len(loss_history) + 1), loss_history, label='loss')
                                plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='val_loss')
                                plt.xlabel('Epochs')
                                plt.ylabel('training and validation loss')
                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                plt.legend()
                                plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/loss/10000EpochFinalExperiments/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                plt.clf()
                                plt.plot(range(1, len(precision_history) + 1), precision_history, label='precision')
                                plt.plot(range(1, len(val_precision_history) + 1), val_precision_history, label='val_precision')
                                plt.xlabel('Epochs')
                                plt.ylabel('training and validation precision')
                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                plt.legend()
                                plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/precision/10000EpochFinalExperiments/" + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                plt.clf()
                                plt.plot(range(1, len(recall_history) + 1), recall_history, label='recall')
                                plt.plot(range(1, len(val_recall_history) + 1), val_recall_history, label='val_recall')
                                plt.xlabel('Epochs')
                                plt.ylabel('training and validation recall')
                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                plt.legend()
                                plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/recall/10000EpochFinalExperiments/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)

                                plt.clf()
                                plt.plot(range(1, len(f1score_history) + 1), f1score_history, label='f1score')
                                plt.plot(range(1, len(val_f1score_history) + 1), val_f1score_history, label='val_f1score')
                                plt.xlabel('Epochs')
                                plt.ylabel('training and validation f1score')
                                plt.title("user" + str(k) + " " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax LR=" + str(lrValue) + " OPT=" + optimizerValue + "_Nodes-" + str(numberOfNodes))
                                plt.legend()
                                plt.savefig("./finalExperiment/UsersForValidation/plots/User" + str(k) + "/f1score/10000EpochFinalExperiments/ " + activationOfFirstLayer + "-" + activationOfSecondLayer + "-" + activationOfThirdLayer + "-softmax_LR-" + str(lrValue) + "_OPT-" + optimizerValue + "_Nodes-" + str(numberOfNodes) + ".jpg", dpi=300)
    return True


if __name__ == '__main__':
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
        numberOfUsers = [i for i in range(0, 25)]
        counterOfUserLeft = 0
        counterOfUserRight = 4
        while True:
            start = time.time()
            p = multiprocessing.Pool(4)
            results = [p.apply_async(program, args=(x,)) for x in range(counterOfUserLeft, counterOfUserRight)]
            output = [p.get() for p in results]
            print(output)
            counterOfUserLeft += 4
            counterOfUserRight += 4
            p.terminate()
            end = time.time()
            print(end - start)
            if counterOfUserLeft == 64:
                break
    else:
        testing()