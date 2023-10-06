import math
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
import keras
import random
import sys
import matplotlib.pyplot as plt
import logging
from multiprocessing import Process


logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def sample_around(first_food, second_food, all_food, pca_food, number_of_sample, index_first_food, index_second_food, pca_to_pass, no_zero, TrainCouple):

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

    counter = 0
    list_food_1_neighbor = []
    list_food_2_neighbor = []
    list_food_1_neighbor_ILASP = []
    list_food_2_neighbor_ILASP = []
    while counter < TrainCouple:
        food1_neighbor = np.zeros(len(first_food)+4, dtype="float32")
        changes = 0
        for index_element_food_1, element_food_1 in enumerate(first_food):
            change = np.random.binomial(1, 0.0025)
            if change == 1:
                changes +=1
                if index_element_food_1 == 0:
                        change_in_set = False
                        while True:
                            if change_in_set:
                                break
                            change_in = np.round(np.random.uniform(1, 5))
                            if change_in == 2.0:
                                continue
                            if change_in != element_food_1:
                                change_in_set = True
                        food1_neighbor[index_element_food_1+int(change_in)-1] = 1
                else:
                    change_up = np.random.binomial(1, 0.5)
                    if change_up != 0:
                        food1_neighbor[index_element_food_1+4] = element_food_1 + 1
                    else:
                        if element_food_1 != 0:
                            food1_neighbor[index_element_food_1+4] = element_food_1 - 1
                        else:
                            food1_neighbor[index_element_food_1 + 4] = element_food_1 + 1
            else:
                if index_element_food_1 == 0:
                    food1_neighbor[index_element_food_1 + int(first_food[0])-1] = 1
                else:
                    food1_neighbor[index_element_food_1 + 4] = element_food_1
        if changes == 0:
            continue

        food1_neighbor_ILASP_temp = np.zeros(24, dtype="float32")
        category_decoded_food1 = 0

        found = False
        for element_index, element_neighbor_1_original in enumerate(food1_neighbor):
            if element_index > 4 or found:
                break
            if element_neighbor_1_original != 0:
                category_decoded_food1 = element_index + 1
                found = True

        food1_neighbor_ILASP_temp[0] = category_decoded_food1
        food1_neighbor_ILASP_temp[1:4] = food1_neighbor[5:8]

        food_1_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')
        # 1, 16, 23, 26, 32
        for element_index, element_neighbor_1_original in enumerate(food1_neighbor):
            if element_index < 8 or element_index >= 39:
                continue
            j_index = element_index - 8
            if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
                continue
            if j_index == 5:
                food_1_neighbor_ILASP_macro_ingredients[0] = food_1_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
            if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
                food_1_neighbor_ILASP_macro_ingredients[1] = food_1_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
            if j_index == 11:
                food_1_neighbor_ILASP_macro_ingredients[2] = food_1_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
            if j_index == 2 or j_index == 12:
                food_1_neighbor_ILASP_macro_ingredients[3] = food_1_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
            if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
                food_1_neighbor_ILASP_macro_ingredients[4] = food_1_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
            if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
                food_1_neighbor_ILASP_macro_ingredients[5] = food_1_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
            if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
                food_1_neighbor_ILASP_macro_ingredients[6] = food_1_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
            if j_index == 20:
                food_1_neighbor_ILASP_macro_ingredients[7] = food_1_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
            if j_index == 10 or j_index == 13:
                food_1_neighbor_ILASP_macro_ingredients[8] = food_1_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
            if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
                food_1_neighbor_ILASP_macro_ingredients[9] = food_1_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
            if j_index == 31:
                food_1_neighbor_ILASP_macro_ingredients[10] = food_1_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
            if j_index == 14 or j_index == 17 or j_index == 33:
                food_1_neighbor_ILASP_macro_ingredients[11] = food_1_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original
        # 0 1 3 4 6
        food1_neighbor_ILASP_temp[4:16] = food_1_neighbor_ILASP_macro_ingredients
        food1_neighbor_ILASP_temp[16:24] = food1_neighbor[39:47]
        guard_identical_element = False
        excluded_indexes = [1, 2, 3, 16, 17, 19, 20, 22]  # in PC82STD cost, difficult, preparation and some preparation are removed, removed ingredient already done
        if len(list_food_1_neighbor_ILASP) != 0:
            for element_in_list in list_food_1_neighbor_ILASP:
                counter_identical_element = 0
                for index_factual_element, factual_element in enumerate(element_in_list):
                    if index_factual_element in excluded_indexes:
                        continue
                    if factual_element == food1_neighbor_ILASP_temp[index_factual_element]:
                        counter_identical_element += 1
                if counter_identical_element == len(food1_neighbor_ILASP_temp) - len(excluded_indexes):
                    guard_identical_element = True
                    break
        if guard_identical_element:
            continue
        list_food_1_neighbor_ILASP.append(food1_neighbor_ILASP_temp)
        list_food_1_neighbor.append(food1_neighbor)
        counter += 1
    counter = 0
    tentative_made = 0
    while counter < TrainCouple:
        food2_neighbor = np.zeros(len(second_food) + 4, dtype="float32")
        changes = 0
        for index_element_food_2, element_food_2 in enumerate(second_food):
            change = np.random.binomial(1, 0.0025)
            if change == 1:
                changes += 1
                if index_element_food_2 == 0:
                        change_in_set = False
                        while True:
                            if change_in_set:
                                break
                            change_in = np.random.uniform(1, 5)
                            if change_in != element_food_2:
                                change_in_set = True
                        food2_neighbor[index_element_food_2+int(change_in)-1] = 1
                else:
                    change_up = np.random.binomial(1, 0.5)
                    if change_up != 0:
                        food2_neighbor[index_element_food_2+4] = element_food_2 + 1
                    else:
                        if element_food_2 != 0:
                            food2_neighbor[index_element_food_2+4] = element_food_2 - 1
                        else:
                            food2_neighbor[index_element_food_2 + 4] = element_food_2 + 1
            else:
                if index_element_food_2 == 0:
                    food2_neighbor[index_element_food_2 + int(first_food[0]) -1] = 1
                else:
                    food2_neighbor[index_element_food_2 + 4] = element_food_2
        if changes == 0:
            continue
        food2_neighbor_ILASP_temp = np.zeros(24, dtype="float32")
        category_decoded_food2 = 0

        found = False
        for element_index, element_neighbor_1_original in enumerate(food2_neighbor):
            if element_index > 4 or found:
                break
            if element_neighbor_1_original != 0:
                category_decoded_food2 = element_index + 1
                found = True

        food2_neighbor_ILASP_temp[0] = category_decoded_food2
        food2_neighbor_ILASP_temp[1:4] = food2_neighbor[5:8]

        food_2_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

        for element_index, element_neighbor_1_original in enumerate(food2_neighbor):
            if element_index < 8 or element_index >= 39:
                continue
            j_index = element_index - 8
            if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
                continue
            if j_index == 5:
                food_2_neighbor_ILASP_macro_ingredients[0] = food_2_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
            if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
                food_2_neighbor_ILASP_macro_ingredients[1] = food_2_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
            if j_index == 11:
                food_2_neighbor_ILASP_macro_ingredients[2] = food_2_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
            if j_index == 2 or j_index == 12:
                food_2_neighbor_ILASP_macro_ingredients[3] = food_2_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
            if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
                food_2_neighbor_ILASP_macro_ingredients[4] = food_2_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
            if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
                food_2_neighbor_ILASP_macro_ingredients[5] = food_2_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
            if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
                food_2_neighbor_ILASP_macro_ingredients[6] = food_2_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
            if j_index == 20:
                food_2_neighbor_ILASP_macro_ingredients[7] = food_2_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
            if j_index == 10 or j_index == 13:
                food_2_neighbor_ILASP_macro_ingredients[8] = food_2_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
            if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
                food_2_neighbor_ILASP_macro_ingredients[9] = food_2_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
            if j_index == 31:
                food_2_neighbor_ILASP_macro_ingredients[10] = food_2_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
            if j_index == 14 or j_index == 17 or j_index == 33:
                food_2_neighbor_ILASP_macro_ingredients[11] = food_2_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

        food2_neighbor_ILASP_temp[4:16] = food_2_neighbor_ILASP_macro_ingredients
        food2_neighbor_ILASP_temp[16:24] = food2_neighbor[39:47]

        counter_identical_element = 0
        guard_identical_element = False
        excluded_indexes = [1, 2, 3, 16, 17, 19, 20, 22]  # in PC82STD cost, difficult, preparation and some preparation are removed, removed ingredient already done
        if len(list_food_2_neighbor_ILASP) != 0:
            for element_in_list in list_food_2_neighbor_ILASP:
                counter_identical_element = 0
                for index_factual_element, factual_element in enumerate(element_in_list):
                    if index_factual_element in excluded_indexes:
                        continue
                    if factual_element == food2_neighbor_ILASP_temp[index_factual_element]:
                        counter_identical_element += 1
                if counter_identical_element == len(food2_neighbor_ILASP_temp) - len(excluded_indexes):
                    guard_identical_element = True
                    break
        if guard_identical_element:
            continue
        list_food_2_neighbor_ILASP.append(food2_neighbor_ILASP_temp)
        list_food_2_neighbor.append(food2_neighbor)
        counter += 1
    food_1_neighbor_original = np.array(list_food_1_neighbor)
    food_2_neighbor_original = np.array(list_food_2_neighbor)
    food1_neighbor_ILASP = np.array(list_food_1_neighbor_ILASP)
    food2_neighbor_ILASP = np.array(list_food_2_neighbor_ILASP)

    # NOTE: the code work if we consider 8PC2STD case, if we are working with other case you have to adjust the part in which we save sampled dataset (e.g.: cost is not always 0 or something like this)

    first_food_ILASP = np.zeros(24, dtype="float32")
    first_food_ILASP[0] = first_food[0]
    first_food_ILASP[1:4] = first_food[1:4]

    food_1_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

    for element_index, element_neighbor_1_original in enumerate(first_food):
        if element_index < 4 or element_index >= 35:
            continue
        j_index = element_index - 4
        if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
            continue
        if j_index == 5:
            food_1_neighbor_ILASP_macro_ingredients[0] = food_1_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
        if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
            food_1_neighbor_ILASP_macro_ingredients[1] = food_1_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
        if j_index == 11:
            food_1_neighbor_ILASP_macro_ingredients[2] = food_1_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
        if j_index == 2 or j_index == 12:
            food_1_neighbor_ILASP_macro_ingredients[3] = food_1_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
        if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
            food_1_neighbor_ILASP_macro_ingredients[4] = food_1_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
        if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
            food_1_neighbor_ILASP_macro_ingredients[5] = food_1_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
        if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
            food_1_neighbor_ILASP_macro_ingredients[6] = food_1_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
        if j_index == 20:
            food_1_neighbor_ILASP_macro_ingredients[7] = food_1_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
        if j_index == 10 or j_index == 13:
            food_1_neighbor_ILASP_macro_ingredients[8] = food_1_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
        if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
            food_1_neighbor_ILASP_macro_ingredients[9] = food_1_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
        if j_index == 31:
            food_1_neighbor_ILASP_macro_ingredients[10] = food_1_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
        if j_index == 14 or j_index == 17 or j_index == 33:
            food_1_neighbor_ILASP_macro_ingredients[11] = food_1_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

    first_food_ILASP[4:16] = food_1_neighbor_ILASP_macro_ingredients
    first_food_ILASP[16:24] = first_food[35:43]

    second_food_ILASP = np.zeros(24, dtype="float32")
    second_food_ILASP[0] = first_food[0]
    second_food_ILASP[1:4] = second_food[1:4]

    food_2_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

    for element_index, element_neighbor_1_original in enumerate(second_food):
        if element_index < 4 or element_index >= 35:
            continue
        j_index = element_index - 4
        if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
            continue
        if j_index == 5:
            food_2_neighbor_ILASP_macro_ingredients[0] = food_2_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
        if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
            food_2_neighbor_ILASP_macro_ingredients[1] = food_2_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
        if j_index == 11:
            food_2_neighbor_ILASP_macro_ingredients[2] = food_2_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
        if j_index == 2 or j_index == 12:
            food_2_neighbor_ILASP_macro_ingredients[3] = food_2_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
        if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
            food_2_neighbor_ILASP_macro_ingredients[4] = food_2_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
        if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
            food_2_neighbor_ILASP_macro_ingredients[5] = food_2_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
        if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
            food_2_neighbor_ILASP_macro_ingredients[6] = food_2_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
        if j_index == 20:
            food_2_neighbor_ILASP_macro_ingredients[7] = food_2_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
        if j_index == 10 or j_index == 13:
            food_2_neighbor_ILASP_macro_ingredients[8] = food_2_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
        if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
            food_2_neighbor_ILASP_macro_ingredients[9] = food_2_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
        if j_index == 31:
            food_2_neighbor_ILASP_macro_ingredients[10] = food_2_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
        if j_index == 14 or j_index == 17 or j_index == 33:
            food_2_neighbor_ILASP_macro_ingredients[11] = food_2_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

    second_food_ILASP[4:16] = food_2_neighbor_ILASP_macro_ingredients
    second_food_ILASP[16:24] = second_food[35:43]

    distances_metric = np.zeros((TrainCouple, 3), dtype="float32")

    for neighbor_couple_index, (food_1_ILASP, food_2_ILASP) in enumerate(zip(food1_neighbor_ILASP, food2_neighbor_ILASP)):
        temp1 = 0
        temp2 = 0
        for index_element in range(0, 2):
            for f in range(0, len(first_food_ILASP)):
                if index_element == 0:
                    if f == 0:
                        temp1 += pow((0 if food_1_ILASP[0] == first_food_ILASP[f] else 3), 2)
                    else:
                        temp1 += pow((abs(food_1_ILASP[f] - first_food_ILASP[f])), 2)
                else:
                    if f == 0:
                        temp2 += pow((0 if food_2_ILASP[0] == second_food_ILASP[f] else 3), 2)
                    else:
                        temp2 += pow((abs(food_2_ILASP[f] - second_food_ILASP[f])), 2)
            if index_element == 0:
                temp1 = math.sqrt(temp1)
            else:
                temp2 = math.sqrt(temp2)
        distances_metric[neighbor_couple_index, 0] = neighbor_couple_index
        distances_metric[neighbor_couple_index, 1] = round(temp1 + temp2)
    max_distance = np.max(distances_metric[:, 1])
    for distance_index, distance in enumerate(distances_metric):
        distances_metric[distance_index, 2] = distance[1]/max_distance

    # I've considered to give priority by divide samples respect to how many std/means (of all samples r.t distance to original point) they are distant from original point, but as can be easily see with the chosen parameter, don't allow to
    # divide samples for more than 2 or 3 groups, while we want something more capillar, possibly
    # mean_of_distances = np.mean(distances_metric[:, 1])
    # std_of_distances = np.std(distances_metric[:, 1])
    #
    # distances_df = pd.DataFrame(distances_metric, columns=['index', 'distance', 'distance_p'])
    # distances_df.drop(['index', 'distance_p'], axis=1, inplace=True)
    # distances_df.hist(bins=len(distances_metric))
    # plt.axhline(y=mean_of_distances, color='r', linestyle='-')
    # plt.axhline(y=mean_of_distances+std_of_distances, color='g', linestyle='-')
    # plt.axhline(y=mean_of_distances+(2*std_of_distances), color='y', linestyle='-')
    # plt.axhline(y=2*mean_of_distances, color='b', linestyle='-')
    # plt.show()
    # plt.clf()

    # remember: penalty are inversely proportional to the distance, because penalty is the cost that ILASP pay to not covering those examples, and for us the nearest example are those most important.
    distances = np.unique(distances_metric[:, 1])
    number_of_distances = len(distances)
    distances_dict = {}
    for distance_index, distance in enumerate(distances):
        distances_dict[str(distance)] = number_of_distances
        number_of_distances -= 1

    max_v_list = [1, 2, 3, 4, 5]
    max_p_list = [1, 2, 3, 4, 5]
    if no_zero:
        Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/las_files"
    else:
        Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/las_files"
    f_output = os.path.join(Dir, 'recipes_sampled_'+str(index_first_food)+'-'+str(index_second_food)+'.las')
    f = open(f_output, 'w+')
    sys.stdout = open(f_output, 'w')

    for index_food1_index, food1_neighbor_to_print in enumerate(food1_neighbor_ILASP):
        if int(food1_neighbor_to_print[0]) == 2:
            item = "#pos(sampled" + str(index_first_food) + "-" + str(index_food1_index) + ", {}, {}, {category(0). value(cost,0). value(difficulty,0). value(prepTime,0)."
        else:
            item = "#pos(sampled" + str(index_first_food) + "-" + str(index_food1_index) + ", {}, {}, {category(" + str(int(food1_neighbor_to_print[0])) + "). value(cost,0). value(difficulty,0). value(prepTime,0)."
        for j_index in range(4, 16):
            if food1_neighbor_to_print[j_index] != 0:
                item = item + " value(" + macro_ingredients_dictionary[j_index-4] + "," + str(int(food1_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + macro_ingredients_dictionary[j_index-4] + ",0)."
        for j_index in range(16, 24):
            if food1_neighbor_to_print[j_index] != 0:
                if (j_index-16 <= 1) or (3 <= j_index-16 <= 4) or (j_index-16 == 6):
                    item = item + " value(" + preparation_dictionary[j_index-16] + ",0)."
                else:
                    item = item + " value(" + preparation_dictionary[j_index-16] + "," + str(int(food1_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + preparation_dictionary[j_index-16] + ",0)."
        item = item + "} )."
        print(item)

    for index_food2_index, food2_neighbor_to_print in enumerate(food2_neighbor_ILASP):
        if int(food2_neighbor_to_print[0]) == 2:
            item = "#pos(sampled" + str(index_second_food) + "-" + str(index_food2_index) + ", {}, {}, {category(0). value(cost,0). value(difficulty,0). value(prepTime,0)."
        else:
            item = "#pos(sampled" + str(index_second_food) + "-" + str(index_food2_index) + ", {}, {}, {category(" + str(int(food2_neighbor_to_print[0])) + "). value(cost,0). value(difficulty,0). value(prepTime,0)."
        for j_index in range(4, 16):
            if food2_neighbor_to_print[j_index] != 0:
                item = item + " value(" + macro_ingredients_dictionary[j_index-4] + "," + str(int(food2_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + macro_ingredients_dictionary[j_index-4] + ",0)."
        for j_index in range(16, 24):
            if food2_neighbor_to_print[j_index] != 0:
                if (j_index-16 <= 1) or (3 <= j_index-16 <= 4) or (j_index-16 == 6):
                    item = item + " value(" + preparation_dictionary[j_index-16] + ",0)."
                else:
                    item = item + " value(" + preparation_dictionary[j_index-16] + "," + str(int(food2_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + preparation_dictionary[j_index-16] + ",0)."
        item = item + "} )."
        print(item)

    sys.stdout = sys.__stdout__
    f.close()

    if no_zero:
        Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/distances"
    else:
        Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/distances"
    f_output = os.path.join(Dir, 'recipes_distances' + str(index_first_food) + '-' + str(index_second_food) + '.txt')
    f = open(f_output, 'w+')
    sys.stdout = open(f_output, 'w')

    for distance_index, distance in enumerate(distances_metric):
        for distance_key in distances_dict.keys():
            if str(distance[1]) == distance_key:
                print(str(distances_dict[distance_key]))

    sys.stdout = sys.__stdout__
    f.close()

    food_1_neighbor_scaled = food_1_neighbor_original.copy()
    food_2_neighbor_scaled = food_2_neighbor_original.copy()
    scaler_foods = preprocessing.StandardScaler()
    food_1_neighbor_scaled[:, 0:8] = scaler_foods.fit_transform(food_1_neighbor_scaled[:, 0:8])
    food_2_neighbor_scaled[:, 0:8] = scaler_foods.fit_transform(food_2_neighbor_scaled[:, 0:8])
    for index_row, row_food1 in enumerate(food_1_neighbor_scaled):
        sum_of_ingredients = 0
        for index_ingredient in range(8, 39):
            sum_of_ingredients += food_1_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(8, 39):
            food_1_neighbor_scaled[index_row, index_ingredient] /= sum_of_ingredients
    for index_row, row_food1 in enumerate(food_2_neighbor_scaled):
        sum_of_ingredients = 0
        for index_ingredient in range(8, 39):
            sum_of_ingredients += food_2_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(8, 39):
            food_2_neighbor_scaled[index_row, index_ingredient] /= sum_of_ingredients
    for index_row, row_food1 in enumerate(food_1_neighbor_scaled):
        sum_of_preparations = 0
        for index_ingredient in range(39, 47):
            sum_of_preparations += food_1_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(39, 47):
            food_1_neighbor_scaled[index_row, index_ingredient] /= sum_of_preparations
    for index_row, row_food1 in enumerate(food_2_neighbor_scaled):
        sum_of_preparations = 0
        for index_ingredient in range(39, 47):
            sum_of_preparations += food_2_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(39, 47):
            food_2_neighbor_scaled[index_row, index_ingredient] /= sum_of_preparations

    food_1_neighbor = pca_to_pass.transform(food_1_neighbor_scaled)
    food_2_neighbor = pca_to_pass.transform(food_2_neighbor_scaled)


    plt.scatter(food_1_neighbor[:, 0], food_1_neighbor[:, 1])
    plt.scatter(food_2_neighbor[:, 0], food_2_neighbor[:, 1])
    plt.scatter(pca_food[index_first_food, 0], pca_food[index_first_food, 1])
    plt.scatter(pca_food[index_second_food, 0], pca_food[index_second_food, 1])
    plt.title("couple: " + str(index_first_food) + "-" + str(index_second_food))
    if no_zero:
        plt.savefig('./Data8Component2Std/sampled-recipes-no-zero/Train' + str(TrainCouple) + '/plots/couple' + str(index_first_food) + '-' + str(index_second_food), dpi=300)
    else:
        plt.savefig('./Data8Component2Std/sampled-recipes-zero/Train' + str(TrainCouple) + '/plots/couple' + str(index_first_food) + '-' + str(index_second_food), dpi=300)
    plt.clf()

    return food_1_neighbor, food_2_neighbor

def sample_around_gauss_2(first_food, second_food, all_food, pca_food, number_of_sample, index_first_food, index_second_food, pca_to_pass, no_zero, TrainCouple):
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

    counter = 0
    list_food_1_neighbor = []
    list_food_2_neighbor = []
    list_food_1_neighbor_ILASP = []
    list_food_2_neighbor_ILASP = []
    while counter < TrainCouple:
        food1_neighbor = np.zeros(len(first_food) + 4, dtype="float32")

        # note: n = 42 toss, p = 0.0025 for single toss. So:
        # P(0) = (42! / (0! * (42-0)!) * 0.0025^0 * 0.9975^42 = 1 * 1 * 0.9975^42 ~ 0.90 = 90%
        # P(1) = (42! / (1! * (42-1)!) * 0.0025^1 * 0.9975^41 = 42 * 0.0025 * 0.9975^41 ~ 0.09 = 9%
        # P(2) = (42! / (2! * (42-2)!) * 0.0025^2 * 0.9975^40 = 861 * 0.00000625 * 0.9975^40 ~ 0.004 = 0.4%
        # ...
        # VAR(X) = n*p*(1-p) = 0.1047375
        # std(X) = sqrt(VAR(X)) ~ 0.32
        # ------------------------------------------------------
        # identical items not admitted. this case is the best trade-off of low std and low execution time to find the sampled couple (p < 0.01 make code execute for very long time even for only 1 of the 100 couple)
        changes = 0
        for index_element_food_1, element_food_1 in enumerate(first_food):
            change = np.random.binomial(1, 0.0025)
            if change == 1:
                changes += 1
                if index_element_food_1 == 0:
                    change_in_set = False
                    while True:
                        if change_in_set:
                            break
                        change_in = np.round(np.random.uniform(1, 5))
                        if change_in == 2.0:
                            continue
                        if change_in != element_food_1:
                            change_in_set = True
                    food1_neighbor[index_element_food_1 + int(change_in) - 1] = 1
                else:
                    change_up = np.random.binomial(1, 0.5)
                    if change_up != 0:
                        food1_neighbor[index_element_food_1 + 4] = element_food_1 + 1
                    else:
                        if element_food_1 != 0:
                            food1_neighbor[index_element_food_1 + 4] = element_food_1 - 1
                        else:
                            food1_neighbor[index_element_food_1 + 4] = element_food_1 + 1
            else:
                if index_element_food_1 == 0:
                    food1_neighbor[index_element_food_1 + int(first_food[0]) - 1] = 1
                else:
                    food1_neighbor[index_element_food_1 + 4] = element_food_1
        if changes == 0:
            continue

        food1_neighbor_ILASP_temp = np.zeros(24, dtype="float32")
        category_decoded_food1 = 0

        found = False
        for element_index, element_neighbor_1_original in enumerate(food1_neighbor):
            if element_index > 4 or found:
                break
            if element_neighbor_1_original != 0:
                category_decoded_food1 = element_index + 1
                found = True

        food1_neighbor_ILASP_temp[0] = category_decoded_food1
        food1_neighbor_ILASP_temp[1:4] = food1_neighbor[5:8]

        food_1_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

        for element_index, element_neighbor_1_original in enumerate(food1_neighbor):
            if element_index < 8 or element_index >= 39:
                continue
            j_index = element_index - 8
            if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
                continue
            if j_index == 5:
                food_1_neighbor_ILASP_macro_ingredients[0] = food_1_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
            if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
                food_1_neighbor_ILASP_macro_ingredients[1] = food_1_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
            if j_index == 11:
                food_1_neighbor_ILASP_macro_ingredients[2] = food_1_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
            if j_index == 2 or j_index == 12:
                food_1_neighbor_ILASP_macro_ingredients[3] = food_1_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
            if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
                food_1_neighbor_ILASP_macro_ingredients[4] = food_1_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
            if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
                food_1_neighbor_ILASP_macro_ingredients[5] = food_1_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
            if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
                food_1_neighbor_ILASP_macro_ingredients[6] = food_1_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
            if j_index == 20:
                food_1_neighbor_ILASP_macro_ingredients[7] = food_1_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
            if j_index == 10 or j_index == 13:
                food_1_neighbor_ILASP_macro_ingredients[8] = food_1_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
            if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
                food_1_neighbor_ILASP_macro_ingredients[9] = food_1_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
            if j_index == 31:
                food_1_neighbor_ILASP_macro_ingredients[10] = food_1_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
            if j_index == 14 or j_index == 17 or j_index == 33:
                food_1_neighbor_ILASP_macro_ingredients[11] = food_1_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original
        # 0 1 3 4 6
        food1_neighbor_ILASP_temp[4:16] = food_1_neighbor_ILASP_macro_ingredients
        food1_neighbor_ILASP_temp[16:24] = food1_neighbor[39:47]
        guard_identical_element = False
        excluded_indexes = [1, 2, 3, 16, 17, 19, 20, 22]  # in PC82STD cost, difficult, preparation and some preparation are removed, removed ingredient already done
        if len(list_food_1_neighbor_ILASP) != 0:
            for element_in_list in list_food_1_neighbor_ILASP:
                counter_identical_element = 0
                for index_factual_element, factual_element in enumerate(element_in_list):
                    if index_factual_element in excluded_indexes:
                        continue
                    if factual_element == food1_neighbor_ILASP_temp[index_factual_element]:
                        counter_identical_element += 1
                if counter_identical_element == len(food1_neighbor_ILASP_temp) - len(excluded_indexes):
                    guard_identical_element = True
                    break
        if guard_identical_element:
            continue
        list_food_1_neighbor_ILASP.append(food1_neighbor_ILASP_temp)
        list_food_1_neighbor.append(food1_neighbor)
        counter += 1
    counter = 0
    tentative_made = 0
    while counter < TrainCouple:
        food2_neighbor = np.zeros(len(second_food) + 4, dtype="float32")
        changes = 0
        for index_element_food_2, element_food_2 in enumerate(second_food):
            change = np.random.binomial(1, 0.0025)
            if change == 1:
                changes += 1
                if index_element_food_2 == 0:
                    change_in_set = False
                    while True:
                        if change_in_set:
                            break
                        change_in = np.random.uniform(1, 5)
                        if change_in != element_food_2:
                            change_in_set = True
                    food2_neighbor[index_element_food_2 + int(change_in) - 1] = 1
                else:
                    change_up = np.random.binomial(1, 0.5)
                    if change_up != 0:
                        food2_neighbor[index_element_food_2 + 4] = element_food_2 + 1
                    else:
                        if element_food_2 != 0:
                            food2_neighbor[index_element_food_2 + 4] = element_food_2 - 1
                        else:
                            food2_neighbor[index_element_food_2 + 4] = element_food_2 + 1
            else:
                if index_element_food_2 == 0:
                    food2_neighbor[index_element_food_2 + int(first_food[0]) - 1] = 1
                else:
                    food2_neighbor[index_element_food_2 + 4] = element_food_2
        if changes == 0:
            continue
        food2_neighbor_ILASP_temp = np.zeros(24, dtype="float32")
        category_decoded_food2 = 0

        found = False
        for element_index, element_neighbor_1_original in enumerate(food2_neighbor):
            if element_index > 4 or found:
                break
            if element_neighbor_1_original != 0:
                category_decoded_food2 = element_index + 1
                found = True

        food2_neighbor_ILASP_temp[0] = category_decoded_food2
        food2_neighbor_ILASP_temp[1:4] = food2_neighbor[5:8]

        food_2_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

        for element_index, element_neighbor_1_original in enumerate(food2_neighbor):
            if element_index < 8 or element_index >= 39:
                continue
            j_index = element_index - 8
            if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
                continue
            if j_index == 5:
                food_2_neighbor_ILASP_macro_ingredients[0] = food_2_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
            if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
                food_2_neighbor_ILASP_macro_ingredients[1] = food_2_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
            if j_index == 11:
                food_2_neighbor_ILASP_macro_ingredients[2] = food_2_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
            if j_index == 2 or j_index == 12:
                food_2_neighbor_ILASP_macro_ingredients[3] = food_2_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
            if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
                food_2_neighbor_ILASP_macro_ingredients[4] = food_2_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
            if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
                food_2_neighbor_ILASP_macro_ingredients[5] = food_2_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
            if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
                food_2_neighbor_ILASP_macro_ingredients[6] = food_2_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
            if j_index == 20:
                food_2_neighbor_ILASP_macro_ingredients[7] = food_2_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
            if j_index == 10 or j_index == 13:
                food_2_neighbor_ILASP_macro_ingredients[8] = food_2_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
            if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
                food_2_neighbor_ILASP_macro_ingredients[9] = food_2_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
            if j_index == 31:
                food_2_neighbor_ILASP_macro_ingredients[10] = food_2_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
            if j_index == 14 or j_index == 17 or j_index == 33:
                food_2_neighbor_ILASP_macro_ingredients[11] = food_2_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

        food2_neighbor_ILASP_temp[4:16] = food_2_neighbor_ILASP_macro_ingredients
        food2_neighbor_ILASP_temp[16:24] = food2_neighbor[39:47]

        counter_identical_element = 0
        guard_identical_element = False
        excluded_indexes = [1, 2, 3, 16, 17, 19, 20, 22]  # in PC82STD cost, difficult, preparation and some preparation are removed, removed ingredient already done
        if len(list_food_2_neighbor_ILASP) != 0:
            for element_in_list in list_food_2_neighbor_ILASP:
                counter_identical_element = 0
                for index_factual_element, factual_element in enumerate(element_in_list):
                    if index_factual_element in excluded_indexes:
                        continue
                    if factual_element == food2_neighbor_ILASP_temp[index_factual_element]:
                        counter_identical_element += 1
                if counter_identical_element == len(food2_neighbor_ILASP_temp) - len(excluded_indexes):
                    guard_identical_element = True
                    break
        if guard_identical_element:
            continue
        list_food_2_neighbor_ILASP.append(food2_neighbor_ILASP_temp)
        list_food_2_neighbor.append(food2_neighbor)
        counter += 1
    food_1_neighbor_original = np.array(list_food_1_neighbor)
    food_2_neighbor_original = np.array(list_food_2_neighbor)
    food1_neighbor_ILASP = np.array(list_food_1_neighbor_ILASP)
    food2_neighbor_ILASP = np.array(list_food_2_neighbor_ILASP)

    # NOTE: the code work if we consider 8PC2STD case, if we are working with other case you have to adjust the part in which we save sampled dataset (e.g.: cost is not always 0 or something like this)

    first_food_ILASP = np.zeros(24, dtype="float32")
    first_food_ILASP[0] = first_food[0]
    first_food_ILASP[1:4] = first_food[1:4]

    food_1_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

    for element_index, element_neighbor_1_original in enumerate(first_food):
        if element_index < 4 or element_index >= 35:
            continue
        j_index = element_index - 4
        if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
            continue
        if j_index == 5:
            food_1_neighbor_ILASP_macro_ingredients[0] = food_1_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
        if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
            food_1_neighbor_ILASP_macro_ingredients[1] = food_1_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
        if j_index == 11:
            food_1_neighbor_ILASP_macro_ingredients[2] = food_1_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
        if j_index == 2 or j_index == 12:
            food_1_neighbor_ILASP_macro_ingredients[3] = food_1_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
        if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
            food_1_neighbor_ILASP_macro_ingredients[4] = food_1_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
        if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
            food_1_neighbor_ILASP_macro_ingredients[5] = food_1_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
        if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
            food_1_neighbor_ILASP_macro_ingredients[6] = food_1_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
        if j_index == 20:
            food_1_neighbor_ILASP_macro_ingredients[7] = food_1_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
        if j_index == 10 or j_index == 13:
            food_1_neighbor_ILASP_macro_ingredients[8] = food_1_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
        if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
            food_1_neighbor_ILASP_macro_ingredients[9] = food_1_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
        if j_index == 31:
            food_1_neighbor_ILASP_macro_ingredients[10] = food_1_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
        if j_index == 14 or j_index == 17 or j_index == 33:
            food_1_neighbor_ILASP_macro_ingredients[11] = food_1_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

    first_food_ILASP[4:16] = food_1_neighbor_ILASP_macro_ingredients
    first_food_ILASP[16:24] = first_food[35:43]

    second_food_ILASP = np.zeros(24, dtype="float32")
    second_food_ILASP[0] = first_food[0]
    second_food_ILASP[1:4] = second_food[1:4]

    food_2_neighbor_ILASP_macro_ingredients = np.zeros(12, dtype='float32')

    for element_index, element_neighbor_1_original in enumerate(second_food):
        if element_index < 4 or element_index >= 35:
            continue
        j_index = element_index - 4
        if (j_index == 1) or (4 <= j_index <= 8) or (j_index == 10) or (j_index == 14) or (16 <= j_index <= 19) or (21 <= j_index <= 24) or (28 <= j_index <= 29):
            continue
        if j_index == 5:
            food_2_neighbor_ILASP_macro_ingredients[0] = food_2_neighbor_ILASP_macro_ingredients[0] + element_neighbor_1_original
        if j_index == 4 or j_index == 6 or j_index == 18 or j_index == 28 or j_index == 35:
            food_2_neighbor_ILASP_macro_ingredients[1] = food_2_neighbor_ILASP_macro_ingredients[1] + element_neighbor_1_original
        if j_index == 11:
            food_2_neighbor_ILASP_macro_ingredients[2] = food_2_neighbor_ILASP_macro_ingredients[2] + element_neighbor_1_original
        if j_index == 2 or j_index == 12:
            food_2_neighbor_ILASP_macro_ingredients[3] = food_2_neighbor_ILASP_macro_ingredients[3] + element_neighbor_1_original
        if j_index == 1 or j_index == 8 or j_index == 21 or j_index == 24 or j_index == 30:
            food_2_neighbor_ILASP_macro_ingredients[4] = food_2_neighbor_ILASP_macro_ingredients[4] + element_neighbor_1_original
        if j_index == 3 or j_index == 9 or j_index == 15 or j_index == 29 or j_index == 34:
            food_2_neighbor_ILASP_macro_ingredients[5] = food_2_neighbor_ILASP_macro_ingredients[5] + element_neighbor_1_original
        if j_index == 0 or j_index == 7 or j_index == 22 or j_index == 23 or j_index == 26 or j_index == 32:
            food_2_neighbor_ILASP_macro_ingredients[6] = food_2_neighbor_ILASP_macro_ingredients[6] + element_neighbor_1_original
        if j_index == 20:
            food_2_neighbor_ILASP_macro_ingredients[7] = food_2_neighbor_ILASP_macro_ingredients[7] + element_neighbor_1_original
        if j_index == 10 or j_index == 13:
            food_2_neighbor_ILASP_macro_ingredients[8] = food_2_neighbor_ILASP_macro_ingredients[8] + element_neighbor_1_original
        if j_index == 16 or j_index == 19 or j_index == 25 or j_index == 27:
            food_2_neighbor_ILASP_macro_ingredients[9] = food_2_neighbor_ILASP_macro_ingredients[9] + element_neighbor_1_original
        if j_index == 31:
            food_2_neighbor_ILASP_macro_ingredients[10] = food_2_neighbor_ILASP_macro_ingredients[10] + element_neighbor_1_original
        if j_index == 14 or j_index == 17 or j_index == 33:
            food_2_neighbor_ILASP_macro_ingredients[11] = food_2_neighbor_ILASP_macro_ingredients[11] + element_neighbor_1_original

    second_food_ILASP[4:16] = food_2_neighbor_ILASP_macro_ingredients
    second_food_ILASP[16:24] = second_food[35:43]

    distances_metric = np.zeros((TrainCouple, 3), dtype="float32")

    for neighbor_couple_index, (food_1_ILASP, food_2_ILASP) in enumerate(zip(food1_neighbor_ILASP, food2_neighbor_ILASP)):
        temp1 = 0
        temp2 = 0
        for index_element in range(0, 2):
            for f in range(0, len(first_food_ILASP)):
                if index_element == 0:
                    if f == 0:
                        temp1 += pow((0 if food_1_ILASP[0] == first_food_ILASP[f] else 3), 2)
                    else:
                        temp1 += pow((abs(food_1_ILASP[f] - first_food_ILASP[f])), 2)
                else:
                    if f == 0:
                        temp2 += pow((0 if food_2_ILASP[0] == second_food_ILASP[f] else 3), 2)
                    else:
                        temp2 += pow((abs(food_2_ILASP[f] - second_food_ILASP[f])), 2)
            if index_element == 0:
                temp1 = math.sqrt(temp1)
            else:
                temp2 = math.sqrt(temp2)
        distances_metric[neighbor_couple_index, 0] = neighbor_couple_index
        distances_metric[neighbor_couple_index, 1] = round(temp1 + temp2)
    max_distance = np.max(distances_metric[:, 1])
    for distance_index, distance in enumerate(distances_metric):
        distances_metric[distance_index, 2] = distance[1] / max_distance

    # I've considered to give priority by divide samples respect to how many std/means (of all samples r.t distance to original point) they are distant from original point, but as can be easily see with the chosen parameter, don't allow to
    # divide samples for more than 2 or 3 groups, while we want something more capillar, possibly
    # mean_of_distances = np.mean(distances_metric[:, 1])
    # std_of_distances = np.std(distances_metric[:, 1])
    #
    # distances_df = pd.DataFrame(distances_metric, columns=['index', 'distance', 'distance_p'])
    # distances_df.drop(['index', 'distance_p'], axis=1, inplace=True)
    # distances_df.hist(bins=len(distances_metric))
    # plt.axhline(y=mean_of_distances, color='r', linestyle='-')
    # plt.axhline(y=mean_of_distances+std_of_distances, color='g', linestyle='-')
    # plt.axhline(y=mean_of_distances+(2*std_of_distances), color='y', linestyle='-')
    # plt.axhline(y=2*mean_of_distances, color='b', linestyle='-')
    # plt.show()
    # plt.clf()

    # remember: penalty are inversely proportional to the distance, because penalty is the cost that ILASP pay to not covering those examples, and for us the nearest example are those most important.
    distances = np.unique(distances_metric[:, 1])
    number_of_distances = len(distances)
    distances_dict = {}
    for distance_index, distance in enumerate(distances):
        distances_dict[str(distance)] = number_of_distances
        number_of_distances -= 1

    max_v_list = [1, 2, 3, 4, 5]
    max_p_list = [1, 2, 3, 4, 5]
    if no_zero:
        Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/las_files"
    else:
        Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/las_files"
    f_output = os.path.join(Dir, 'recipes_sampled_' + str(index_first_food) + '-' + str(index_second_food) + '.las')
    f = open(f_output, 'w+')
    sys.stdout = open(f_output, 'w')

    for index_food1_index, food1_neighbor_to_print in enumerate(food1_neighbor_ILASP):
        if int(food1_neighbor_to_print[0]) == 2:
            item = "#pos(sampled" + str(index_first_food) + "-" + str(index_food1_index) + ", {}, {}, {category(0). value(cost,0). value(difficulty,0). value(prepTime,0)."
        else:
            item = "#pos(sampled" + str(index_first_food) + "-" + str(index_food1_index) + ", {}, {}, {category(" + str(int(food1_neighbor_to_print[0])) + "). value(cost,0). value(difficulty,0). value(prepTime,0)."
        for j_index in range(4, 16):
            if food1_neighbor_to_print[j_index] != 0:
                item = item + " value(" + macro_ingredients_dictionary[j_index - 4] + "," + str(int(food1_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + macro_ingredients_dictionary[j_index - 4] + ",0)."
        for j_index in range(16, 24):
            if food1_neighbor_to_print[j_index] != 0:
                if (j_index - 16 <= 1) or (3 <= j_index - 16 <= 4) or (j_index - 16 == 6):
                    item = item + " value(" + preparation_dictionary[j_index - 16] + ",0)."
                else:
                    item = item + " value(" + preparation_dictionary[j_index - 16] + "," + str(int(food1_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + preparation_dictionary[j_index - 16] + ",0)."
        item = item + "} )."
        print(item)

    for index_food2_index, food2_neighbor_to_print in enumerate(food2_neighbor_ILASP):
        if int(food2_neighbor_to_print[0]) == 2:
            item = "#pos(sampled" + str(index_second_food) + "-" + str(index_food2_index) + ", {}, {}, {category(0). value(cost,0). value(difficulty,0). value(prepTime,0)."
        else:
            item = "#pos(sampled" + str(index_second_food) + "-" + str(index_food2_index) + ", {}, {}, {category(" + str(int(food2_neighbor_to_print[0])) + "). value(cost,0). value(difficulty,0). value(prepTime,0)."
        for j_index in range(4, 16):
            if food2_neighbor_to_print[j_index] != 0:
                item = item + " value(" + macro_ingredients_dictionary[j_index - 4] + "," + str(int(food2_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + macro_ingredients_dictionary[j_index - 4] + ",0)."
        for j_index in range(16, 24):
            if food2_neighbor_to_print[j_index] != 0:
                if (j_index - 16 <= 1) or (3 <= j_index - 16 <= 4) or (j_index - 16 == 6):
                    item = item + " value(" + preparation_dictionary[j_index - 16] + ",0)."
                else:
                    item = item + " value(" + preparation_dictionary[j_index - 16] + "," + str(int(food2_neighbor_to_print[j_index])) + ")."
            else:
                item = item + " value(" + preparation_dictionary[j_index - 16] + ",0)."
        item = item + "} )."
        print(item)

    sys.stdout = sys.__stdout__
    f.close()

    if no_zero:
        Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/distances"
    else:
        Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/distances"
    f_output = os.path.join(Dir, 'recipes_distances' + str(index_first_food) + '-' + str(index_second_food) + '.txt')
    f = open(f_output, 'w+')
    sys.stdout = open(f_output, 'w')

    for distance_index, distance in enumerate(distances_metric):
        for distance_key in distances_dict.keys():
            if str(distance[1]) == distance_key:
                print(str(distances_dict[distance_key]))

    sys.stdout = sys.__stdout__
    f.close()

    food_1_neighbor_scaled = food_1_neighbor_original.copy()
    food_2_neighbor_scaled = food_2_neighbor_original.copy()
    scaler_foods = preprocessing.StandardScaler()
    food_1_neighbor_scaled[:, 0:8] = scaler_foods.fit_transform(food_1_neighbor_scaled[:, 0:8])
    food_2_neighbor_scaled[:, 0:8] = scaler_foods.fit_transform(food_2_neighbor_scaled[:, 0:8])
    for index_row, row_food1 in enumerate(food_1_neighbor_scaled):
        sum_of_ingredients = 0
        for index_ingredient in range(8, 39):
            sum_of_ingredients += food_1_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(8, 39):
            food_1_neighbor_scaled[index_row, index_ingredient] /= sum_of_ingredients
    for index_row, row_food1 in enumerate(food_2_neighbor_scaled):
        sum_of_ingredients = 0
        for index_ingredient in range(8, 39):
            sum_of_ingredients += food_2_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(8, 39):
            food_2_neighbor_scaled[index_row, index_ingredient] /= sum_of_ingredients
    for index_row, row_food1 in enumerate(food_1_neighbor_scaled):
        sum_of_preparations = 0
        for index_ingredient in range(39, 47):
            sum_of_preparations += food_1_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(39, 47):
            food_1_neighbor_scaled[index_row, index_ingredient] /= sum_of_preparations
    for index_row, row_food1 in enumerate(food_2_neighbor_scaled):
        sum_of_preparations = 0
        for index_ingredient in range(39, 47):
            sum_of_preparations += food_2_neighbor_scaled[index_row, index_ingredient]
        for index_ingredient in range(39, 47):
            food_2_neighbor_scaled[index_row, index_ingredient] /= sum_of_preparations

    food_1_neighbor = pca_to_pass.transform(food_1_neighbor_scaled)
    food_2_neighbor = pca_to_pass.transform(food_2_neighbor_scaled)

    plt.scatter(food_1_neighbor[:, 0], food_1_neighbor[:, 1])
    plt.scatter(food_2_neighbor[:, 0], food_2_neighbor[:, 1])
    plt.scatter(pca_food[index_first_food, 0], pca_food[index_first_food, 1])
    plt.scatter(pca_food[index_second_food, 0], pca_food[index_second_food, 1])
    plt.title("couple: " + str(index_first_food) + "-" + str(index_second_food))
    if no_zero:
        plt.savefig('./Data8Component2Std/sampled-recipes-no-zero/Train' + str(TrainCouple) + '/plots/couple' + str(index_first_food) + '-' + str(index_second_food), dpi=300)
    else:
        plt.savefig('./Data8Component2Std/sampled-recipes-zero/Train' + str(TrainCouple) + '/plots/couple' + str(index_first_food) + '-' + str(index_second_food), dpi=300)
    plt.clf()

    return food_1_neighbor, food_2_neighbor


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
    del model
    del to_predict
    del to_predict_reshaped
    del predictions
    return to_print

# def create_samples(no_zero, TrainCouple):

no_zero = True
TrainCouple = 45

# to recover unexpected error of program
if no_zero:
    Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/distances/"
else:
    Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/distances/"
already_done_one = []
already_done_two = []
for filename in os.listdir(Dir):
    f = filename
    already_done_one.append(int(f[17:f.find("-")]))
    already_done_two.append(int(f[f.find("-") + 1:f.find(".")]))
if len(already_done_one) != 0:
    already_done_np = np.zeros((len(already_done_one), 2), dtype="int32")
    for already_done_len in range(len(already_done_one)):
        already_done_np[already_done_len] = [already_done_one[already_done_len], already_done_two[already_done_len]]
else:
    already_done_np = np.zeros(999, dtype="int32")
if len(already_done_np) == 999:
    already_done = 0
else:
    already_done = len(already_done_np)
# ----

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

linesOfPTrain = dataCC.split('\n')
couples_dataset = np.zeros((len(linesOfPTrain) - 1, 210, 2), dtype='int32')
for i, line in enumerate(linesOfPTrain):
    couples = [x for x in line.split(';')[0:]]
    for j, couple in enumerate(couples):
        if couple == '':
            continue
        elements = couple.split(',')
        couples_dataset[i, j, 0] = int(elements[0])
        couples_dataset[i, j, 1] = int(elements[1])

linesOfLTrain = dataL.split('\n')
couple_label = np.zeros((len(linesOfLTrain) - 1, 210), dtype='int32')
for i, line in enumerate(linesOfLTrain):
    if line == '':
        continue
    values = [x for x in line.split(' ')[:]]
    for j, value in enumerate(values):
        veryValue = int(value)
        couple_label[i, j] = veryValue


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
list_of_couple = []
# if already_done != 0:
#     for element in already_done_np:
#         list_of_couple.append([element[0], element[1]])
#     counter_of_couple =  len(already_done_np)
# else:
#     counter_of_couple = 0
for user in range(0, 54):
    while True:
        # if counter_of_couple == 100:
        #     break
        first_food_id_to_pass = np.random.randint(0, 100, size=1)
        second_food_id_to_pass = np.random.randint(0, 100, size=1)
        if first_food_id_to_pass[0] == second_food_id_to_pass[0]:
            continue
        if [first_food_id_to_pass[0], second_food_id_to_pass[0]] in list_of_couple:
            continue
        if [second_food_id_to_pass[0], first_food_id_to_pass[0]] in list_of_couple:
            continue
        to_not_append = False
        for i in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
            test_prediction = single_prediction(user_id=i, first_food=pca_data[first_food_id_to_pass[0]], second_food=pca_data[first_food_id_to_pass[0]])
            if no_zero and (test_prediction == "0"):
                to_not_append = True
                break
        if to_not_append:
            continue
        list_of_couple.append([first_food_id_to_pass[0], second_food_id_to_pass[0]])
        # print("--------------------couple queried--------------------")
        # print(str(int(first_food_id_to_pass)) + "," + str(int(second_food_id_to_pass)))
        # print("--------------------prediction of i-th user's model on queried couple--------------------")
        # for i in range(0, 48):
        #     single_prediction(user_id=i, first_food=pca_data[int(first_food_id_to_pass)], second_food=pca_data[int(second_food_id_to_pass)])
        # print("--------------------generation samples for train--------------------")
        X_PCA_1, X_PCA_2 = sample_around(first_food=all_data_original[int(first_food_id_to_pass)], second_food=all_data_original[int(second_food_id_to_pass)], all_food=all_data_original, pca_food=pca_data, number_of_sample=105, index_first_food=int(first_food_id_to_pass), index_second_food=int(second_food_id_to_pass), pca_to_pass=pca, no_zero=no_zero, TrainCouple=TrainCouple)
        # print("--------------------generation label for train--------------------")


        for i in [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]:
            if no_zero:
                Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/user_prediction/train"
            else:
                Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/user_prediction/train"
            f_output = os.path.join(Dir, 'user' + str(i) + '_Couple' + str(first_food_id_to_pass[0]) + "-" + str(second_food_id_to_pass[0]) + '.txt')
            f = open(f_output, 'w+')
            sys.stdout = open(f_output, 'w')
            for j, (food1, food2) in enumerate(zip(X_PCA_1, X_PCA_2)):
                train_prediction = single_prediction(user_id=i, first_food=food1, second_food=food2)
                print(train_prediction)
            sys.stdout = sys.__stdout__
            f.close()
            if no_zero:
                Dir = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/user_prediction/test"
            else:
                Dir = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/user_prediction/test"
            f_output = os.path.join(Dir, 'user' + str(i) + '_Couple' + str(first_food_id_to_pass[0]) + "-" + str(second_food_id_to_pass[0]) + '.txt')
            f = open(f_output, 'w+')
            sys.stdout = open(f_output, 'w')
            test_prediction = single_prediction(user_id=i, first_food=pca_data[first_food_id_to_pass[0]], second_food=pca_data[first_food_id_to_pass[0]])
            print(test_prediction)
            sys.stdout = sys.__stdout__
            f.close()
        # counter_of_couple += 1
    if no_zero:
        f_output = "Data8Component2Std/sampled-recipes-no-zero/Train" + str(TrainCouple) + "/couple.txt"
    else:
        f_output = "Data8Component2Std/sampled-recipes-zero/Train" + str(TrainCouple) + "/couple.txt"
    f = open(f_output, 'w+')
    sys.stdout = open(f_output, 'w')
    for couple in list_of_couple:
        print(str(couple[0]) + ";" + str(couple[1]))
    sys.stdout = sys.__stdout__
    f.close()

# if __name__ == '__main__':
#     p_zero = Process(target=create_samples, args=(False, 210))
#     p_no_zero = Process(target=create_samples, args=(True, 210))
#
#     p_zero.start()
#     p_no_zero.start()
#
#     p_zero.join()
#     p_no_zero.join()