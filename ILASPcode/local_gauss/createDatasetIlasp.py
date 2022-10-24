import os
import numpy as np
import sys

data = np.loadtxt('./Data8Component2Std/NNoutput/train/210CouplesGaussianNoise/data.csv', delimiter=';')


food_data_scalars = data[:, 3]
food_data_categories = data[:, 0:3]
food_data_ingredients = data[:, 4:35]
food_data_preparation = data[:, 35:]
# convert from classes to macro-classes
food_data_macro_ingredients = np.zeros((len(data), 12), dtype='float32')
for i, row in enumerate(food_data_ingredients): # different from global or local cases because here missing ingredients are already deleted
    for j, ingredient in enumerate(row):
        if j == 4:
            food_data_macro_ingredients[i, 0] = food_data_macro_ingredients[i, 0] + ingredient
        if j == 3 or j == 5 or j == 16 or j == 24 or j == 30:
            food_data_macro_ingredients[i, 1] = food_data_macro_ingredients[i, 1] + ingredient
        if j == 10:
            food_data_macro_ingredients[i, 2] = food_data_macro_ingredients[i, 2] + ingredient
        if j == 1 or j == 11:
            food_data_macro_ingredients[i, 3] = food_data_macro_ingredients[i, 3] + ingredient
        if j == 7 or j == 19 or j == 21 or j == 26:
            food_data_macro_ingredients[i, 4] = food_data_macro_ingredients[i, 4] + ingredient
        if j == 2 or j == 8 or j == 14 or j == 25 or j == 29:
            food_data_macro_ingredients[i, 5] = food_data_macro_ingredients[i, 5] + ingredient
        if j == 0 or j == 6 or j == 20:
            food_data_macro_ingredients[i, 6] = food_data_macro_ingredients[i, 6] + ingredient
        if j == 18:
            food_data_macro_ingredients[i, 7] = food_data_macro_ingredients[i, 7] + ingredient
        if j == 9 or j == 12:
            food_data_macro_ingredients[i, 8] = food_data_macro_ingredients[i, 8] + ingredient
        if j == 17 or j == 22 or j == 23:
            food_data_macro_ingredients[i, 9] = food_data_macro_ingredients[i, 9] + ingredient
        if j == 27:
            food_data_macro_ingredients[i, 10] = food_data_macro_ingredients[i, 10] + ingredient
        if j == 13 or j == 15 or j == 28:
            food_data_macro_ingredients[i, 11] = food_data_macro_ingredients[i, 11] + ingredient

# dictionary for macro_ingredients and preparation

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
                          7: "cottura_a_vapore",
                          8: "stufato"}


max_v_list = [1, 2, 3, 4, 5]
max_p_list = [1, 2, 3, 4, 5]
Dir = "Data8Component2Std/recipes/"

for max_v in max_v_list:
    for max_p in max_p_list:
        if int(max_v) > 0 and int(max_p) > 0:
            f_output = os.path.join(Dir, 'recipes_max_v(' + str(max_v) + ')-max_p(' + str(max_p) + ').las')
        elif int(max_v) > 0 or int(max_p) > 0:
            if int(max_v) > 0:
                f_output = os.path.join(Dir, 'recipes_max_v(' + str(max_v) + ')-max_p(default).las')
            else:
                f_output = os.path.join(Dir, 'recipes_max_v(default)-max_p(' + str(max_p) + ').las')
        else:
            f_output = os.path.join(Dir, 'recipes_max_v(default)-max_p(default).las')
        f = open(f_output, 'w+')
        sys.stdout = open(f_output, 'w')

        for i in range(0, 28):
            if int(food_data_categories[i, 0]) == 2:
                # item = "#pos(item" + str(i) + ", {}, {}, value(difficulty," + str(int(food_data_categories[i, 2])) + ")."
                item = "#pos(item" + str(i) + ", {}, {}, {"
            else:
                item = "#pos(item" + str(i) + ", {}, {}, {category(" + str(int(food_data_categories[i, 0])) + ")."
            # item = "#pos(item" + str(i) + ", {}, {}, {category(" + str(int(food_data_categories[i, 0])) + "). value(cost," + str(int(food_data_categories[i, 1])) + "). value(difficulty," + str(int(food_data_categories[i, 2])) + "). value(prepTime," + str(int(food_data_scalars[i, 0])) + ")."
            for j, macro_ingredient in enumerate(food_data_macro_ingredients[i]):
                if macro_ingredient != 0:
                    item = item + " value(" + macro_ingredients_dictionary[j] + "," + str(int(macro_ingredient)) + ")."
            for j, preparation in enumerate(food_data_preparation[i]):
                if (j <= 1) or (3 <= j <= 4) or (j == 6) or (j == 8):
                    continue
                if preparation != 0:
                    item = item + " value(" + preparation_dictionary[j] + "," + str(int(preparation)) + ")."
            item = item + "} )."
            print(item)

        print("")

        if int(max_v) > 0:
            print("#maxv(" + str(max_v) + ").")
        if int(max_p) > 0:
            print("#maxp(" + str(max_p) + ").")

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
        # print("#constant(mg, 2).")
        print("#constant(mg, 3).")
        print("#constant(mg, 4).")
        print("#constant(mg, 5).")
        for key in macro_ingredients_dictionary.keys():
            if key == 0:
                continue
            print("#constant(val, " + macro_ingredients_dictionary[key] + ").")
        for key in preparation_dictionary.keys():
            if (key == 0) or (key == 4) or (key == 6) or (key == 7):
                continue
            print("#constant(val, " + preparation_dictionary[key] + ").")


        sys.stdout = sys.__stdout__
        f.close()
