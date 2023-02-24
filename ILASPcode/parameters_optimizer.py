import os
import math
import numpy as np
import ilaspReadWriteUtils as ilasp
import gurobipy as gb
import clingo
import pandas as pd

from CompareStableModels import create_preamble



# Problem formulation
# Variables:
# X[i, j] = value of penalty j on recipes i respect to current theory
# z[j] = multiplicative factor of penalty j
# t = threshold
# q[i] = penalty score on recipes i respect to the current theory
# v[l] = classification of couple l respect to the theory (1 → A ≻ B; 0 → A ∼ B; -1 → A ≺ B)
# y[l] = correct classification of couple l
# s[l] = 1 if v[l] == y[l]; 0 otherwise
# Constraints:
# sum{j=1,...,h}(z[j]) = 1
# z[1] <= z[2] <= ... <= z[h]
# q[i] = sum{j=1,...,h}(X[i, j]z[j]) for each i=1,...,n
# v[l] = 1 if q[l1] > q[l2] + t; -1 if q[l1] < q[l2] - t; 0 otherwise for each l=(l1, l2)
# z[j] ∈ [0, 1]
# t ∈ [0, 1]
# q[i] ∈ Integers
# Objective function
# f = max(sum{l=1,...,k}(s[l]))
# Notes:
# h is the number of weak constraint given by the theory (in our case, h ∈ [1, 5])
# n is the number of recipes in the dataset in consideration (in our case n = 10, 15 or 20)
# k is the number of couple in the dataset in consideration (in our case, k = 45, 105 or 190; respect to n value)

# DEFINE GENERALITIES

USERS = [i for i in range(0, 54)]
COUPLES = [45, 105, 210]
max_v_list = [1]
max_p_list = [4, 5]

# LOAD RECIPES DATA

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

# convert from classes to macro-classes
food_data_macro_ingredients = np.zeros((len(linesOfI), 12), dtype='float32')
for i, row in enumerate(food_data_ingredients):
    for j, ingredient in enumerate(row):
        if (j == 1) or (4 <= j <= 8) or (j == 10) or (j == 14) or (16 <= j <= 19) or (21 <= j <= 24) or (28 <= j <= 29):
            continue
        if j == 5:
            food_data_macro_ingredients[i, 0] = food_data_macro_ingredients[i, 0] + ingredient
        if j == 4 or j == 6 or j == 18 or j == 28 or j == 35:
            food_data_macro_ingredients[i, 1] = food_data_macro_ingredients[i, 1] + ingredient
        if j == 11:
            food_data_macro_ingredients[i, 2] = food_data_macro_ingredients[i, 2] + ingredient
        if j == 2 or j == 12:
            food_data_macro_ingredients[i, 3] = food_data_macro_ingredients[i, 3] + ingredient
        if j == 1 or j == 8 or j == 21 or j == 24 or j == 30:
            food_data_macro_ingredients[i, 4] = food_data_macro_ingredients[i, 4] + ingredient
        if j == 3 or j == 9 or j == 15 or j == 29 or j == 34:
            food_data_macro_ingredients[i, 5] = food_data_macro_ingredients[i, 5] + ingredient
        if j == 0 or j == 7 or j == 22 or j == 23 or j == 26 or j == 32:
            food_data_macro_ingredients[i, 6] = food_data_macro_ingredients[i, 6] + ingredient
        if j == 20:
            food_data_macro_ingredients[i, 7] = food_data_macro_ingredients[i, 7] + ingredient
        if j == 10 or j == 13:
            food_data_macro_ingredients[i, 8] = food_data_macro_ingredients[i, 8] + ingredient
        if j == 16 or j == 19 or j == 25 or j == 27:
            food_data_macro_ingredients[i, 9] = food_data_macro_ingredients[i, 9] + ingredient
        if j == 31:
            food_data_macro_ingredients[i, 10] = food_data_macro_ingredients[i, 10] + ingredient
        if j == 14 or j == 17 or j == 33:
            food_data_macro_ingredients[i, 11] = food_data_macro_ingredients[i, 11] + ingredient

linesOfP = dataP.split('\n')
food_data_preparation = np.zeros((len(linesOfP), 8), dtype='float32')
for i, line in enumerate(linesOfP):

    values = [x for x in line.split(' ')[1:]]
    food_data_preparation[i, :] = values

all_data = np.concatenate([food_data_categories, food_data_scalars, food_data_macro_ingredients, food_data_preparation], axis=1)

macro_ingredients_dictionary = {"cereali": 0,
                                "latticini": 1,
                                "uova": 2,
                                "farinacei":3,
                                "frutta": 4,
                                "erbe_spezie_e_condimenti": 5,
                                "carne": 6,
                                "funghi_e_tartufi": 7,
                                "pasta": 8,
                                "pesce": 9,
                                "dolcificanti": 10,
                                "verdure_e_ortaggi": 11}


preparation_dictionary = {"bollitura": 0,
                          "rosolatura": 1,
                          "frittura": 2,
                          "marinatura": 3,
                          "mantecatura": 4,
                          "forno": 5,
                          "cottura_a_fiamma": 6,
                          "stufato": 7}


macro_ingredients_norm_dictionary = {"cereali": 0.0,
                                "latticini": 1.0,
                                "uova": 4.0,
                                "farinacei": 7.0,
                                "frutta": 2.0,
                                "erbe_spezie_e_condimenti": 11.0,
                                "carne": 5.0,
                                "funghi_e_tartufi": 3.0,
                                "pasta": 5.0,
                                "pesce": 9.0,
                                "dolcificanti": 1.0,
                                "verdure_e_ortaggi": 8.0}

preparation_norm_dictionary = {"bollitura": 5.0,
                          "rosolatura": 5.0,
                          "frittura": 5.0,
                          "marinatura": 3.0,
                          "mantecatura": 4.0,
                          "forno": 5.0,
                          "cottura_a_fiamma": 5.0,
                          "cottura_a_vapore": 5.0,
                          "stufato": 5.0}

results = pd.DataFrame(columns=("USER_ID", "MAX_V", "MAX_P", "DATASET_SIZE", "OPT", "THRESHOLD", "F1", "F2", "F3", "F4", "F5"))

for COUPLE in COUPLES:
    for max_v in max_v_list:
        for max_p in max_p_list:
            for USER in USERS:
                if COUPLE == 210:
                    break
                if COUPLE == 45:
                    if max_v != 1 or max_p != 4:
                        continue
                else:
                    if max_v != 1 or max_p != 5:
                        continue

                if COUPLE == 210:
                    print("dataset size = 190;")
                else:
                    print("dataset size = " + str(COUPLE) + ";")
                print("max_v = " + str(max_v) + ", max_p = " + str(max_p) + ";")
                print("user id = " + str(USER))
                print("----------------------------------------------------------------------------------")

                # LOAD DATA

                if int(max_v) > 0 and int(max_p) > 0:
                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                elif int(max_v) > 0 or int(max_p) > 0:
                    if int(max_v) > 0:
                        items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                        language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                    else:
                        items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                        language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                else:
                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                preferences_train = ilasp.preferencesFromFileSpacesAndSign("Data8Component2Std/users_new_version_second/no_zero/train/210Couples/user" + str(USER) + ".txt")
                output_train_data_dir = "./Data8Component2Std/users_new_version_second/zero/train/" + str(COUPLE) + "Couples/"
                output_dir_for_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                for filename in os.listdir(output_dir_for_train_data_dir):
                    if "default" in filename:
                        continue
                    start_index_max_v_max_p = filename.find("max-v(") + len("max-v(")
                    first_middle_index_max_v_max_p = filename.find(")-max_p(")
                    second_middle_index_max_v_max_p = filename.find(")-max_p(") + len(")-max_p(")
                    end_index_max_v_max_p = filename.find(").txt")
                    max_v_theory = int(filename[start_index_max_v_max_p:first_middle_index_max_v_max_p])
                    max_p_theory = int(filename[second_middle_index_max_v_max_p:end_index_max_v_max_p])
                    if COUPLE == 45:
                        if int(max_v_theory) != 1 or int(max_p_theory) != 4:
                            continue
                    else:
                        if int(max_v_theory) != 1 or int(max_p_theory) != 5:
                            continue
                    f_train = os.path.join(output_dir_for_train_data_dir, filename)
                    f_train_data = os.path.join(output_train_data_dir, 'user' + str(USER) + ".txt")
                    F_TRAIN = open(f_train)
                    data_train = F_TRAIN.read()
                    F_TRAIN.close()
                    train_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)
                    train_size = len(train_set)
                    if ':~' not in data_train:
                        continue
                    else:
                        lines = data_train.split('\n')
                        theory = ""
                        for line in lines:
                            if ':~' in line:
                                theory += line + "\n"
                    break

                # DEFINE VARIABLES
                if COUPLE == 45:
                    n = 10
                elif COUPLE == 105:
                    n = 15
                else:
                    n = 20
                theory_entries = theory.split("\n")
                theory_entries.pop()
                list_of_item = []
                y = np.zeros(len(train_set), dtype='int')
                for index, couple in enumerate(train_set):
                    for element in couple:
                        if element == '>':
                            y[index] = 1
                        elif element == "<":
                            y[index] = -1
                        elif element == "=":
                            y[index] = 0
                        else:
                            if element in list_of_item:
                                continue
                            else:
                                list_of_item.append(element)
                recipes_indexes = np.zeros(n, dtype="int")
                for index, item in enumerate(list_of_item):
                    recipes_indexes[index] = int(item.split('m')[1])
                # note: for my own comfort vector of weight is encoded as [1, 2, 3, 4, 5] but remember that clingo reverse the order, and so encode [5, 4, 3, 2, 1]
                X = np.zeros((n, max_p), dtype="float32")   # note: respect to the definition in formulation, here i memorize directly (x[i, j]*p[j])/n[j]
                for i in range(0, n):
                    item = items[list_of_item[i]]
                    for j in range(0, len(theory_entries)):
                        first_split = theory_entries[j].split('(')[1]
                        second_literal = ""
                        if " " in first_split[0:len(first_split)-4]:
                            literal = first_split.split(',')[0]
                            second_literal = first_split.split('), ')[1]
                            second_split = theory_entries[j].split('[')[1]
                        else:
                            literal = first_split.split(',')[0]   # NOTE: max_v = 1, so we have at most 1 literal
                            second_split = first_split.split('[')[1]
                        third_split = second_split.split(',')[0]
                        if "]" in third_split:
                            third_split = third_split[0:len(third_split)-1]
                        weigth = third_split.split('@')[0]
                        level = int(third_split.split('@')[1])
                        if literal in item or second_literal in item:
                            if "V" in weigth:
                                if "-" in weigth:
                                    if literal in macro_ingredients_dictionary:
                                        X[i, level-1] = -(all_data[recipes_indexes[i], 4 + macro_ingredients_dictionary[literal]])/macro_ingredients_norm_dictionary[literal]
                                    elif literal in preparation_dictionary:
                                        X[i, level-1] = -(all_data[recipes_indexes[i], 4 + len(macro_ingredients_dictionary)+preparation_dictionary[literal]])/preparation_norm_dictionary[literal]
                                    elif literal == "cost":
                                        X[i, level-1] = -all_data[recipes_indexes[i], 1]
                                    elif literal == "prepTime":
                                        X[i, level - 1] = -(all_data[recipes_indexes[i], 2])/280
                                    else:
                                        X[i, level-1] = -all_data[recipes_indexes[i], 2]
                                else:
                                    if literal in macro_ingredients_dictionary:
                                        X[i, level-1] = (all_data[recipes_indexes[i], 4 + macro_ingredients_dictionary[literal]])/macro_ingredients_norm_dictionary[literal]
                                    elif literal in preparation_dictionary:
                                        X[i, level-1] = (all_data[recipes_indexes[i], 4 + len(macro_ingredients_dictionary)+preparation_dictionary[literal]])/preparation_norm_dictionary[literal]
                                    elif literal == "cost":
                                        X[i, level-1] = all_data[recipes_indexes[i], 1]
                                    elif literal == "prepTime":
                                        X[i, level - 1] = -(all_data[recipes_indexes[i], 2])/280
                                    else:
                                        X[i, level-1] = all_data[recipes_indexes[i], 2]
                            else:
                                if second_literal != "":
                                    if second_literal in macro_ingredients_dictionary:
                                        X[i, level - 1] = (int(weigth))/macro_ingredients_norm_dictionary[literal]
                                    elif second_literal in preparation_dictionary:
                                        X[i, level - 1] = (int(weigth)) / preparation_norm_dictionary[literal]
                                    elif second_literal == "prepTime":
                                        X[i, level - 1] = (int(weigth)) / 280
                                    else:
                                        X[i, level - 1] = int(weigth)
                                elif literal in item:   # this double check is caused because "" is always in items
                                    if literal in macro_ingredients_dictionary:
                                        X[i, level - 1] = (int(weigth)) / macro_ingredients_norm_dictionary[literal]
                                    elif literal in preparation_dictionary:
                                        X[i, level - 1] = (int(weigth)) / preparation_norm_dictionary[literal]
                                    elif literal == "prepTime":
                                        X[i, level - 1] = (int(weigth)) / 280
                                    else:
                                        X[i, level - 1] = int(weigth)


                # DEFINE GUROBI MODEL

                # VARIABLES DEFINITION (X and y are already defined as nparray)
                if COUPLE == 210:
                    l = 190
                else:
                    l = COUPLE
                gb_model = gb.Model()
                z = gb_model.addVars(max_p, vtype=gb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='z')
                t = gb_model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='t')
                q = gb_model.addVars(n, vtype=gb.GRB.CONTINUOUS, name='q', lb=-gb.GRB.INFINITY)
                s = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s')
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')
                # v decomposition:

                # b1[l] = q[l1] > q[l2] + t ? 1 : 0
                # b2[l] = q[l1] < q[l2] - t ? 1 : 0
                # v[l] = b1[l] - b2[l]

                # note that b1[l] and b2[l] exclude themselves

                # 3 cases:  b1[l] |   b2[l]
                #            0         0   [case in which classification return 0]
                #            0         1   [case in which classification return -1]
                #            1         0   [case in which classification return 1]
                #            1         1   [impossible]

                # if q[l1] > q[l2] + t then b1[l] = 1 and b2[l] = 0, so v[l] = 1 - 0 = 1
                # if q[l1] < q[l2] - t then b1[l] = 0 and b2[l] = 1, so v[l] = 0 - 1 = -1
                # if q[l2] - t <= q[l1] <= q[l2] + t  then b1[l] = 0 and b2[l] = 0 and so v[l] = 0 - 0 = 0
                # there aren't other cases, and v[l] get values correctly respect his definition


                b1 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='b1')
                b2 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='b2')
                z1 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='z1')    # support variables
                z2 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='z2')    # support variables
                v = gb_model.addVars(l, vtype=gb.GRB.INTEGER, lb=-1, ub=1, name='v')
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')

                # CONSTRAINT DEFINITION
                gb_model.addConstr(z.sum('*') == 1)     # (1)
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')
                for j in reversed(range(max_p)):
                    if j > 0:
                        for k in range(0, j):
                            gb_model.addConstr(z[j] >= z[k])      # (2)
                gb_model.addConstr(z[0] >= 0)
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')
                for i in range(n):
                    gb_model.addConstr(q[i] == gb.quicksum(X[i, j] * z[j] for j in range(max_p)))     # (3)
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')
                couple_counter = 0
                eps = 0.01
                M = 100
                for l1, recipe1 in enumerate(recipes_indexes):
                    for l2, recipe2 in enumerate(recipes_indexes):
                        if l2 <= l1:
                            continue
                        gb_model.addConstr(q[l1] >= q[l2] + t + eps - M * (1 - z1[couple_counter]))
                        gb_model.addConstr(q[l1] <= q[l2] + t + M * z1[couple_counter])
                        gb_model.addConstr(q[l1] <= q[l2] - t - eps + M * (1 - z2[couple_counter]))   # - eps
                        gb_model.addConstr(q[l1] >= q[l2] - t - M * z2[couple_counter])
                        gb_model.addConstr(z1[couple_counter] + z2[couple_counter] <= 1)
                        gb_model.addGenConstrIndicator(z1[couple_counter], True, b1[couple_counter] == 1)
                        gb_model.addGenConstrIndicator(z1[couple_counter], False, b1[couple_counter] == 0)
                        gb_model.addGenConstrIndicator(z2[couple_counter], True, b2[couple_counter] == 1)
                        gb_model.addGenConstrIndicator(z2[couple_counter], False, b2[couple_counter] == 0)
                        couple_counter += 1
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')
                gb_model.addConstrs(v[couple_number] == b1[couple_number] - b2[couple_number] for couple_number in range(l))     # (4)
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')

                couple_counter = 0
                for l1, recipe1 in enumerate(recipes_indexes):
                    for l2, recipe2 in enumerate(recipes_indexes):
                        if l2 <= l1:
                            continue
                        gb_model.addGenConstrIndicator(s[couple_counter], True, v[couple_counter] == y[couple_counter])
                        couple_counter += 1
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')

                # objective function

                gb_model.setObjective(gb.quicksum(s), sense=gb.GRB.MAXIMIZE,)
                gb_model.update()
                gb_model.write('./Lp_files/model.lp')

                gb_model.Params.LogToConsole = 0
                gb_model.optimize()

                if gb_model.Status == 3:
                    gb_model.computeIIS()
                    gb_model.update()
                    gb_model.write('./Lp_files/IIS/model' + str(COUPLE) + '-' + str(USER) + '.ilp')
                    print("infeasible")
                    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                    continue

                print("optimal solution: " + str(gb_model.ObjVal))

                print("z values")
                for i, z_value in enumerate(z):
                    print("z[" + str(i) + "] = " + str(z[i]))

                print("t = " + str(t))

                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                df_index = len(results)
                if COUPLE == 210:
                    results.loc[df_index] = [USER, max_v, max_p, 190, int(gb_model.ObjVal), np.round(t.X, decimals=5), 0, 0, 0, 0, 0]
                else:
                    results.loc[df_index] = [USER, max_v, max_p, COUPLE, int(gb_model.ObjVal), np.round(t.X, decimals=5), 0, 0, 0, 0, 0]
                for i, z_value in enumerate(z):
                    results.iloc[df_index, 6+i] = np.round(z[i].X, decimals=5)
results.to_csv("./Data8Component2Std/optResultsSearch.csv", sep=";", index=False)
