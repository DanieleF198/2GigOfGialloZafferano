import os
import math
import numpy as np
import ilaspReadWriteUtils as ilasp
import gurobipy as gb
import clingo
import pandas as pd

USERS = [i for i in range(0, 54)]
COUPLES = [45, 105, 210]
max_v_list = [1]
max_p_list = [4, 5]

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
#
#
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

def obtain_hpyerparameters(COUPLE, max_v, max_p, USER):
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
        # if int(max_v_theory) != 3 or int(max_p_theory) != 5:
        #     continue
        f_train = os.path.join(output_dir_for_train_data_dir, filename)
        f_train_data = os.path.join(output_train_data_dir, 'user' + str(USER) + ".txt")
        F_TRAIN = open(f_train)
        data_train = F_TRAIN.read()
        F_TRAIN.close()
        train_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)
        train_size = len(train_set)
        wc_counter = 0
        if ':~' not in data_train:
            continue
        else:
            lines = data_train.split('\n')
            theory = ""
            for line in lines:
                if ':~' in line:
                    theory += line + "\n"
                    wc_counter = wc_counter + 1
        break

    theory_entries = theory.split("\n")
    theory_entries.pop()
    list_of_item = []
    list_of_couple = []
    counter_of_class_one = 0
    counter_of_class_zero = 0
    counter_of_class_minus_one = 0
    y = np.zeros(len(train_set), dtype='int')
    for index, couple in enumerate(train_set):
        list_of_couple.append([couple[0], couple[1]])
        for element in couple:
            if element == '>':
                y[index] = -1
                counter_of_class_minus_one += 1
            elif element == "<":
                y[index] = 1
                counter_of_class_one += 1
            elif element == "=":
                y[index] = 0
                counter_of_class_zero += 1
            else:
                if element in list_of_item:
                    continue
                else:
                    list_of_item.append(element)
    recipes_indexes = np.zeros(len(list_of_item), dtype="int")
    couple_indexes = np.zeros((len(list_of_couple), 2), dtype="int")
    for index, item in enumerate(list_of_item):
        recipes_indexes[index] = int(item.split('m')[1])
    for index, couple in enumerate(list_of_couple):
        for ii in range(2):
            couple_indexes[index, ii] = int(couple[ii].split('m')[1])
    # note: for my own comfort vector of weight is encoded as [1, 2, 3, 4, 5] but remember that clingo reverse the order, and so encode [5, 4, 3, 2, 1]
    X = np.zeros((len(list_of_item), max_p), dtype="float32")   # note: respect to the definition in formulation, here i memorize directly (x[i, j]*p[j])/n[j]
    list_to_check = ["@1", "@2", "@3", "@4", "@5"]
    list_checked = [False, False, False, False, False]
    for iter_check, to_check in enumerate(list_to_check):
        for entry in theory_entries:
            if to_check in entry:
                list_checked[iter_check] = True
    for i in range(0, len(list_of_item)):
        item = items[list_of_item[i]]
        for j in range(0, len(theory_entries)):
            if theory_entries[j].count("V2") > 1:
                print("ERROR: there is a weack constraint with more than 2 variables inside")  # I've already checked manually, but i'll insert this if to prevent and handle this kind of error.
                quit()
            if theory_entries[j].count("), ") > 1:
                print("ERROR: there is a weack constraint with more than 2 literals inside")    # I've already checked manually, but i'll insert this if to prevent and handle this kind of error.
                quit()
            if ("category" in theory_entries[j]) and ("), " not in theory_entries[j]):  # there's only category literal
                literal = theory_entries[j].split(".")[0][3:]
                second_literal = ""
                first_split = theory_entries[j].split(".")[1]
                weight = first_split.split('@')[0]
                level = first_split.split('@')[1]
                weight = weight[1:]
                level = int(level[0:len(level)-1])
            elif("category" in theory_entries[j]) and ("), " in theory_entries[j]): # there's category literal but is not alone
                first_part = theory_entries[j].split("), ")[0]
                second_part = theory_entries[j].split("), ")[1]
                if "category" in first_part:
                    literal = first_part + ")"
                    literal = literal[3:]
                    second_literal_temp = second_part.split("(")[1]
                    second_literal = second_literal_temp.split(",")[0]
                    first_split = second_part.split("[")[1]
                    second_split = first_split.split(',')[0]
                    if "]" in second_split:
                        second_split = second_split[0:len(second_split) - 1]
                    weight = second_split.split('@')[0]
                    level = int(second_split.split('@')[1])
                    temp_value = literal
                    literal = second_literal
                    second_literal = temp_value
                else:
                    second_literal = second_part.split(".")[0]
                    literal_temp = first_part.split("(")[1]
                    literal = literal_temp.split(",")[0]
                    first_split = second_part.split("[")[1]
                    second_split = first_split.split(',')[0]
                    if "]" in second_split:
                        second_split = second_split[0:len(third_split) - 1]
                    weight = second_split.split('@')[0]
                    level = int(second_split.split('@')[1])
            else:
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
                weight = third_split.split('@')[0]
                level = int(third_split.split('@')[1])
            if literal in item and second_literal in item:
                if "V" in weight:
                    if "-" in weight:
                        # X[i, level - 1] = -1
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
                        # X[i, level - 1] = 1
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
                    X[i, level - 1] = int(weight)

    for iter_check in range(max_p):
        if np.all(X[:, iter_check] == 0):
            list_checked[iter_check] = False
    if True not in list_checked:
        return 0, [[0, 0, 0, 0, 0]]
    # DEFINE GUROBI MODEL

    # VARIABLES DEFINITION (X and y are already defined as nparray)
    if COUPLE == 210:
        l = 190
    else:
        l = COUPLE
    if COUPLE == 150:
        l = len(couple_indexes)
    gb_model = gb.Model()
    z = gb_model.addVars(max_p, vtype=gb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='z')
    t = gb_model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0.0, ub=0.0, name='t')
    if COUPLE == 150:
        q = gb_model.addVars(l, vtype=gb.GRB.CONTINUOUS, name='q', lb=-gb.GRB.INFINITY)
    else:
        q = gb_model.addVars(len(list_of_item), vtype=gb.GRB.CONTINUOUS, name='q', lb=-gb.GRB.INFINITY)
    s = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s')
    # s1 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s1')
    # s2 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s2')
    # s3 = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s3')

    gb_model.update()
    gb_model.write('./Lp_files/model.lp')
    # v decomposition:

    # b1[l] = q[l1] > q[l2] + t ? 1 : 0
    # b2[l] = q[l1] < q[l2] - t ? 1 : 0
    # v[l] = b2[l] - b1[l]

    # note that b1[l] and b2[l] exclude themselves  (t could be equal to 0)
    #                   ----------[q[l2]-t----------q[l2]----------q[l2]+t]----------
    #   b1[l]=1     =>                                                    (--q[l1]---
    #   b2[l]=1     =>  ---q[l1]--)
    #   b1/2[l] = 0 =>            [-----------------q[l1]-----------------]
    #   b1/2[l] = 1 =>                            impossible

    # 3 cases:  b1[l] |   b2[l]
    #            0         0   [case in which classification return 0]
    #            0         1   [case in which classification return -1]
    #            1         0   [case in which classification return 1]
    #            1         1   [impossible]

    # if q[l1] > q[l2] + t then b1[l] = 1 and b2[l] = 0, so v[l] = 0 - 1 = -1
    # if q[l1] < q[l2] - t then b1[l] = 0 and b2[l] = 1, so v[l] = 1 - 0 = 1
    # if q[l2] - t <= q[l1] <= q[l2] + t then b1[l] = 0 and b2[l] = 0 and so v[l] = 0 - 0 = 0
    # there aren't other cases, and v[l] get values correctly respect his definition
    # finally, I use this gurobi guide for constraint definition: https://support.gurobi.com/hc/en-us/articles/4414392016529-How-do-I-model-conditional-statements-in-Gurobi-


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
        if j >= 0:
            if list_checked[j]:
                if j == 0:
                    continue
                for k in range(0, j):
                    gb_model.addConstr(z[j] >= z[k])      # (2)
            else:
                gb_model.addConstr(z[j] == 0)  # (2)
    first_z_reached = False
    for j in range(max_p):
        if first_z_reached:
            break
        elif list_checked[j]:
            gb_model.addConstr(z[j] >= 0.01)
            first_z_reached = True
    gb_model.update()
    gb_model.write('./Lp_files/model.lp')
    # temp for test
    # for j in reversed(range(max_p)):
    #     if j > 1:
    #         if list_checked[j]:
    #             gb_model.addConstr(z[j] >= gb.quicksum(z[i] for i in range(j)) + 0.01)
    # TEST:
    # z[j]*max(X[i, j] for each i as food in x) > sum_k(z[k]*max(x[i, k] for each i as food in x)) with k < j
    # NOTE: here max is intended in abs value
    max_values_per_z =[0, 0, 0, 0, 0]
    for sample in X:
        for index_value, value in enumerate(sample):
            if max_values_per_z[index_value] < abs(value):
                max_values_per_z[index_value] = abs(value)
    # normalize max_values_per_p so that it does sum up to 1
    for j in reversed(range(max_p)):
        if j > 0:
            if list_checked[j]:
                gb_model.addConstr(z[j] * max_values_per_z[j] >= gb.quicksum(z[i] * max_values_per_z[i] for i in range(j)) + 0.01)
    gb_model.update()
    gb_model.write('./Lp_files/model.lp')
    for i in range(len(list_of_item)):
        gb_model.addConstr(q[i] == gb.quicksum(X[i, j] * z[j] for j in range(max_p) if list_checked[j]))     # (3)
    gb_model.update()
    gb_model.write('./Lp_files/model.lp')
    couple_counter = 0
    eps = 0.0000001
    M = 100
    if COUPLE == 150:
        for couple in couple_indexes:
            inserted = False
            for l1, recipe1 in enumerate(recipes_indexes):
                if inserted:
                    break
                for l2, recipe2 in enumerate(recipes_indexes):
                    if inserted:
                        break
                    if recipe1 == couple[0] and recipe2 == couple[1]:
                        gb_model.addConstr(q[l1] >= q[l2] + t + eps - M * (1 - z1[couple_counter]))
                        gb_model.addConstr(q[l1] <= q[l2] + t + M * z1[couple_counter])
                        gb_model.addConstr(q[l1] <= q[l2] - t - eps + M * (1 - z2[couple_counter]))  # - eps
                        gb_model.addConstr(q[l1] >= q[l2] - t - M * z2[couple_counter])
                        gb_model.addConstr(z1[couple_counter] + z2[couple_counter] <= 1)
                        gb_model.addConstr(b1[couple_counter] == z1[couple_counter])
                        gb_model.addConstr(b2[couple_counter] == z2[couple_counter])
                        couple_counter += 1
                        inserted = True
    else:
        for l1, recipe1 in enumerate(recipes_indexes):
            for l2, recipe2 in enumerate(recipes_indexes):
                if l2 <= l1:
                    continue
                gb_model.addConstr(q[l1] >= q[l2] + t + eps - M * (1 - z1[couple_counter]))
                gb_model.addConstr(q[l1] <= q[l2] + t + M * z1[couple_counter])
                gb_model.addConstr(q[l1] <= q[l2] - t - eps + M * (1 - z2[couple_counter]))  # - eps
                gb_model.addConstr(q[l1] >= q[l2] - t - M * z2[couple_counter])
                gb_model.addConstr(z1[couple_counter] + z2[couple_counter] <= 1)
                gb_model.addConstr(b1[couple_counter] == z1[couple_counter])
                gb_model.addConstr(b2[couple_counter] == z2[couple_counter])
                couple_counter += 1
    gb_model.update()
    gb_model.write('./Lp_files/model.lp')
    gb_model.addConstrs(v[couple_number] == b2[couple_number] - b1[couple_number] for couple_number in range(l))     # (4)
    gb_model.update()
    gb_model.write('./Lp_files/model.lp')

    couple_counter = 0
    for l1, recipe1 in enumerate(recipes_indexes):
        for l2, recipe2 in enumerate(recipes_indexes):
            if COUPLE == 150:
                if couple_counter >= l:
                    gb_model.addGenConstrIndicator(s[couple_counter], True, v[couple_counter] == y[couple_counter])
                    couple_counter += 1
            else:
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
        return -1, [-1, -1, -1, -1, -1, -1]
        
    list_to_return = []
    for i, z_value in enumerate(z):
        list_to_return.append(np.round(z[i].X, decimals=5))
    # for i, v_value in enumerate(v):
        # print(str(np.round(v[i].X, decimals=5)))
    return np.round(t.X, decimals=5), [list_to_return]
                
choices = [1]
for choice in choices:
    if choice == 0:
        continue
    path = './Data8Component2Std/testOutput/results_zero_after_opt-after-grid(single-user).csv'
    with open(path, 'w+', encoding='UTF8') as f_output:
        f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;ACCURACYP;PRECISIONP;RECALLP;TRAIN_TIME;THEORY;NUMBER_WC;treshold;F1;F2;F3;F4;F5\n")
        for COUPLE in COUPLES:
            for max_v in max_v_list:
                for max_p in max_p_list:
                    for USER in USERS:
                        if COUPLE == 45:
                            if max_v != 1 or max_p != 4:
                                continue
                        else:
                            if max_v != 1 or max_p != 5:
                                continue
                        if COUPLE == 150:
                            if max_v != 3 or max_p != 5:
                                continue
                        returned_t, returned_z = obtain_hpyerparameters(COUPLE=COUPLE, max_v=max_v, max_p=max_p, USER=USER)
                        returned_z = returned_z[0]
                        if -1 == returned_z:
                            continue
                        if len(returned_z) < 5:     # for cases such as 45 COUPLE where we have 4 wc, but in the code then this cause "list index out of range"
                            returned_z.append(0)
                        if int(choice) == 0:
                            output_train_data_dir = "./Data8Component2Std/users_new_version_second/no_zero/train/" + str(COUPLE) + "Couples/"
                            output_dir_for_train_data_dir = "./Data8Component2Std/final/users/no_zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                            output_test_data_dir = "./Data8Component2Std/final/users/no_zero/test/105Couples/User" + str(USER) + "/testFiles/"
                        else:
                            output_train_data_dir = "./Data8Component2Std/users_new_version_second/zero/train/" + str(COUPLE) + "Couples/"
                            output_dir_for_train_data_dir = "./Data8Component2Std/final/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                            output_test_data_dir = "./Data8Component2Std/final/users/zero/test/105Couples/User" + str(USER) + "/testFiles/"
                        for filename in os.listdir(output_dir_for_train_data_dir):
                            if "default" in filename:
                                continue
                            start_index_max_v_max_p = filename.find("max-v(") + len("max-v(")
                            first_middle_index_max_v_max_p = filename.find(")-max_p(")
                            second_middle_index_max_v_max_p = filename.find(")-max_p(") + len(")-max_p(")
                            end_index_max_v_max_p = filename.find(").txt")
                            max_v = int(filename[start_index_max_v_max_p:first_middle_index_max_v_max_p])
                            max_p = int(filename[second_middle_index_max_v_max_p:end_index_max_v_max_p])
                            # if int(max_v) == 10 or int(max_p) == 10:
                            #     continue
                            # if int(max_v) != int(max_p):
                            #     continue
                            if COUPLE == 45:
                                if max_v != 1 or max_p != 4:
                                    continue
                            else:
                                if max_v != 1 or max_p != 5:
                                    continue
                            if COUPLE == 150:
                                if max_v != 3 or max_p != 5:
                                    continue
                            if int(max_v) > 0 and int(max_p) > 0:
                                items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                                language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                            elif int(max_v) > 0 or int(max_p) > 0:
                                if int(max_v) > 0:
                                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                else:
                                    items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                                    language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(" + str(max_p) + ").las")
                            else:
                                items = ilasp.itemsFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                                language_bias = ilasp.languageBiasFromFile("Data8Component2Std/recipes/recipes_max_v(default)-max_p(default).las")
                            f_train = os.path.join(output_dir_for_train_data_dir, filename)
                            f_train_data = os.path.join(output_train_data_dir, 'user' + str(USER) + ".txt")
                            temp_filename = filename.replace("outputTrain", "test")
                            test_filename = temp_filename.replace("txt", "las")
                            test_filename = "test_max-v(2)-max_p(3).las" # picked one random, it's the same
                            f_test = os.path.join(output_test_data_dir, test_filename)
                            F_TRAIN = open(f_train)
                            data_train = F_TRAIN.read()
                            F_TRAIN.close()
                            # train_set = ilasp.preferencesFromFileSpaces(f_train_data)
                            train_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)
                            test_set = ilasp.preferencesFromFileSign(f_test)
                            # test_set = ilasp.preferencesFromFileSpacesAndSign(f_train_data)

                            train_size = len(train_set)
                            test_size = len(test_set)

                            wc_counter = 0
                            if ':~' not in data_train:
                                continue
                            else:
                                lines = data_train.split('\n')
                                theory = ""
                                training_time = ""
                                for line in lines:
                                    if ':~' in line:
                                        theory += line + "\n"
                                        wc_counter = wc_counter + 1
                                    if '%% Total' in line:
                                        start_index = line.find(':')
                                        end_index = line.find('s')
                                        training_time += line[start_index + 2:end_index]
                            if wc_counter == 1:
                                if COUPLE == 150:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 45:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 105:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                elif COUPLE == 210:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 1 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                break
                            if wc_counter == 2:
                                if COUPLE == 150:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 45:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 105:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                elif COUPLE == 210:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 2 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                break
                            if wc_counter == 3:
                                if COUPLE == 150:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 45:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 105:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                elif COUPLE == 210:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 3 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                break
                            if wc_counter == 4:
                                if COUPLE == 150:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 45:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 105:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                elif COUPLE == 210:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                break
                            if wc_counter == 5:
                                if COUPLE == 150:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                if COUPLE == 45:
                                    print("error")  # max_p is set to 4 so couldn't be that there are 5 weak constraints
                                if COUPLE == 105:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                elif COUPLE == 210:
                                    results = ilasp.test_cm_grid(theory, items, test_set, returned_t, returned_z)  # cases where there are 4 weak constraint but maxp is setted to 5 (in this case the missing weak constraint could be of any level)
                                    f_output.write(str(USER) + ";" + str(max_v) + ";" + str(max_p) + ";3;" + str(train_size) + ";" + str(test_size) + ";" + str(results["avg_accuracy"]) + ";" + str(results["avg_precision"]) + ";" + str(results["avg_recall"]) + ";" + str(training_time) + ";" + theory.replace("\n", "") + ";" + str(float(wc_counter)) + ";" + str(returned_t) + ";" + str(returned_z[0]) + ";" + str(returned_z[1]) + ";" + str(returned_z[2]) + ";" + str(returned_z[3]) + ";" + str(returned_z[4]) + "\n")
                                break