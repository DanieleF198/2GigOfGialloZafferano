import os
import math
import numpy as np
import ilaspReadWriteUtils as ilasp
import gurobipy as gb
import clingo
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing, decomposition


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
COUPLES = [45, 105, 210, 150]
max_v_list = [1]
max_p_list = [5]

# LOAD RECIPES DATA
pca_data = pd.read_csv("./PCAexperiment/PCA_values_recipes.csv", header=0, sep=";").to_numpy()

PC_dictionary = {"pc0": 0,
                 "pc1": 1,
                 "pc2": 2,
                 "pc3": 3,
                 "pc4": 4,
                 "pc5": 5,
                 "pc6": 6,
                 "pc7": 7,
                 "pc8": 8,
                 "pc9": 9,
                 "pc10": 10,
                 "pc11": 11,
                 "pc12": 12,
                 "pc13": 13,
                 "pc14": 14,
                 "pc15": 15,
                 "pc16": 16,
                 "pc17": 17,
                 "pc18": 18,
                 "pc19": 19
                 }


PC_norm_dictionary = {"pc0": 106,   # note that these component have this high value because in ILASP negative and float values are not allowed, and so in the files the PC components have been rescaled and shifted.
                      "pc1": 118,
                      "pc2": 184,
                      "pc3": 124,
                      "pc4": 110,
                      "pc5": 98,
                      "pc6": 137,
                      "pc7": 104,
                      "pc8": 110,
                      "pc9": 87,
                      "pc10": 91,
                      "pc11": 87,
                      "pc12": 101,
                      "pc13": 87,
                      "pc14": 87,
                      "pc15": 79,
                      "pc16": 89,
                      "pc17": 101,
                      "pc18": 88,
                      "pc19": 79
                 }

results = pd.DataFrame(columns=("USER_ID", "MAX_V", "MAX_P", "DATASET_SIZE", "OPT", "THRESHOLD", "F1", "F2", "F3", "F4", "F5"))

PCAindexes = [15]   # [5, 10, 15, 20]
scopes = ["_original"]  # ["", "_original"]
for scope in scopes:
    for PCAindex in PCAindexes:
        for COUPLE in COUPLES:
            for max_v in max_v_list:
                for max_p in max_p_list:
                    for USER in USERS:
                        if COUPLE == 150 and scope != "_original":
                            continue
                        if COUPLE != 150 and scope == "_original":
                            continue
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
                            items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                            language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
                        elif int(max_v) > 0 or int(max_p) > 0:
                            if int(max_v) > 0:
                                items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                                language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
                            else:
                                items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                                language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(" + str(max_p) +").las")
                        else:
                            items = ilasp.itemsFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(default).las")
                            language_bias = ilasp.languageBiasFromFile("PCAexperiment/recipes" + str(PCAindex) + "/recipes_max_v(default)-max_p(default).las")
                        if scope == "_original":
                            output_train_data_dir = "./PCAexperiment/users_original/zero/train/150Couples/"
                            output_dir_for_train_data_dir = "./PCAexperiment/final_original" + str(PCAindex) + "/users/zero/train/150Couples/User" + str(USER) + "/outputTrain/"
                        else:
                            output_train_data_dir = "./PCAexperiment/users_new_version_second/zero/train/" + str(COUPLE) + "Couples/"
                            output_dir_for_train_data_dir = "./PCAexperiment/final" + str(PCAindex) + "/users/zero/train/" + str(COUPLE) + "Couples/User" + str(USER) + "/outputTrain/"
                        for filename in os.listdir(output_dir_for_train_data_dir):
                            if "default" in filename:
                                continue
                            start_index_max_v_max_p = filename.find("max-v(") + len("max-v(")
                            first_middle_index_max_v_max_p = filename.find(")-max_p(")
                            second_middle_index_max_v_max_p = filename.find(")-max_p(") + len(")-max_p(")
                            end_index_max_v_max_p = filename.find(").txt")
                            max_v_theory = int(filename[start_index_max_v_max_p:first_middle_index_max_v_max_p])
                            max_p_theory = int(filename[second_middle_index_max_v_max_p:end_index_max_v_max_p])
                            # if COUPLE == 45:
                            #     if int(max_v_theory) != 1 or int(max_p_theory) != 4:
                            #         continue
                            # else:
                            #     if int(max_v_theory) != 1 or int(max_p_theory) != 5:
                            #         continue
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
        
                        theory_entries = theory.split("\n")
                        theory_entries.pop()
                        list_of_item = []
                        list_of_couple = []
                        y = np.zeros(len(train_set), dtype='int')
                        for index, couple in enumerate(train_set):
                            list_of_couple.append([couple[0], couple[1]])
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
                        recipes_indexes = np.zeros(len(list_of_item), dtype="int")
                        couple_indexes = np.zeros((len(list_of_couple), 2), dtype="int")
                        for index, item in enumerate(list_of_item):
                            recipes_indexes[index] = int(item.split('m')[1])
                        for index, couple in enumerate(list_of_couple):
                            for ii in range(2):
                                couple_indexes[index, ii] = int(couple[ii].split('m')[1])
                        # note: for my own comfort vector of weight is encoded as [1, 2, 3, 4, 5] but remember that clingo reverse the order, and so encode [5, 4, 3, 2, 1]
                        X = np.zeros((len(list_of_item), max_p), dtype="float32")   # note: respect to the definition in formulation, here i memorize directly (x[i, j]*p[j])/n[j]
                        for i in range(0, len(list_of_item)):
                            item = items[list_of_item[i]]
                            for j in range(0, len(theory_entries)):
                                if theory_entries[j].count("), ") > 1:
                                    print("ERROR: there is a weack constraint with more than 2 literals inside")    # I've already checked manually, but i'll insert this if to prevent and handle this kind of error.
                                    quit()
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
                                            X[i, level-1] = -(pca_data[recipes_indexes[i], PC_dictionary[literal]])/PC_norm_dictionary[literal]
                                        else:
                                            X[i, level-1] = (pca_data[recipes_indexes[i], PC_dictionary[literal]])/PC_norm_dictionary[literal]
                                    else:
                                        if second_literal != "":
                                            X[i, level - 1] = (int(weight))/PC_norm_dictionary[literal]
                                        elif literal in item:   # this double check is caused because "" is always in items
                                            X[i, level - 1] = (int(weight)) / PC_norm_dictionary[literal]

        
        
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
                        t = gb_model.addVar(vtype=gb.GRB.CONTINUOUS, lb=0.0, ub=1.0, name='t')
                        q = gb_model.addVars(len(list_of_item), vtype=gb.GRB.CONTINUOUS, name='q', lb=-gb.GRB.INFINITY)
                        s = gb_model.addVars(l, vtype=gb.GRB.BINARY, name='s')
                        gb_model.update()
                        gb_model.write('./Lp_files/model.lp')
                        # v decomposition:
        
                        # b1[l] = q[l1] > q[l2] + t ? 1 : 0
                        # b2[l] = q[l1] < q[l2] - t ? 1 : 0
                        # v[l] = b1[l] - b2[l]
        
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
        
                        # if q[l1] > q[l2] + t then b1[l] = 1 and b2[l] = 0, so v[l] = 1 - 0 = 1
                        # if q[l1] < q[l2] - t then b1[l] = 0 and b2[l] = 1, so v[l] = 0 - 1 = -1
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
                            if j > 0:
                                for k in range(0, j):
                                    gb_model.addConstr(z[j] >= z[k])      # (2)
                        gb_model.addConstr(z[0] >= 0)
                        gb_model.update()
                        gb_model.write('./Lp_files/model.lp')
                        for i in range(len(list_of_item)):
                            gb_model.addConstr(q[i] == gb.quicksum(X[i, j] * z[j] for j in range(max_p)))     # (3)
                        gb_model.update()
                        gb_model.write('./Lp_files/model.lp')
                        couple_counter = 0
                        eps = 0.01
                        M = 100
                        for l1, recipe1 in enumerate(recipes_indexes):
                            for l2, recipe2 in enumerate(recipes_indexes):
                                if COUPLE == 150:
                                    if [recipe1, recipe2] in couple_indexes.tolist():
                                        gb_model.addConstr(q[l1] >= q[l2] + t + eps - M * (1 - z1[couple_counter]))
                                        gb_model.addConstr(q[l1] <= q[l2] + t + M * z1[couple_counter])
                                        gb_model.addConstr(q[l1] <= q[l2] - t - eps + M * (1 - z2[couple_counter]))   # - eps
                                        gb_model.addConstr(q[l1] >= q[l2] - t - M * z2[couple_counter])
                                        gb_model.addConstr(z1[couple_counter] + z2[couple_counter] <= 1)
                                        gb_model.addConstr(b1[couple_counter] == z1[couple_counter])
                                        gb_model.addConstr(b2[couple_counter] == z2[couple_counter])
                                        couple_counter += 1
                                else:
                                    if l2 <= l1:
                                        continue
                                    gb_model.addConstr(q[l1] >= q[l2] + t + eps - M * (1 - z1[couple_counter]))
                                    gb_model.addConstr(q[l1] <= q[l2] + t + M * z1[couple_counter])
                                    gb_model.addConstr(q[l1] <= q[l2] - t - eps + M * (1 - z2[couple_counter]))   # - eps
                                    gb_model.addConstr(q[l1] >= q[l2] - t - M * z2[couple_counter])
                                    gb_model.addConstr(z1[couple_counter] + z2[couple_counter] <= 1)
                                    gb_model.addConstr(b1[couple_counter] == z1[couple_counter])
                                    gb_model.addConstr(b2[couple_counter] == z2[couple_counter])
                                    couple_counter += 1
                        gb_model.update()
                        gb_model.write('./Lp_files/model.lp')
                        gb_model.addConstrs(v[couple_number] == b1[couple_number] - b2[couple_number] for couple_number in range(l))     # (4)
                        gb_model.update()
                        gb_model.write('./Lp_files/model.lp')
        
                        couple_counter = 0
                        for l1, recipe1 in enumerate(recipes_indexes):
                            for l2, recipe2 in enumerate(recipes_indexes):
                                if COUPLE == 150:
                                    if [recipe1, recipe2] in couple_indexes.tolist():
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
        results.to_csv("./PCAexperiment/parameter_research_files/optResultsSearch" + scope + str(PCAindex) + ".csv", sep=";", index=False)
        del results
        results = pd.DataFrame(columns=("USER_ID", "MAX_V", "MAX_P", "DATASET_SIZE", "OPT", "THRESHOLD", "F1", "F2", "F3", "F4", "F5"))

