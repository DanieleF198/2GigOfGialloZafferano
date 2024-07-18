import ILASPparser
import numpy as np


def similarity_scores(t1, t2):
    wc1_list = ILASPparser.weak_constraint_in_theory(t1)
    wc2_list = ILASPparser.weak_constraint_in_theory(t2)
    intersection = 0
    p_intersection = 0
    for wc1 in wc1_list:
        literal_found = False
        literal_found_same_level = False
        for wc2 in wc2_list:
            if literal_found and literal_found_same_level:
                break
            literals1 = wc1.get_literals()
            literals2 = wc2.get_literals()
            for literal1 in literals1:
                for literal2 in literals2:
                    if literal1 == literal2:
                        if wc1.get_priority() == wc2.get_priority():
                            if literal_found and (not literal_found_same_level):
                                p_intersection += 1
                                literal_found_same_level = True
                            elif not literal_found:
                                intersection += 1
                                p_intersection += 1
                                literal_found = True
                                literal_found_same_level = True
                        elif not literal_found:
                            intersection += 1
                            literal_found = True
    size1 = 0
    size2 = 0
    for wc1 in wc1_list:
        literals1 = wc1.get_literals()
        for literal1 in literals1:
            size1 += 1
    for wc2 in wc2_list:
        literals2 = wc2.get_literals()
        for literal2 in literals2:
            size2 += 1
    union = (size1 + size2) - intersection
    j = intersection/union
    s = p_intersection/union
    if s > j:
        print("Yoh")
    return j, s


users = [3, 4, 7, 11, 14, 15, 20, 29, 32, 36]
approaches = ["noPCA", "indirect", "direct"]
indirect_approaches_modes = ["Data8Component2Std", "Data17Component2Std"]
direct_approaches_modes = [5, 10, 15, 20]
approximation_modes = ["global", "local"]
sizes = [45, 105, 210]
stds = [1, 0.1, 0.01, 0.001]

global_noPCAvs8PC2STD = []
global_noPCAvs17PC2STD = []
global_noPCAvs5PC = []
global_noPCAvs10PC = []
global_noPCAvs15PC = []
global_noPCAvs20PC = []
global_8PC2STDvsNoPCA = []
global_8PC2STDvs17PC2STD = []
global_8PC2STDvs5PC = []
global_8PC2STDvs10PC = []
global_8PC2STDvs15PC = []
global_8PC2STDvs20PC = []
global_17PC2STDvsNoPCA = []
global_17PC2STDvs8PC2STD = []
global_17PC2STDvs5PC = []
global_17PC2STDvs10PC = []
global_17PC2STDvs15PC = []
global_17PC2STDvs20PC = []
global_5PCvsNoPCA = []
global_5PCvs8PC2STD = []
global_5PCvs17PC2STD = []
global_5PCvs10PC = []
global_5PCvs15PC = []
global_5PCvs20PC = []
global_10PCvsNoPCA = []
global_10PCvs8PC2STD = []
global_10PCvs17PC2STD = []
global_10PCvs5PC = []
global_10PCvs15PC = []
global_10PCvs20PC = []
global_15PCvsNoPCA = []
global_15PCvs8PC2STD = []
global_15PCvs17PC2STD = []
global_15PCvs5PC = []
global_15PCvs10PC = []
global_15PCvs20PC = []
global_20PCvsNoPCA = []
global_20PCvs8PC2STD = []
global_20PCvs17PC2STD = []
global_20PCvs5PC = []
global_20PCvs10PC = []
global_20PCvs15PC = []

for user in users:
    for approximation_mode in approximation_modes:
        if approximation_mode == "global":
            for size in sizes:
                for approach1 in approaches:
                    for approach2 in approaches:
                        if approach1 == "noPCA" and approach2 == "noPCA":
                            continue
                        if approach1 == "noPCA":
                            filename_theory = './ILASPcode/Data/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                            f_local_output = open(filename_theory)
                            local_output = f_local_output.read()
                            f_local_output.close()
                            theory1 = ""
                            for line in local_output.split("\n"):
                                if ':~' not in line:
                                    continue
                                else:
                                    theory1 += line
                            if approach2 == "indirect":
                                for indirect_approaches_mode2 in indirect_approaches_modes:
                                    filename_theory = './ILASPcode/' + indirect_approaches_mode2 + '/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                    f_local_output = open(filename_theory)
                                    local_output = f_local_output.read()
                                    f_local_output.close()
                                    theory2 = ""
                                    for line in local_output.split("\n"):
                                        if ':~' not in line:
                                            continue
                                        else:
                                            theory2 += line
                                    j_score, s_score = similarity_scores(theory1, theory2)
                                    if indirect_approaches_mode2 == "Data8Component2Std":
                                        global_noPCAvs8PC2STD.append([j_score, s_score])
                                    else:
                                        global_noPCAvs17PC2STD.append([j_score, s_score])
                            else:
                                for direct_approaches_mode2 in direct_approaches_modes:
                                    filename_theory = './ILASPcode/PCAexperiment/final' + str(direct_approaches_mode2) + '/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                    f_local_output = open(filename_theory)
                                    local_output = f_local_output.read()
                                    f_local_output.close()
                                    theory2 = ""
                                    for line in local_output.split("\n"):
                                        if ':~' not in line:
                                            continue
                                        else:
                                            theory2 += line
                                    j_score, s_score = similarity_scores(theory1, theory2)
                                    if direct_approaches_mode2 == 5:
                                        global_noPCAvs5PC.append([j_score, s_score])
                                    elif direct_approaches_mode2 == 10:
                                        global_noPCAvs10PC.append([j_score, s_score])
                                    elif direct_approaches_mode2 == 15:
                                        global_noPCAvs15PC.append([j_score, s_score])
                                    else:
                                        global_noPCAvs20PC.append([j_score, s_score])
                        elif approach1 == "indirect":
                            for indirect_approaches_mode1 in indirect_approaches_modes:
                                filename_theory = './ILASPcode/' + indirect_approaches_mode1 + '/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                f_local_output = open(filename_theory)
                                local_output = f_local_output.read()
                                f_local_output.close()
                                theory1 = ""
                                for line in local_output.split("\n"):
                                    if ':~' not in line:
                                        continue
                                    else:
                                        theory1 += line
                                if approach2 == "noPCA":
                                    filename_theory = './ILASPcode/Data/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                    f_local_output = open(filename_theory)
                                    local_output = f_local_output.read()
                                    f_local_output.close()
                                    theory2 = ""
                                    for line in local_output.split("\n"):
                                        if ':~' not in line:
                                            continue
                                        else:
                                            theory2 += line
                                    j_score, s_score = similarity_scores(theory1, theory2)
                                    if indirect_approaches_mode1 == "Data8Component2Std":
                                        global_8PC2STDvsNoPCA.append([j_score, s_score])
                                    else:
                                        global_17PC2STDvsNoPCA.append([j_score, s_score])
                                if approach2 == "indirect":
                                    for indirect_approaches_mode2 in indirect_approaches_modes:
                                        if indirect_approaches_mode1 == indirect_approaches_mode2:
                                            continue
                                        filename_theory = './ILASPcode/' + indirect_approaches_mode2 + '/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                        f_local_output = open(filename_theory)
                                        local_output = f_local_output.read()
                                        f_local_output.close()
                                        theory2 = ""
                                        for line in local_output.split("\n"):
                                            if ':~' not in line:
                                                continue
                                            else:
                                                theory2 += line
                                        j_score, s_score = similarity_scores(theory1, theory2)
                                        if indirect_approaches_mode1 == "Data8Component2Std":
                                            global_8PC2STDvs17PC2STD.append([j_score, s_score])
                                        else:
                                            global_17PC2STDvs8PC2STD.append([j_score, s_score])
                                else:
                                    for direct_approaches_mode2 in direct_approaches_modes:
                                        filename_theory = './ILASPcode/PCAexperiment/final' + str(direct_approaches_mode2) + '/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                        f_local_output = open(filename_theory)
                                        local_output = f_local_output.read()
                                        f_local_output.close()
                                        theory2 = ""
                                        for line in local_output.split("\n"):
                                            if ':~' not in line:
                                                continue
                                            else:
                                                theory2 += line
                                        j_score, s_score = similarity_scores(theory1, theory2)
                                        if indirect_approaches_mode1 == "Data8Component2Std":
                                            if direct_approaches_mode2 == 5:
                                                global_8PC2STDvs5PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 10:
                                                global_8PC2STDvs10PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 15:
                                                global_8PC2STDvs15PC.append([j_score, s_score])
                                            else:
                                                global_8PC2STDvs20PC.append([j_score, s_score])
                                        else:
                                            if direct_approaches_mode2 == 5:
                                                global_17PC2STDvs5PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 10:
                                                global_17PC2STDvs10PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 15:
                                                global_17PC2STDvs15PC.append([j_score, s_score])
                                            else:
                                                global_17PC2STDvs20PC.append([j_score, s_score])
                        else:
                            for direct_approaches_mode1 in direct_approaches_modes:
                                filename_theory = './ILASPcode/PCAexperiment/final' + str(direct_approaches_mode1) + '/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                f_local_output = open(filename_theory)
                                local_output = f_local_output.read()
                                f_local_output.close()
                                theory1 = ""
                                for line in local_output.split("\n"):
                                    if ':~' not in line:
                                        continue
                                    else:
                                        theory1 += line
                                if approach2 == "noPCA":
                                    filename_theory = './ILASPcode/Data/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                    f_local_output = open(filename_theory)
                                    local_output = f_local_output.read()
                                    f_local_output.close()
                                    theory2 = ""
                                    for line in local_output.split("\n"):
                                        if ':~' not in line:
                                            continue
                                        else:
                                            theory2 += line
                                    j_score, s_score = similarity_scores(theory1, theory2)
                                    if direct_approaches_mode1 == 5:
                                        global_5PCvsNoPCA.append([j_score, s_score])
                                    elif direct_approaches_mode1 == 10:
                                        global_10PCvsNoPCA.append([j_score, s_score])
                                    elif direct_approaches_mode1 == 15:
                                        global_15PCvsNoPCA.append([j_score, s_score])
                                    elif direct_approaches_mode1 == 20:
                                        global_20PCvsNoPCA.append([j_score, s_score])
                                if approach2 == "indirect":
                                    for indirect_approaches_mode2 in indirect_approaches_modes:
                                        filename_theory = './ILASPcode/' + indirect_approaches_mode2 + '/final/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                        f_local_output = open(filename_theory)
                                        local_output = f_local_output.read()
                                        f_local_output.close()
                                        theory2 = ""
                                        for line in local_output.split("\n"):
                                            if ':~' not in line:
                                                continue
                                            else:
                                                theory2 += line
                                        j_score, s_score = similarity_scores(theory1, theory2)
                                        if direct_approaches_mode1 == 5:
                                            if indirect_approaches_mode2 == "Data8Component2Std":
                                                global_5PCvs8PC2STD.append([j_score, s_score])
                                            else:
                                                global_5PCvs17PC2STD.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 10:
                                            if indirect_approaches_mode2 == "Data8Component2Std":
                                                global_10PCvs8PC2STD.append([j_score, s_score])
                                            else:
                                                global_10PCvs17PC2STD.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 15:
                                            if indirect_approaches_mode2 == "Data8Component2Std":
                                                global_15PCvs8PC2STD.append([j_score, s_score])
                                            else:
                                                global_15PCvs17PC2STD.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 20:
                                            if indirect_approaches_mode2 == "Data8Component2Std":
                                                global_20PCvs8PC2STD.append([j_score, s_score])
                                            else:
                                                global_20PCvs17PC2STD.append([j_score, s_score])
                                else:
                                    for direct_approaches_mode2 in direct_approaches_modes:
                                        if direct_approaches_mode1 == direct_approaches_mode2:
                                            continue
                                        filename_theory = './ILASPcode/PCAexperiment/final' + str(direct_approaches_mode2) + '/users/zero/train/' + str(size) + 'Couples/User' + str(user) + '/outputTrain/outputTrain_max-v(5)-max_p(5).txt'
                                        f_local_output = open(filename_theory)
                                        local_output = f_local_output.read()
                                        f_local_output.close()
                                        theory2 = ""
                                        for line in local_output.split("\n"):
                                            if ':~' not in line:
                                                continue
                                            else:
                                                theory2 += line
                                        j_score, s_score = similarity_scores(theory1, theory2)
                                        if direct_approaches_mode1 == 5:
                                            if direct_approaches_mode2 == 10:
                                                global_5PCvs10PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 15:
                                                global_5PCvs15PC.append([j_score, s_score])
                                            else:
                                                global_5PCvs20PC.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 10:
                                            if direct_approaches_mode2 == 5:
                                                global_10PCvs5PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 15:
                                                global_10PCvs15PC.append([j_score, s_score])
                                            else:
                                                global_10PCvs20PC.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 15:
                                            if direct_approaches_mode2 == 5:
                                                global_15PCvs5PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 10:
                                                global_15PCvs10PC.append([j_score, s_score])
                                            else:
                                                global_15PCvs20PC.append([j_score, s_score])
                                        elif direct_approaches_mode1 == 20:
                                            if direct_approaches_mode2 == 5:
                                                global_20PCvs5PC.append([j_score, s_score])
                                            elif direct_approaches_mode2 == 10:
                                                global_20PCvs10PC.append([j_score, s_score])
                                            else:
                                                global_20PCvs15PC.append([j_score, s_score])


np_global_noPCAvs8PC2STD = np.array(global_noPCAvs8PC2STD)
np_global_noPCAvs17PC2STD = np.array(global_noPCAvs17PC2STD)
np_global_noPCAvs5PC = np.array(global_noPCAvs5PC)
np_global_noPCAvs10PC = np.array(global_noPCAvs10PC)
np_global_noPCAvs15PC = np.array(global_noPCAvs15PC)
np_global_noPCAvs20PC = np.array(global_noPCAvs20PC)
np_global_8PC2STDvsNoPCA = np.array(global_8PC2STDvsNoPCA)
np_global_8PC2STDvs17PC2STD = np.array(global_8PC2STDvs17PC2STD)
np_global_8PC2STDvs5PC = np.array(global_8PC2STDvs5PC)
np_global_8PC2STDvs10PC = np.array(global_8PC2STDvs10PC)
np_global_8PC2STDvs15PC = np.array(global_8PC2STDvs15PC)
np_global_8PC2STDvs20PC = np.array(global_8PC2STDvs20PC)
np_global_17PC2STDvsNoPCA = np.array(global_17PC2STDvsNoPCA)
np_global_17PC2STDvs8PC2STD = np.array(global_17PC2STDvs8PC2STD)
np_global_17PC2STDvs5PC = np.array(global_17PC2STDvs5PC)
np_global_17PC2STDvs10PC = np.array(global_17PC2STDvs10PC)
np_global_17PC2STDvs15PC = np.array(global_17PC2STDvs15PC)
np_global_17PC2STDvs20PC = np.array(global_17PC2STDvs20PC)
np_global_5PCvsNoPCA = np.array(global_5PCvsNoPCA)
np_global_5PCvs8PC2STD = np.array(global_5PCvs8PC2STD)
np_global_5PCvs17PC2STD = np.array(global_5PCvs17PC2STD)
np_global_5PCvs10PC = np.array(global_5PCvs10PC)
np_global_5PCvs15PC = np.array(global_5PCvs15PC)
np_global_5PCvs20PC = np.array(global_5PCvs20PC)
np_global_10PCvsNoPCA = np.array(global_10PCvsNoPCA)
np_global_10PCvs8PC2STD = np.array(global_10PCvs8PC2STD)
np_global_10PCvs17PC2STD = np.array(global_10PCvs17PC2STD)
np_global_10PCvs5PC = np.array(global_10PCvs5PC)
np_global_10PCvs15PC = np.array(global_10PCvs15PC)
np_global_10PCvs20PC = np.array(global_10PCvs20PC)
np_global_15PCvsNoPCA = np.array(global_15PCvsNoPCA)
np_global_15PCvs8PC2STD = np.array(global_15PCvs8PC2STD)
np_global_15PCvs17PC2STD = np.array(global_15PCvs17PC2STD)
np_global_15PCvs5PC = np.array(global_15PCvs5PC)
np_global_15PCvs10PC = np.array(global_15PCvs10PC)
np_global_15PCvs20PC = np.array(global_15PCvs20PC)
np_global_20PCvsNoPCA = np.array(global_20PCvsNoPCA)
np_global_20PCvs8PC2STD = np.array(global_20PCvs8PC2STD)
np_global_20PCvs17PC2STD = np.array(global_20PCvs17PC2STD)
np_global_20PCvs5PC = np.array(global_20PCvs5PC)
np_global_20PCvs10PC = np.array(global_20PCvs10PC)
np_global_20PCvs15PC = np.array(global_20PCvs15PC)

print("size 45")
print("[NoPCA vs 8PC2STD] - j = " + str(np.mean(np_global_noPCAvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs8PC2STD[0:10], axis=0)[1]))
print("[NoPCA vs 17PC2STD] - j = " + str(np.mean(np_global_noPCAvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs17PC2STD[0:10], axis=0)[1]))
print("[NoPCA vs 5PC] - j = " + str(np.mean(np_global_noPCAvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs5PC[0:10], axis=0)[1]))
print("[NoPCA vs 10PC] - j = " + str(np.mean(np_global_noPCAvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs10PC[0:10], axis=0)[1]))
print("[NoPCA vs 15PC] - j = " + str(np.mean(np_global_noPCAvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs15PC[0:10], axis=0)[1]))
print("[NoPCA vs 20PC] - j = " + str(np.mean(np_global_noPCAvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs20PC[0:10], axis=0)[1]))

print("[8PC2STD vs NoPCA] - j = " + str(np.mean(np_global_8PC2STDvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvsNoPCA[0:10], axis=0)[1]))
print("[8PC2STD vs 17PC2STD] - j = " + str(np.mean(np_global_8PC2STDvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs17PC2STD[0:10], axis=0)[1]))
print("[8PC2STD vs 5PC] - j = " + str(np.mean(np_global_8PC2STDvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs5PC[0:10], axis=0)[1]))
print("[8PC2STD vs 10PC] - j = " + str(np.mean(np_global_8PC2STDvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs10PC[0:10], axis=0)[1]))
print("[8PC2STD vs 15PC] - j = " + str(np.mean(np_global_8PC2STDvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs15PC[0:10], axis=0)[1]))
print("[8PC2STD vs 20PC] - j = " + str(np.mean(np_global_8PC2STDvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs20PC[0:10], axis=0)[1]))

print("[17PC2STD vs NoPCA] - j = " + str(np.mean(np_global_17PC2STDvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvsNoPCA[0:10], axis=0)[1]))
print("[17PC2STD vs 8PC2STD] - j = " + str(np.mean(np_global_17PC2STDvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs8PC2STD[0:10], axis=0)[1]))
print("[17PC2STD vs 5PC] - j = " + str(np.mean(np_global_17PC2STDvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs5PC[0:10], axis=0)[1]))
print("[17PC2STD vs 10PC] - j = " + str(np.mean(np_global_17PC2STDvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs10PC[0:10], axis=0)[1]))
print("[17PC2STD vs 15PC] - j = " + str(np.mean(np_global_17PC2STDvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs15PC[0:10], axis=0)[1]))
print("[17PC2STD vs 20PC] - j = " + str(np.mean(np_global_17PC2STDvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs20PC[0:10], axis=0)[1]))

print("[5PC vs NoPCA] - j = " + str(np.mean(np_global_5PCvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvsNoPCA[0:10], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs8PC2STD[0:10], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs17PC2STD[0:10], axis=0)[1]))
print("[5PC vs 10PC] - j = " + str(np.mean(np_global_5PCvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs10PC[0:10], axis=0)[1]))
print("[5PC vs 15PC] - j = " + str(np.mean(np_global_5PCvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs15PC[0:10], axis=0)[1]))
print("[5PC vs 20PC] - j = " + str(np.mean(np_global_5PCvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs20PC[0:10], axis=0)[1]))

print("[10PC vs NoPCA] - j = " + str(np.mean(np_global_10PCvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvsNoPCA[0:10], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs8PC2STD[0:10], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs17PC2STD[0:10], axis=0)[1]))
print("[10PC vs 5PC] - j = " + str(np.mean(np_global_10PCvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs5PC[0:10], axis=0)[1]))
print("[10PC vs 15PC] - j = " + str(np.mean(np_global_10PCvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs15PC[0:10], axis=0)[1]))
print("[10PC vs 20PC] - j = " + str(np.mean(np_global_10PCvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs20PC[0:10], axis=0)[1]))

print("[15PC vs NoPCA] - j = " + str(np.mean(np_global_15PCvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvsNoPCA[0:10], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs8PC2STD[0:10], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs17PC2STD[0:10], axis=0)[1]))
print("[15PC vs 5PC] - j = " + str(np.mean(np_global_15PCvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs5PC[0:10], axis=0)[1]))
print("[15PC vs 10PC] - j = " + str(np.mean(np_global_15PCvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs10PC[0:10], axis=0)[1]))
print("[15PC vs 20PC] - j = " + str(np.mean(np_global_15PCvs20PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs20PC[0:10], axis=0)[1]))

print("[20PC vs NoPCA] - j = " + str(np.mean(np_global_20PCvsNoPCA[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvsNoPCA[0:10], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs8PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs8PC2STD[0:10], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs17PC2STD[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs17PC2STD[0:10], axis=0)[1]))
print("[20PC vs 5PC] - j = " + str(np.mean(np_global_20PCvs5PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs5PC[0:10], axis=0)[1]))
print("[20PC vs 10PC] - j = " + str(np.mean(np_global_20PCvs10PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs10PC[0:10], axis=0)[1]))
print("[20PC vs 15PC] - j = " + str(np.mean(np_global_20PCvs15PC[0:10], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs15PC[0:10], axis=0)[1]))

print("")
print("----------------------------------------------------------")
print("")

print("size 105")
print("[NoPCA vs 8PC2STD] - j = " + str(np.mean(np_global_noPCAvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs8PC2STD[10:20], axis=0)[1]))
print("[NoPCA vs 17PC2STD] - j = " + str(np.mean(np_global_noPCAvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs17PC2STD[10:20], axis=0)[1]))
print("[NoPCA vs 5PC] - j = " + str(np.mean(np_global_noPCAvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs5PC[10:20], axis=0)[1]))
print("[NoPCA vs 10PC] - j = " + str(np.mean(np_global_noPCAvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs10PC[10:20], axis=0)[1]))
print("[NoPCA vs 15PC] - j = " + str(np.mean(np_global_noPCAvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs15PC[10:20], axis=0)[1]))
print("[NoPCA vs 20PC] - j = " + str(np.mean(np_global_noPCAvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs20PC[10:20], axis=0)[1]))

print("[8PC2STD vs NoPCA] - j = " + str(np.mean(np_global_8PC2STDvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvsNoPCA[10:20], axis=0)[1]))
print("[8PC2STD vs 17PC2STD] - j = " + str(np.mean(np_global_8PC2STDvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs17PC2STD[10:20], axis=0)[1]))
print("[8PC2STD vs 5PC] - j = " + str(np.mean(np_global_8PC2STDvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs5PC[10:20], axis=0)[1]))
print("[8PC2STD vs 10PC] - j = " + str(np.mean(np_global_8PC2STDvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs10PC[10:20], axis=0)[1]))
print("[8PC2STD vs 15PC] - j = " + str(np.mean(np_global_8PC2STDvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs15PC[10:20], axis=0)[1]))
print("[8PC2STD vs 20PC] - j = " + str(np.mean(np_global_8PC2STDvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs20PC[10:20], axis=0)[1]))

print("[17PC2STD vs NoPCA] - j = " + str(np.mean(np_global_17PC2STDvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvsNoPCA[10:20], axis=0)[1]))
print("[17PC2STD vs 8PC2STD] - j = " + str(np.mean(np_global_17PC2STDvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs8PC2STD[10:20], axis=0)[1]))
print("[17PC2STD vs 5PC] - j = " + str(np.mean(np_global_17PC2STDvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs5PC[10:20], axis=0)[1]))
print("[17PC2STD vs 10PC] - j = " + str(np.mean(np_global_17PC2STDvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs10PC[10:20], axis=0)[1]))
print("[17PC2STD vs 15PC] - j = " + str(np.mean(np_global_17PC2STDvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs15PC[10:20], axis=0)[1]))
print("[17PC2STD vs 20PC] - j = " + str(np.mean(np_global_17PC2STDvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs20PC[10:20], axis=0)[1]))

print("[5PC vs NoPCA] - j = " + str(np.mean(np_global_5PCvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvsNoPCA[10:20], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs8PC2STD[10:20], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs17PC2STD[10:20], axis=0)[1]))
print("[5PC vs 10PC] - j = " + str(np.mean(np_global_5PCvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs10PC[10:20], axis=0)[1]))
print("[5PC vs 15PC] - j = " + str(np.mean(np_global_5PCvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs15PC[10:20], axis=0)[1]))
print("[5PC vs 20PC] - j = " + str(np.mean(np_global_5PCvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs20PC[10:20], axis=0)[1]))

print("[10PC vs NoPCA] - j = " + str(np.mean(np_global_10PCvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvsNoPCA[10:20], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs8PC2STD[10:20], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs17PC2STD[10:20], axis=0)[1]))
print("[10PC vs 5PC] - j = " + str(np.mean(np_global_10PCvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs5PC[10:20], axis=0)[1]))
print("[10PC vs 15PC] - j = " + str(np.mean(np_global_10PCvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs15PC[10:20], axis=0)[1]))
print("[10PC vs 20PC] - j = " + str(np.mean(np_global_10PCvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs20PC[10:20], axis=0)[1]))

print("[15PC vs NoPCA] - j = " + str(np.mean(np_global_15PCvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvsNoPCA[10:20], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs8PC2STD[10:20], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs17PC2STD[10:20], axis=0)[1]))
print("[15PC vs 5PC] - j = " + str(np.mean(np_global_15PCvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs5PC[10:20], axis=0)[1]))
print("[15PC vs 10PC] - j = " + str(np.mean(np_global_15PCvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs10PC[10:20], axis=0)[1]))
print("[15PC vs 20PC] - j = " + str(np.mean(np_global_15PCvs20PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs20PC[10:20], axis=0)[1]))

print("[20PC vs NoPCA] - j = " + str(np.mean(np_global_20PCvsNoPCA[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvsNoPCA[10:20], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs8PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs8PC2STD[10:20], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs17PC2STD[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs17PC2STD[10:20], axis=0)[1]))
print("[20PC vs 5PC] - j = " + str(np.mean(np_global_20PCvs5PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs5PC[10:20], axis=0)[1]))
print("[20PC vs 10PC] - j = " + str(np.mean(np_global_20PCvs10PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs10PC[10:20], axis=0)[1]))
print("[20PC vs 15PC] - j = " + str(np.mean(np_global_20PCvs15PC[10:20], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs15PC[10:20], axis=0)[1]))

print("")
print("----------------------------------------------------------")
print("")

print("size 210")
print("[NoPCA vs 8PC2STD] - j = " + str(np.mean(np_global_noPCAvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs8PC2STD[20:], axis=0)[1]))
print("[NoPCA vs 17PC2STD] - j = " + str(np.mean(np_global_noPCAvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs17PC2STD[20:], axis=0)[1]))
print("[NoPCA vs 5PC] - j = " + str(np.mean(np_global_noPCAvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs5PC[20:], axis=0)[1]))
print("[NoPCA vs 10PC] - j = " + str(np.mean(np_global_noPCAvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs10PC[20:], axis=0)[1]))
print("[NoPCA vs 15PC] - j = " + str(np.mean(np_global_noPCAvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs15PC[20:], axis=0)[1]))
print("[NoPCA vs 20PC] - j = " + str(np.mean(np_global_noPCAvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_noPCAvs20PC[20:], axis=0)[1]))

print("[8PC2STD vs NoPCA] - j = " + str(np.mean(np_global_8PC2STDvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvsNoPCA[20:], axis=0)[1]))
print("[8PC2STD vs 17PC2STD] - j = " + str(np.mean(np_global_8PC2STDvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs17PC2STD[20:], axis=0)[1]))
print("[8PC2STD vs 5PC] - j = " + str(np.mean(np_global_8PC2STDvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs5PC[20:], axis=0)[1]))
print("[8PC2STD vs 10PC] - j = " + str(np.mean(np_global_8PC2STDvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs10PC[20:], axis=0)[1]))
print("[8PC2STD vs 15PC] - j = " + str(np.mean(np_global_8PC2STDvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs15PC[20:], axis=0)[1]))
print("[8PC2STD vs 20PC] - j = " + str(np.mean(np_global_8PC2STDvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_8PC2STDvs20PC[20:], axis=0)[1]))

print("[17PC2STD vs NoPCA] - j = " + str(np.mean(np_global_17PC2STDvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvsNoPCA[20:], axis=0)[1]))
print("[17PC2STD vs 8PC2STD] - j = " + str(np.mean(np_global_17PC2STDvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs8PC2STD[20:], axis=0)[1]))
print("[17PC2STD vs 5PC] - j = " + str(np.mean(np_global_17PC2STDvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs5PC[20:], axis=0)[1]))
print("[17PC2STD vs 10PC] - j = " + str(np.mean(np_global_17PC2STDvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs10PC[20:], axis=0)[1]))
print("[17PC2STD vs 15PC] - j = " + str(np.mean(np_global_17PC2STDvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs15PC[20:], axis=0)[1]))
print("[17PC2STD vs 20PC] - j = " + str(np.mean(np_global_17PC2STDvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_17PC2STDvs20PC[20:], axis=0)[1]))

print("[5PC vs NoPCA] - j = " + str(np.mean(np_global_5PCvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvsNoPCA[20:], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs8PC2STD[20:], axis=0)[1]))
print("[5PC vs 8PC2STD] - j = " + str(np.mean(np_global_5PCvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs17PC2STD[20:], axis=0)[1]))
print("[5PC vs 10PC] - j = " + str(np.mean(np_global_5PCvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs10PC[20:], axis=0)[1]))
print("[5PC vs 15PC] - j = " + str(np.mean(np_global_5PCvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs15PC[20:], axis=0)[1]))
print("[5PC vs 20PC] - j = " + str(np.mean(np_global_5PCvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_5PCvs20PC[20:], axis=0)[1]))

print("[10PC vs NoPCA] - j = " + str(np.mean(np_global_10PCvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvsNoPCA[20:], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs8PC2STD[20:], axis=0)[1]))
print("[10PC vs 8PC2STD] - j = " + str(np.mean(np_global_10PCvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs17PC2STD[20:], axis=0)[1]))
print("[10PC vs 5PC] - j = " + str(np.mean(np_global_10PCvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs5PC[20:], axis=0)[1]))
print("[10PC vs 15PC] - j = " + str(np.mean(np_global_10PCvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs15PC[20:], axis=0)[1]))
print("[10PC vs 20PC] - j = " + str(np.mean(np_global_10PCvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_10PCvs20PC[20:], axis=0)[1]))

print("[15PC vs NoPCA] - j = " + str(np.mean(np_global_15PCvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvsNoPCA[20:], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs8PC2STD[20:], axis=0)[1]))
print("[15PC vs 8PC2STD] - j = " + str(np.mean(np_global_15PCvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs17PC2STD[20:], axis=0)[1]))
print("[15PC vs 5PC] - j = " + str(np.mean(np_global_15PCvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs5PC[20:], axis=0)[1]))
print("[15PC vs 10PC] - j = " + str(np.mean(np_global_15PCvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs10PC[20:], axis=0)[1]))
print("[15PC vs 20PC] - j = " + str(np.mean(np_global_15PCvs20PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_15PCvs20PC[20:], axis=0)[1]))

print("[20PC vs NoPCA] - j = " + str(np.mean(np_global_20PCvsNoPCA[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvsNoPCA[20:], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs8PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs8PC2STD[20:], axis=0)[1]))
print("[20PC vs 8PC2STD] - j = " + str(np.mean(np_global_20PCvs17PC2STD[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs17PC2STD[20:], axis=0)[1]))
print("[20PC vs 5PC] - j = " + str(np.mean(np_global_20PCvs5PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs5PC[20:], axis=0)[1]))
print("[20PC vs 10PC] - j = " + str(np.mean(np_global_20PCvs10PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs10PC[20:], axis=0)[1]))
print("[20PC vs 15PC] - j = " + str(np.mean(np_global_20PCvs15PC[20:], axis=0)[0]) + "; s = " + str(np.mean(np_global_20PCvs15PC[20:], axis=0)[1]))