import subprocess
from CompareStableModels import compare
from CompareStableModels import compare_case_no_zero
from CompareStableModels import compare_cm
from CompareStableModels import compare_cm_grid
from CompareStableModels import compare_cm_grid_2
import re
import numpy as np

# ------- READ FROM FILES: ------- 
def languageBiasFromFile(path):
    output = ""
    
    with open(path) as f:
        for line in f:
            match = re.match('(^#modeo.*|^#weight.*|^#constant.*|^#maxv.*|^#maxp.*)',line)
            if match: output = output + match.groups()[0] + "\n"
            
    return output

def itemsFromFile(path):
    item = {}
    with open(path) as f:
        for line in f:
            match = re.match('#pos\(\s*([a-zA-Z]+\d+)\s*,.*\{(.*)\}.*',line)
            if match: item[match.groups()[0]] = match.groups()[1]
            
    return item

def preferencesFromFile(path):

    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,([a-zA-Z]+\d+),([a-zA-Z]+\d+).*',line)
            if match: preferences.append((match.groups()[0], match.groups()[1]))

    return preferences

def preferencesFromFileSpaces(path):

    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,\s*([a-zA-Z]+\d+)\s*,\s+([a-zA-Z]+\d+).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1]))

    return preferences

def preferencesFromFileSpacesAndSign(path):
    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,\s+([a-zA-Z]+\d+),\s+([a-zA-Z]+\d+),\s+([<>=]).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1], match.groups()[2]))

    return preferences

def preferencesFromFileSign(path):
    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,([a-zA-Z]+\d+),([a-zA-Z]+\d+),([<>=]).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1], match.groups()[2]))

    return preferences

# ------- CONVERT TO STRING: ------- 
def itemsToPos(items):
    output = ""

    for key in items:
        output = output + "#pos({},{{}},{{}},{{{}}}).\n".format(key,items[key])
        
    return output

def preferencesToBraveOrderings(preferences):
    output = ""
    i = 0

    for preference in preferences:
        output = output + "#brave_ordering(o{}@1,{},{}).\n".format(i,preferences[i][0],preferences[i][1])
        i = i+1

    return output

def preferencesToBraveOrderingsSign(preferences):
    output = ""
    i = 0

    for preference in preferences:
        output = output + "#brave_ordering(o{}@1,{},{},{}).\n".format(i, preferences[i][0],preferences[i][1],preferences[i][2])
        i = i+1

    return output

# ------- TRAIN AND TEST: ------- 
def train(theory_path,options=""):
    ilasp = subprocess.run("ILASP --version=4 {} --quiet {}".format(options,theory_path), stdout=subprocess.PIPE, shell=True)
    ilasp_output = ilasp.stdout.decode("utf-8")
    return ilasp_output

def test_cm_grid(theory, items, test_set, treshold_value, factors_combination):

    confusion_matrix = np.zeros((3, 3), dtype='float32')
    #                     labels
    #            ___>___|___=___|___<___
    #   o p  >  |   5   |   6   |   4      class 1
    #   u u  =  |   8   |   9   |   7      class 0
    #   t t  <  |   2   |   3   |   1      class -1

    for preference in test_set:
        c = compare_cm_grid(items[preference[0]], items[preference[1]], preference[2], theory, treshold_value, factors_combination)
        if c == 5:
            confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
        if c == 6:
            confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
        if c == 4:
            confusion_matrix[0, 2] = confusion_matrix[0, 2] + 1
        if c == 8:
            confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
        if c == 9:
            confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
        if c == 7:
            confusion_matrix[1, 2] = confusion_matrix[1, 2] + 1
        if c == 2:
            confusion_matrix[2, 0] = confusion_matrix[2, 0] + 1
        if c == 3:
            confusion_matrix[2, 1] = confusion_matrix[2, 1] + 1
        if c == 1:
            confusion_matrix[2, 2] = confusion_matrix[2, 2] + 1

    # accuracy of class i = TP(i) + TN(i) / TP(i) + TN(i) + FP(i) + FN(i)
    # precision of class i = TP(i) / TP(i) + FP(i)
    # recall of class i = TP(i) / TP(i) + FN(i)

    # font: https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428


    accuracy_class1 = (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class0 = (confusion_matrix[1, 1] + confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class_minus1 = (confusion_matrix[2, 2] + confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

    if np.isnan(accuracy_class1):
        accuracy_class1 = 0
    if np.isnan(accuracy_class0):
        accuracy_class0 = 0
    if np.isnan(accuracy_class_minus1):
        accuracy_class_minus1 = 0

    precision_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 2])
    precision_class0 = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2])
    precision_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[2, 0] + confusion_matrix[2, 1] + confusion_matrix[2, 2])

    if np.isnan(precision_class1):
        precision_class1 = 0
    if np.isnan(precision_class0):
        precision_class0 = 0
    if np.isnan(precision_class_minus1):
        precision_class_minus1 = 0

    recall_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0] + confusion_matrix[2, 0])
    recall_class0 = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1] + confusion_matrix[2, 1])
    recall_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[0, 2] + confusion_matrix[1, 2] + confusion_matrix[2, 2])

    if np.isnan(recall_class1):
        recall_class1 = 0
    if np.isnan(recall_class0):
        recall_class0 = 0
    if np.isnan(recall_class_minus1):
        recall_class_minus1 = 0


    average_accuracy = (accuracy_class1 + accuracy_class0 + accuracy_class_minus1)/3
    average_precision = (precision_class1 + precision_class0 + precision_class_minus1)/3
    average_recall = (recall_class1 + recall_class0 + recall_class_minus1)/3

    return {'avg_accuracy': average_accuracy, 'avg_precision': average_precision, 'avg_recall': average_recall}


def test_cm_grid_2(theory, items, test_set, treshold_value, factors_combination):

    confusion_matrix = np.zeros((3, 3), dtype='float32')
    #                     labels
    #            ___>___|___=___|___<___
    #   o p  >  |   5   |   6   |   4      class 1
    #   u u  =  |   8   |   9   |   7      class 0
    #   t t  <  |   2   |   3   |   1      class -1

    for preference in test_set:
        c = compare_cm_grid_2(items[preference[0]], items[preference[1]], preference[2], theory, treshold_value, factors_combination)
        if c == 5:
            confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
        if c == 6:
            confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
        if c == 4:
            confusion_matrix[0, 2] = confusion_matrix[0, 2] + 1
        if c == 8:
            confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
        if c == 9:
            confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
        if c == 7:
            confusion_matrix[1, 2] = confusion_matrix[1, 2] + 1
        if c == 2:
            confusion_matrix[2, 0] = confusion_matrix[2, 0] + 1
        if c == 3:
            confusion_matrix[2, 1] = confusion_matrix[2, 1] + 1
        if c == 1:
            confusion_matrix[2, 2] = confusion_matrix[2, 2] + 1

    # accuracy of class i = TP(i) + TN(i) / TP(i) + TN(i) + FP(i) + FN(i)
    # precision of class i = TP(i) / TP(i) + FP(i)
    # recall of class i = TP(i) / TP(i) + FN(i)

    # font: https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428


    accuracy_class1 = (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class0 = (confusion_matrix[1, 1] + confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class_minus1 = (confusion_matrix[2, 2] + confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

    if np.isnan(accuracy_class1):
        accuracy_class1 = 0
    if np.isnan(accuracy_class0):
        accuracy_class0 = 0
    if np.isnan(accuracy_class_minus1):
        accuracy_class_minus1 = 0

    precision_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 2])
    precision_class0 = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2])
    precision_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[2, 0] + confusion_matrix[2, 1] + confusion_matrix[2, 2])

    if np.isnan(precision_class1):
        precision_class1 = 0
    if np.isnan(precision_class0):
        precision_class0 = 0
    if np.isnan(precision_class_minus1):
        precision_class_minus1 = 0

    recall_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0] + confusion_matrix[2, 0])
    recall_class0 = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1] + confusion_matrix[2, 1])
    recall_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[0, 2] + confusion_matrix[1, 2] + confusion_matrix[2, 2])

    if np.isnan(recall_class1):
        recall_class1 = 0
    if np.isnan(recall_class0):
        recall_class0 = 0
    if np.isnan(recall_class_minus1):
        recall_class_minus1 = 0


    average_accuracy = (accuracy_class1 + accuracy_class0 + accuracy_class_minus1)/3
    average_precision = (precision_class1 + precision_class0 + precision_class_minus1)/3
    average_recall = (recall_class1 + recall_class0 + recall_class_minus1)/3

    return {'avg_accuracy': average_accuracy, 'avg_precision': average_precision, 'avg_recall': average_recall}



def test_cm(theory, items, test_set):
    to_print = ""
    confusion_matrix = np.zeros((3, 3), dtype='float32')
    #                     labels
    #            ___>___|___=___|___<___
    #   o p  >  |   5   |   6   |   4      class 1
    #   u u  =  |   8   |   9   |   7      class 0
    #   t t  <  |   2   |   3   |   1      class -1
    # NOTE: label > here is intended as ILASP syntax in brave ordering, and so a > b means that b is preferred over a (cause it activates a greater amount of weights from wc)
    #       similarly a < b means that a is preferred over b.
    for preference in test_set:
        c = compare_cm(items[preference[0]], items[preference[1]], preference[2], theory)
        if c == 5:
            confusion_matrix[0, 0] = confusion_matrix[0, 0] + 1
        if c == 6:
            confusion_matrix[0, 1] = confusion_matrix[0, 1] + 1
        if c == 4:
            confusion_matrix[0, 2] = confusion_matrix[0, 2] + 1
        if c == 8:
            confusion_matrix[1, 0] = confusion_matrix[1, 0] + 1
        if c == 9:
            confusion_matrix[1, 1] = confusion_matrix[1, 1] + 1
        if c == 7:
            confusion_matrix[1, 2] = confusion_matrix[1, 2] + 1
        if c == 2:
            confusion_matrix[2, 0] = confusion_matrix[2, 0] + 1
        if c == 3:
            confusion_matrix[2, 1] = confusion_matrix[2, 1] + 1
        if c == 1:
            confusion_matrix[2, 2] = confusion_matrix[2, 2] + 1

    # accuracy of class i = TP(i) + TN(i) / TP(i) + TN(i) + FP(i) + FN(i)
    # precision of class i = TP(i) / TP(i) + FP(i)
    # recall of class i = TP(i) / TP(i) + FN(i)

    # font: https://medium.com/usf-msds/choosing-the-right-metric-for-evaluating-machine-learning-models-part-2-86d5649a5428


    accuracy_class1 = (confusion_matrix[0, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2] + confusion_matrix[2, 1] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class0 = (confusion_matrix[1, 1] + confusion_matrix[0, 0] + confusion_matrix[0, 2] + confusion_matrix[2, 0] + confusion_matrix[2, 2]) / np.sum(confusion_matrix)
    accuracy_class_minus1 = (confusion_matrix[2, 2] + confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1]) / np.sum(confusion_matrix)

    if np.isnan(accuracy_class1):
        accuracy_class1 = 0
    if np.isnan(accuracy_class0):
        accuracy_class0 = 0
    if np.isnan(accuracy_class_minus1):
        accuracy_class_minus1 = 0

    precision_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[0, 2])
    precision_class0 = confusion_matrix[1, 1] / (confusion_matrix[1, 0] + confusion_matrix[1, 1] + confusion_matrix[1, 2])
    precision_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[2, 0] + confusion_matrix[2, 1] + confusion_matrix[2, 2])

    if np.isnan(precision_class1):
        precision_class1 = 0
    if np.isnan(precision_class0):
        precision_class0 = 0
    if np.isnan(precision_class_minus1):
        precision_class_minus1 = 0

    recall_class1 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0] + confusion_matrix[2, 0])
    recall_class0 = confusion_matrix[1, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1] + confusion_matrix[2, 1])
    recall_class_minus1 = confusion_matrix[2, 2] / (confusion_matrix[0, 2] + confusion_matrix[1, 2] + confusion_matrix[2, 2])

    if np.isnan(recall_class1):
        recall_class1 = 0
    if np.isnan(recall_class0):
        recall_class0 = 0
    if np.isnan(recall_class_minus1):
        recall_class_minus1 = 0


    average_accuracy = (accuracy_class1 + accuracy_class0 + accuracy_class_minus1)/3
    average_precision = (precision_class1 + precision_class0 + precision_class_minus1)/3
    average_recall = (recall_class1 + recall_class0 + recall_class_minus1)/3

    return {'avg_accuracy': average_accuracy, 'avg_precision': average_precision, 'avg_recall': average_recall}


def test(theory,items,test_set):
    correct_instances = 0
    uncertain_instances = 0

    for preference in test_set:
        c = compare(items[preference[0]], items[preference[1]],  preference[2], theory)
        if (c==1):
            correct_instances = correct_instances+1
        if (c==0):
            uncertain_instances = uncertain_instances+1

    if (uncertain_instances==len(test_set)): uncertain_instances = uncertain_instances+1

    return { 'correct': correct_instances, 'uncertain': uncertain_instances, 'incorrect': len(test_set)-uncertain_instances-correct_instances }

def test_case_no_zero(theory,items,test_set):
    correct_instances = 0
    uncertain_instances = 0

    for preference in test_set:
        c = compare_case_no_zero(items[preference[0]], items[preference[1]],  preference[2], theory)
        if (c==1):
            correct_instances = correct_instances+1
        if (c==0):
            uncertain_instances = uncertain_instances+1

    if (uncertain_instances==len(test_set)): uncertain_instances = uncertain_instances+1

    return { 'correct': correct_instances, 'uncertain': uncertain_instances, 'incorrect': len(test_set)-uncertain_instances-correct_instances }
