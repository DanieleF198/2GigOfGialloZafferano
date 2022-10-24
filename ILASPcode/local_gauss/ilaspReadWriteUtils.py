import subprocess
from CompareStableModels import compare
import re

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

def preferencesFromFileSpacesAndSign(path):

    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,\s*([a-zA-Z]+\d+)\s*,\s+([a-zA-Z]+\d+),\s+([<>=]).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1], match.groups()[2]))

    return preferences

def preferencesFromFileSpaces(path):

    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*,\s*([a-zA-Z]+\d+)\s*,\s+([a-zA-Z]+\d+).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1]))

    return preferences

def preferencesFromFileSpacesAndSignAndNumber(path):
    preferences = []
    with open(path) as f:
        for line in f:
            match = re.match('#brave_ordering\(.*@(\d+),\s*([a-zA-Z]+\d+)\s*,\s+([a-zA-Z]+\d+),\s+([<>=]).*', line)
            if match: preferences.append((match.groups()[0], match.groups()[1], match.groups()[2], match.groups()[3]))

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

def preferencesToBraveOrderingsSignAndNumber(preferences):
    output = ""
    i = 0

    for preference in preferences:
        output = output + "#brave_ordering(o{}@{},{},{},{}).\n".format(i, preferences[i][0],preferences[i][1], preferences[i][2], preferences[i][3])
        i = i+1

    return output

# ------- TRAIN AND TEST: ------- 
def train(theory_path,options=""):
    ilasp = subprocess.run("ILASP --version=4 {} --quiet {}".format(options,theory_path), stdout=subprocess.PIPE, shell=True)
    ilasp_output = ilasp.stdout.decode("utf-8")
    return ilasp_output

def test(theory,items,test_set):
    correct_instances = 0
    uncertain_instances = 0

    for preference in test_set:
        c = compare(items[preference[0]], items[preference[1]], theory)
        if (c==1):
            correct_instances = correct_instances+1
        if (c==0):
            uncertain_instances = uncertain_instances+1

    if (uncertain_instances==len(test_set)): uncertain_instances = uncertain_instances+1

    return { 'correct': correct_instances, 'uncertain': uncertain_instances, 'incorrect': len(test_set)-uncertain_instances-correct_instances }
