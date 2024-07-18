import csv
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import sys
import os
import pandas as pd
import numpy as np
import re
from deep_translator import GoogleTranslator

class WeakConstraintClass:
    def __init__(self, wc_literals, wc_weight, wc_priority, wc_terms):
        self.literals = wc_literals
        self.weight = wc_weight
        self.priority = wc_priority
        self.terms = wc_terms

    def __lt__(self, other):
        return self.priority < other.priority

    def get_literals(self):
        return self.literals

    def set_literals(self, l):
        self.literals = l

    def get_weight(self):
        return self.weight

    def set_weight(self, w):
        self.weight = w

    def get_priority(self):
        return self.priority

    def set_priority(self, p):
        self.priority = p

    def get_terms(self):
        return self.terms

    def set_terms(self, t):
        self.terms = t

def retro_projection(WCSObjectList):
    for l, WeakConstraint in enumerate(WCSObjectList):
        if "pc0" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['category(4)', 'value(carne,V1)', 'value(pasta,V1)'])
        if "pc1" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['category(1)', 'value(frittura,V1)'])
        if "pc2" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(frutta,V1)', 'value(dolcificanti,V1)'])
        if "pc3" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(latticini,V1)'])
        if "pc4" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['category(5)', 'value(farinacei,V1)', 'value(pasta,V1)', 'value(forno,V1)'])
        if "pc5" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(verdure_e_ortaggi,V1)', 'value(carne,V1)'])
        if "pc6" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(erbe_spezie_e_condimenti,V1)', 'value(latticini,V1)'])
        if "pc7" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(erbe_spezie_e_condimenti,V1)', 'value(verdure_e_ortaggi,V1)', 'value(cottura_a_vapore,V1)'])
        if "pc8" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(erbe_spezie_e_condimenti,V1)', 'value(verdure_e_ortaggi,V1)', 'value(marinatura,V1)'])
        if "pc9" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(erbe_spezie_e_condimenti,V1)', 'value(verdure_e_ortaggi,V1)'])
        if "pc10" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(difficulty,V1)', 'value(carne,V1)', 'value(cottura_a_vapore,V1)'])
        if "pc11" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(uova,V1)', 'value(frutta,V1)'])
        if "pc12" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(frutta,V1)', 'value(rosolatura,V1)'])
        if "pc13" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(latticini,V1)', 'value(funghi,V1)', 'value(erbe_spezie_e_condimenti,V1)'])
        if "pc14" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(latticini,V1)'])
        if "pc15" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['category(1)', 'value(latticini,V1)', 'value(cottura_a_vapore,V1)'])
        if "pc16" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(latticini,V1)', 'value(pesce,V1)'])
        if "pc17" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(carne,V1)', 'value(pesce,V1)'])
        if "pc18" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(verdure_e_ortaggi,V1)', 'value(latticini,V1)'])
        if "pc19" in str(WeakConstraint.get_literals()):
            WeakConstraint.set_literals(['value(carne,V1)', 'value(funghi,V1)', 'value(marinatura,V1)'])
    return WCSObjectList


def weak_constraint_in_theory(t):
    WeakConstraintObjectList = []
    listWeakConstraint = t.split(':~')
    del listWeakConstraint[0]  # del empty occurrence
    for weakConstraint in listWeakConstraint:
        partsOfWeakConstraint = weakConstraint.split('.')
        literals = list(partsOfWeakConstraint[0].split(', '))  # note: there works if space after commas in literals are only from a literal to the next.
        weight = partsOfWeakConstraint[1].split('@')[0][1:]
        priorityAndTerms = partsOfWeakConstraint[1].split('@')[1]
        priorityAndTerms = priorityAndTerms[0:priorityAndTerms.find(']')]
        if priorityAndTerms.find(',') != -1:
            priority = priorityAndTerms[0:priorityAndTerms.find(',')]
            terms = list(priorityAndTerms[priorityAndTerms.find(',') + 2:])  # note: also there space after commas between a terms and his next is required.
        else:
            priority = priorityAndTerms
            terms = []
        wc = WeakConstraintClass(literals, str(weight), int(priority), terms)
        WeakConstraintObjectList.append(wc)
    WeakConstraintObjectList.sort()
    if "pc" in str(WeakConstraintObjectList[0].get_literals()):
        WeakConstraintObjectList = retro_projection(WeakConstraintObjectList)
    return WeakConstraintObjectList


def printTheory(path, data_collector, data_counter_collector, user = 99, max_v = 99, max_p = 99, couple = 99):
    print('Preferences are written from the one with less priority to the one with high priority')
    print('The conflicts are grouped by the right part of the statement (so wcs x(1), x(2), ..., x(n) which are contradicted by the same wc y) and are written with the same order considered for preferences referred to wc y')
    print('---------------------------------------------------------------------------------------')
    with open(path, newline='') as csvFile:
        answerSet = csv.reader(csvFile, delimiter=';')
        for i, row in enumerate(answerSet):
            if i == 0:  # avoid CSV header
                continue
            else:
                user_to_print = ""
                max_v_to_print = ""
                max_p_to_print = ""
                couple_to_print = ""
                for j, element in enumerate(row):
                    if j != 10:
                        if j == 0:
                            if user != 99:
                                if int(element) != user:
                                    break
                            user_to_print = element
                        if j == 1:
                            if max_v != 99:
                                if int(element) != max_v:
                                    break
                            max_v_to_print = element
                        if j == 2:
                            if max_p != 99:
                                if int(element) != max_p:
                                    break
                            max_p_to_print = element
                        if j == 4:
                            if couple != 99:
                                if couple != 210:
                                    if int(element) != couple:
                                        break
                                else:
                                    if (int(element) != 190) and (int(element) != 210):
                                        break
                            couple_to_print = element
                        continue
                    else:
                        print("User " + user_to_print + "; (max_v, max_p) = (" + str(max_v_to_print) + ", " + str(max_p_to_print + "); Dataset of size " + str(couple_to_print)))
                        WeakConstraintObjectList = weak_constraint_in_theory(element)
                        # ---- start data collection for plots ----

                        for l, WeakConstraint in enumerate(WeakConstraintObjectList):
                            if "category(1)" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["antipasti"] = WeakConstraint.get_priority()
                            if "category(2)" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["piatto_unico"] = WeakConstraint.get_priority()
                            if "category(3)" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["primo"] = WeakConstraint.get_priority()
                            if "category(4)" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["secondo"] = WeakConstraint.get_priority()
                            if "category(5)" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["torta_salata"] = WeakConstraint.get_priority()
                            if "dolcificanti" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["dolcificanti"] = WeakConstraint.get_priority()
                            if "farinacei" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["farinacei"] = WeakConstraint.get_priority()
                            if "erbe_spezie_e_condimenti" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["erbe_spezie_e_condimenti"] = WeakConstraint.get_priority()
                            if "carne" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["carne"] = WeakConstraint.get_priority()
                            if "cereali" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["cereali"] = WeakConstraint.get_priority()
                            if "frutta" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["frutta"] = WeakConstraint.get_priority()
                            if "funghi_e_tartufi" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["funghi_e_tartufi"] = WeakConstraint.get_priority()
                            if "latticini" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["latticini"] = WeakConstraint.get_priority()
                            if "pasta" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["pasta"] = WeakConstraint.get_priority()
                            if "pesce" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["pesce"] = WeakConstraint.get_priority()
                            if "uova" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["uova"] = WeakConstraint.get_priority()
                            if "verdure_e_ortaggi" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["verdure_e_ortaggi"] = WeakConstraint.get_priority()
                            if "bollitura" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["bollitura"] = WeakConstraint.get_priority()
                            if "cottura_a_fiamma" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["cottura_a_fiamma"] = WeakConstraint.get_priority()
                            if "cottura_a_vapore" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["cottura_a_vapore"] = WeakConstraint.get_priority()
                            if "cottura_in_forno" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["cottura_in_forno"] = WeakConstraint.get_priority()
                            if "frittura" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["frittura"] = WeakConstraint.get_priority()
                            if "mantecatura" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["mantecatura"] = WeakConstraint.get_priority()
                            if "marinatura" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["marinatura"] = WeakConstraint.get_priority()
                            if "rosolatura" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["rosolatura"] = WeakConstraint.get_priority()
                            if "stufato" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["stufato"] = WeakConstraint.get_priority()
                            if "difficulty" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["difficolta"] = WeakConstraint.get_priority()
                            if "prepTime" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["tempo_di_preparazione"] = WeakConstraint.get_priority()
                            if "cost" in str(WeakConstraint.get_literals()):
                                data_collector.loc[data_counter_collector+l]["cost"] = WeakConstraint.get_priority()

                        # ---- end data collection for plots ----
                        print('given two recipes:')
                        if len(WeakConstraintObjectList) < 2:
                            numberOfLiterals = len(WeakConstraintObjectList[0].get_literals())  # convert literals and terms from string to list of string!
                            numberOfTerms = len(WeakConstraintObjectList[0].get_terms())
                            if numberOfLiterals > 1:
                                if '-' in WeakConstraintObjectList[0].get_weight():
                                    stringToPrint = "priority 1 - you appreciate more the one with "
                                else:
                                    stringToPrint = "priority 1 - you appreciate less the one with "
                                for k, literal in enumerate(WeakConstraintObjectList[0].get_literals()):
                                    if k != 0:
                                        stringToPrint = stringToPrint + " and with "
                                    if "value" in literal:
                                        wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                        stringToPrint = stringToPrint + " more " + wordOfInterest
                                    else:
                                        stringToPrint = stringToPrint + literal
                                    if k == numberOfLiterals - 1:
                                        stringToPrint = stringToPrint + " with the same level of priority"
                                print(stringToPrint)
                            else:
                                literal = WeakConstraintObjectList[0].get_literals()[0]
                                if "value" in literal:
                                    wordOfInterest = WeakConstraintObjectList[0].get_literals()[0][WeakConstraintObjectList[0].get_literals()[0].find('(') + 1:WeakConstraintObjectList[0].get_literals()[0].find(',')]
                                    if '-' in WeakConstraintObjectList[0].get_weight():
                                        print("priority 1 - you appreciate more the one with more " + wordOfInterest)
                                    else:
                                        print("priority 1 - you appreciate less the one with more " + wordOfInterest)
                                else:
                                    if '-' in WeakConstraintObjectList[0].get_weight():
                                        print("priority 1 - you appreciate more the one with " + WeakConstraintObjectList[0].get_literals()[0])
                                    else:
                                        print("priority 1 - you appreciate less the one with " + WeakConstraintObjectList[0].get_literals()[0])
                        else:   # in this first version the semantic sense for a conflict is determined by the literals: if the one or more literals in the wcs of the candidate conflict are owned by different types of data , then it's a conflict; else it's not.
                            conflicts = np.zeros(((len(WeakConstraintObjectList)*(len(WeakConstraintObjectList))), 3), dtype='int32')  # a conflict it's when the weight of a wc with high priority have a sign which discord with the sign of a wc with lower priority and the literals of two wc are semantically linked (how to define that these two wc are semantically linked?)
                            conflicts_counter = 0
                            for l1, WeakConstraint1 in enumerate(WeakConstraintObjectList):
                                for l2, WeakConstraint2 in enumerate(WeakConstraintObjectList):
                                    if l2 == l1:
                                        continue
                                    if '-' in WeakConstraint1.get_weight() and not ('-' in WeakConstraint2.get_weight()):
                                        proceedCheck = False
                                        literals1 = WeakConstraint1.get_literals()
                                        literals2 = WeakConstraint2.get_literals()
                                        # In this first version of code the type of data are recognized hard-coded because i don't have ILASP code.
                                        for literal1 in literals1:
                                            if proceedCheck:
                                                break
                                            for literal2 in literals2:
                                                if proceedCheck:
                                                    break
                                                if (('difficulty' in literal1) and not ('difficulty' in literal2)) or ((('uova' in literal1) or ('carne' in literal1) or ('pesce' in literal1) or ('latticini' in literal1) or ('verdure_e_ortaggi' in literal1) or ('pasta' in literal1)) and not(('uova' in literal2) or ('carne' in literal2) or ('pesce' in literal2) or ('latticini' in literal2) or ('verdure_e_ortaggi' in literal2) or ('pasta' in literal2))) or ((('cotturainforno' in literal1) or ('frittura' in literal1) or ('cotturaafiamma' in literal1)) and not(('cotturainforno' in literal2) or ('frittura' in literal2) or ('cotturaafiamma' in literal2))):
                                                    proceedCheck = True
                                        alreadyPresent = False
                                        if [l2, l1, 1] in conflicts.tolist():
                                            alreadyPresent = True
                                        if alreadyPresent or not proceedCheck:
                                            continue
                                        else:
                                            conflicts[conflicts_counter] = [l1, l2, 0]  # third value indicates which of the two wc has the negative weight
                                            conflicts_counter = conflicts_counter + 1
                                    if not ('-' in WeakConstraint1.get_weight()) and '-' in WeakConstraint2.get_weight():
                                        proceedCheck = False
                                        literals1 = WeakConstraint1.get_literals()
                                        literals2 = WeakConstraint2.get_literals()
                                        # In this first version of code the type of data are recognized hard-coded because i don't have ILASP code.
                                        for literal1 in literals1:
                                            if proceedCheck:
                                                break
                                            for literal2 in literals2:
                                                if proceedCheck:
                                                    break
                                                if (('difficulty' in literal1) and not ('difficulty' in literal2)) or ((('uova' in literal1) or ('carne' in literal1) or ('pesce' in literal1) or ('latticini' in literal1) or ('verdure_e_ortaggi' in literal1) or ('pasta' in literal1)) and not (('uova' in literal2) or ('carne' in literal2) or ('pesce' in literal2) or ('latticini' in literal2) or ('verdure_e_ortaggi' in literal2) or ('pasta' in literal2))) or ((('cotturainforno' in literal1) or ('frittura' in literal1) or ('cotturaafiamma' in literal1)) and not (('cotturainforno' in literal2) or ('frittura' in literal2) or ('cotturaafiamma' in literal2))):
                                                    proceedCheck = True
                                        alreadyPresent = False
                                        if [l2, l1, 0] in conflicts.tolist():
                                            alreadyPresent = True
                                        if alreadyPresent or not proceedCheck:
                                            continue
                                        else:
                                            conflicts[conflicts_counter] = [l1, l2, 1]
                                            conflicts_counter = conflicts_counter + 1
                            for z, conflict in reversed(list(enumerate(conflicts))):
                                if np.all(conflict == 0):
                                    conflicts = np.delete(conflicts, z, 0)
                            for z, conflict in list(enumerate(conflicts)):
                                if conflict[0] > conflict[1]:
                                    swap = conflict[0]
                                    if conflict[2] == 1:
                                        conflicts[z, 0] = conflicts[z, 1]
                                        conflicts[z, 1] = swap
                                        conflicts[z, 2] = 0
                                    else:
                                        conflicts[z, 0] = conflicts[z, 1]
                                        conflicts[z, 1] = swap
                                        conflicts[z, 2] = 1
                            stringToPrintConflicts = ""
                            for z, WeakConstraint in enumerate(WeakConstraintObjectList):
                                numberOfLiterals = len(WeakConstraint.get_literals())     # convert literals and terms from string to list of string!
                                numberOfTerms = len(WeakConstraint.get_terms())
                                if z not in conflicts[:, 1]:
                                    if numberOfLiterals > 1:
                                        if '-' in WeakConstraint.get_weight():
                                            stringToPrint = "priority " + str(z+1) + " - you appreciate more the one with "
                                        else:
                                            stringToPrint = "priority " + str(z+1) + " - you appreciate less the one with "
                                        for k, literal in enumerate(WeakConstraint.get_literals()):
                                            if k != 0:
                                                stringToPrint = stringToPrint + " and with "
                                            if "value" in literal:
                                                wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                                stringToPrint = stringToPrint + " more " + wordOfInterest
                                            else:
                                                stringToPrint = stringToPrint + literal
                                            if k == numberOfLiterals - 1:
                                                stringToPrint = stringToPrint + " with the same level of priority"
                                        print(stringToPrint)
                                    else:
                                        literal = WeakConstraint.get_literals()[0]
                                        if "value" in literal:
                                            wordOfInterest = WeakConstraint.get_literals()[0][WeakConstraint.get_literals()[0].find('(')+1:WeakConstraint.get_literals()[0].find(',')]
                                            if '-' in WeakConstraint.get_weight():
                                                print("priority " + str(z+1) + " - you appreciate more the one with more  " + wordOfInterest)
                                            else:
                                                print("priority " + str(z+1) + " - you appreciate less the one with more " + wordOfInterest)
                                        else:
                                            if '-' in WeakConstraint.get_weight():
                                                print("priority " + str(z+1) + " - you appreciate more the one with " + WeakConstraint.get_literals()[0])
                                            else:
                                                print("priority " + str(z+1) + " - you appreciate less the one with " + WeakConstraint.get_literals()[0])
                                else:
                                    if numberOfLiterals > 1:
                                        if '-' in WeakConstraint.get_weight():
                                            stringToPrint = "priority " + str(z+1) + " - you appreciate more the one with "
                                        else:
                                            stringToPrint = "priority " + str(z+1) + " - you appreciate less the one with "
                                        for k, literal in enumerate(WeakConstraint.get_literals()):
                                            if k != 0:
                                                stringToPrint = stringToPrint + " and with "
                                            if "value" in literal:
                                                wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                                stringToPrint = stringToPrint + " more " + wordOfInterest
                                            else:
                                                stringToPrint = stringToPrint + literal
                                            if k == numberOfLiterals - 1:
                                                stringToPrint = stringToPrint + " with the same level of priority"
                                        print(stringToPrint)
                                    else:
                                        literal = WeakConstraint.get_literals()[0]
                                        if "value" in literal:
                                            wordOfInterest = WeakConstraint.get_literals()[0][WeakConstraint.get_literals()[0].find('(')+1:WeakConstraint.get_literals()[0].find(',')]
                                            if '-' in WeakConstraint.get_weight():
                                                print("priority " + str(z+1) + " - you appreciate more the one with more " + wordOfInterest)
                                            else:
                                                print("priority " + str(z+1) + " - you appreciate less the one with more " + wordOfInterest)
                                        else:
                                            if '-' in WeakConstraint.get_weight():
                                                print("priority " + str(z+1) + " - you appreciate more the one with " + WeakConstraint.get_literals()[0])
                                            else:
                                                print("priority " + str(z+1) + " - you appreciate less the one with " + WeakConstraint.get_literals()[0])
                                    stringToPrintConflicts = stringToPrintConflicts + "- Although is true that in general "
                                    checker = 0 # counter of how many wcs are in conflicts with the interested one (wich is denoted by z)
                                    for conflict in conflicts:
                                        if z == conflict[1]:
                                            checker = checker + 1
                                    second_checker = checker - 1
                                    for conflict in conflicts:
                                        if z == conflict[1]:
                                            for z2, WeakConstraint2 in enumerate(WeakConstraintObjectList):
                                                if z2 == conflict[0]:
                                                    checker = checker - 1
                                                    if conflict[2] == 0:
                                                        if second_checker == checker:
                                                            stringToPrintConflicts = stringToPrintConflicts + "you appreciate more the one with "
                                                        numberOfLiterals2 = len(WeakConstraint2.get_literals())  # convert literals and terms from string to list of string!
                                                        numberOfTerms2 = len(WeakConstraint2.get_terms())
                                                        if numberOfLiterals2 > 1:
                                                            for k, literal in enumerate(WeakConstraint2.get_literals()):
                                                                if k != 0:
                                                                    stringToPrintConflicts = stringToPrintConflicts + " and with "
                                                                if "value" in literal:
                                                                    wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                                                    stringToPrintConflicts = stringToPrintConflicts + " more " + wordOfInterest
                                                                else:
                                                                    stringToPrintConflicts = stringToPrintConflicts + literal
                                                                if k == numberOfLiterals2 - 1:
                                                                    stringToPrintConflicts = stringToPrintConflicts
                                                        else:
                                                            literal = WeakConstraint2.get_literals()[0]
                                                            if "value" in literal:
                                                                wordOfInterest = WeakConstraint2.get_literals()[0][WeakConstraint2.get_literals()[0].find('(') + 1:WeakConstraint2.get_literals()[0].find(',')]
                                                                stringToPrintConflicts = stringToPrintConflicts + " more " + wordOfInterest
                                                            else:
                                                                stringToPrintConflicts = stringToPrintConflicts + WeakConstraint2.get_literals()[0]
                                                    else:
                                                        if second_checker == checker:
                                                            stringToPrintConflicts = stringToPrintConflicts + "you appreciate less the one with"
                                                        numberOfLiterals2 = len(WeakConstraint2.get_literals())  # convert literals and terms from string to list of string!
                                                        numberOfTerms2 = len(WeakConstraint2.get_terms())
                                                        if numberOfLiterals2 > 1:
                                                            for k, literal in enumerate(WeakConstraint2.get_literals()):
                                                                if k != 0:
                                                                    stringToPrintConflicts = stringToPrintConflicts + " and with "
                                                                if "value" in literal:
                                                                    wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                                                    stringToPrintConflicts = stringToPrintConflicts + " more " + wordOfInterest
                                                                else:
                                                                    stringToPrintConflicts = stringToPrintConflicts + literal
                                                                if k == numberOfLiterals2 - 1:
                                                                    stringToPrintConflicts = stringToPrintConflicts
                                                        else:
                                                            literal = WeakConstraint2.get_literals()[0]
                                                            if "value" in literal:
                                                                wordOfInterest = WeakConstraint2.get_literals()[0][WeakConstraint2.get_literals()[0].find('(') + 1:WeakConstraint2.get_literals()[0].find(',')]
                                                                stringToPrintConflicts = stringToPrintConflicts + " more " + wordOfInterest
                                                            else:
                                                                stringToPrintConflicts = stringToPrintConflicts + WeakConstraint2.get_literals()[0]
                                            if checker != 0:
                                                stringToPrintConflicts = stringToPrintConflicts + ", or with "
                                    if checker == 0:
                                        stringToPrintConflicts = stringToPrintConflicts + "; it\'s also true that "
                                    if numberOfLiterals > 1:
                                        if '-' in WeakConstraint.get_weight():
                                            stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also "
                                        else:
                                            stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also "
                                        for k, literal in enumerate(WeakConstraint.get_literals()):
                                            if k != 0:
                                                stringToPrintConflicts = stringToPrintConflicts + " and with "
                                            if "value" in literal:
                                                wordOfInterest = literal[literal.find('(') + 1:literal.find(',')]
                                                stringToPrintConflicts = stringToPrintConflicts + " more " + wordOfInterest
                                            else:
                                                stringToPrintConflicts = stringToPrintConflicts + literal
                                            if k == numberOfLiterals - 1:
                                                stringToPrint = stringToPrint + " with the same level of priority"
                                    else:
                                        literal = WeakConstraint.get_literals()[0]
                                        if "value" in literal:
                                            wordOfInterest = WeakConstraint.get_literals()[0][WeakConstraint.get_literals()[0].find('(')+1:WeakConstraint.get_literals()[0].find(',')]
                                            if '-' in WeakConstraint.get_weight():
                                                stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also more " + wordOfInterest
                                            else:
                                                stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also more " + wordOfInterest
                                        else:
                                            if '-' in WeakConstraint.get_weight():
                                                stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also " + WeakConstraint.get_literals()[0]
                                            else:
                                                stringToPrintConflicts = stringToPrintConflicts + "this it's not true when there is also " + WeakConstraint.get_literals()[0]
                                    stringToPrintConflicts = stringToPrintConflicts + "\n"
                            if stringToPrintConflicts != "":
                                macro_ingredients_list = ["cereali", "latticini", "uova", "farinacei", "frutta", "erbe_spezie_e_condimenti", "carne", "funghi_e_tartufi", "pasta", "pesce", "dolcificanti", "verdure_e_ortaggi"]

                                preparations = ["bollitura", "rosolatura", "frittura", "marinatura", "mantecatura", "forno", "cottura_a_fiamma", "stufato"]

                                categories = ["category(1)", "category(2)", "category(3)", "category(4)", "category(5)"]
                                list_conflict = list(stringToPrintConflicts.split("\n"))
                                for line_counter, lineConflict in enumerate(stringToPrintConflicts.split("\n")):
                                    for macro_ingredient in macro_ingredients_list:
                                        if lineConflict.count(macro_ingredient) >= 2:
                                            temp_line = stringToPrintConflicts.replace(" and with more " + macro_ingredient, "")
                                            list_conflict[line_counter] = temp_line.replace(" and with " + macro_ingredient, "")
                                    for preparation in preparations:
                                        if lineConflict.count(preparation) >= 2:
                                            temp_line = stringToPrintConflicts.replace(" and with more " + preparation, "")
                                            list_conflict[line_counter] = temp_line.replace(" and with " + preparation, "")
                                    for category in categories:
                                        if lineConflict.count(category) >= 2:
                                            temp_line = stringToPrintConflicts.replace(" and with more " + category, "")
                                            list_conflict[line_counter] = temp_line.replace(" and with " + category, "")
                                stringToPrintConflicts = ""
                                for conflict_in_list in list_conflict:
                                    stringToPrintConflicts += conflict_in_list + "\n"
                                stringToPrintConflicts = stringToPrintConflicts[0:(len(stringToPrintConflicts) - 1)]    # remove last "\n"
                                print('Given two recipes is also true that:')
                                print(stringToPrintConflicts)
                        print('---------------------------------------------------------------------------------------')

def translate_theory_old(theory1):
    macro_ingredients_art_i = ["cereali", "latticini", "farinacei", "dolcificanti"]
    macro_ingredients_art_le = ["uova"]
    macro_ingredients_art_la = ["frutta", "carne", "pasta"]
    macro_ingredients_art_il = ["pesce"]
    macro_ingredients_art_comp = ["erbe_spezie_e_condimenti", "funghi_e_tartufi", "verdure_e_ortaggi"]

    preparations = ["bollitura", "rosolatura", "frittura", "marinatura", "mantecatura", "forno", "cottura_a_fiamma", "stufato"]

    categories = ["category(1)", "category(2)", "category(3)", "category(4)", "category(5)"]

    theory2 = theory1.replace("  ", " ")
    theory3 = theory2.replace("you appreciate more the one with", "apprezzi di più quella con")
    theory4 = theory3.replace("you appreciate less the one with", "apprezzi di meno quella con")
    theory5 = theory4.replace("proportionally to its quantity/importance in the recipe", "in base alla sua quantità/importanza nella ricetta")
    theory6 = theory5.replace("given two recipes:", "Date due ricette:")
    theory7 = theory6.replace("Given two recipes is also true that:", "Date due ricette è anche vero che:")
    theory8 = theory7.replace("Although is true that in general", "Seppur sia vero che in generale")
    theory9 = theory8.replace("and", "e")
    theory10 = theory9.replace("with the same level of priority", "")
    theory11 = theory10.replace("Preferences are written from the one with less priority to the one with high priority", "Le preferenze sono scritte in ordine di priorità crescente.")
    theory12 = theory11.replace("The conflicts are grouped by the right part of the statement (so wcs x(1), x(2), ..., x(n) which are contradicted by the same wc y) e are written with the same order considered for preferences referred to wc y", "I conflitti sono raggruppati rispetto alla parte destra della sentenza (quindi wcs x(1), x(2), ..., x(n) i quali sono contraddetti da un wc y) e sono scritti sempre in ordine di priorità crescente.")
    theory13 = theory12.replace("User", "Utente")
    theory14 = theory13.replace("Dataset of size", "Dataset di taglia")
    theory15 = theory14.replace(" it's also true that", "")
    theory16 = theory15.replace("them when there is", "esso/i quando vi è")
    theory17 = theory16.replace("priority", "priorità")
    theory18 = theory17.replace("you appreciate more", "apprezzi di più")
    theory19 = theory18.replace("you appreciate less", "apprezzi di meno")
    theory20 = theory19.replace("the second when there is", "la seconda quando vi è")
    theory21 = theory20.replace("this it's not true when there is also", "ciò ciambia quando vi è presente anche")
    theory22 = theory21.replace("with", "con")
    theory23 = theory22.replace("cost", "costo")
    theory24 = theory23.replace("difficulty", "difficolta")
    theory25 = theory24.replace("prepTime", "tempo di preparazione")
    theory = theory25.replace(" or ", " o ")
    del theory1, theory2, theory3, theory4, theory5, theory6, theory7, theory8, theory9, theory10, theory11, theory12, theory13, theory14, theory15, theory16, theory17, theory18, theory19, theory20, theory21, theory22

    temp_theory = theory
    for macro_ingredient_art_i in macro_ingredients_art_i:
        theory = temp_theory.replace(macro_ingredient_art_i, "i " + str(macro_ingredient_art_i))
        temp_theory = theory
    for macro_ingredient_art_le in macro_ingredients_art_le:
        theory = temp_theory.replace(macro_ingredient_art_le, "le " + str(macro_ingredient_art_le))
        temp_theory = theory
    for macro_ingredient_art_la in macro_ingredients_art_la:
        theory = temp_theory.replace(macro_ingredient_art_la, "la " + str(macro_ingredient_art_la))
        temp_theory = theory
    for macro_ingredient_art_il in macro_ingredients_art_il:
        theory = temp_theory.replace(macro_ingredient_art_il, "il " + str(macro_ingredient_art_il))
        temp_theory = theory
    for i, macro_ingredient_art_comp in enumerate(macro_ingredients_art_comp):
        if i == 0:
            theory = temp_theory.replace(macro_ingredient_art_comp, "erbe, spezie e condimenti")
            temp_theory = theory
        elif i == 1:
            theory = temp_theory.replace(macro_ingredient_art_comp, "funghi e tartufi")
            temp_theory = theory
        else:
            theory = temp_theory.replace(macro_ingredient_art_comp, "verdure ed ortaggi")
            temp_theory = theory

    for preparation in preparations:
        theory = temp_theory.replace(preparation, "la cottura \"" + str(preparation) + "\"")
        temp_theory = theory

    for i, category in enumerate(categories):
        if i == 0:
            theory = temp_theory.replace("con " + category, "che è un \"antipasto\"")
            temp_theory = theory
            theory = temp_theory.replace("e " + category, "e che è un \"antipasto\"")
            temp_theory = theory
            theory = temp_theory.replace("vi è " + category, "la ricetta in questione è un \"antipasto\"")
            temp_theory = theory
        elif i == 1:
            theory = temp_theory.replace("con " + category, "che è un \"piatto unico\"")
            temp_theory = theory
            theory = temp_theory.replace("e " + category, "e che è un \"piatto unico\"")
            temp_theory = theory
            theory = temp_theory.replace("vi è " + category, "la ricetta in questione è un \"piatto unico\"")
            temp_theory = theory
        elif i == 2:
            theory = temp_theory.replace("con " + category, "che è un \"primo\"")
            temp_theory = theory
            theory = temp_theory.replace("e " + category, "e che è un \"primo\"")
            temp_theory = theory
            theory = temp_theory.replace("vi è " + category, "la ricetta in questione è un \"primo\"")
            temp_theory = theory
        elif i == 3:
            theory = temp_theory.replace("con " + category, "che è un \"secondo\"")
            temp_theory = theory
            theory = temp_theory.replace("e " + category, "e che è un \"secondo\"")
            temp_theory = theory
            theory = temp_theory.replace("vi è " + category, "la ricetta in questione è un \"secondo\"")

        else:
            theory = temp_theory.replace("con " + category, "che è una \"torta salata\"")
            temp_theory = theory
            theory = temp_theory.replace("e " + category, "e che è una \"torta salata\"")
            temp_theory = theory
            theory = temp_theory.replace("vi è " + category, "la ricetta in questione è un \"torta salata\"")
            temp_theory = theory

    theory1 = theory.replace("vi è presente anche che è", "è")
    theory2 = theory1.replace("vi è presente anche i", "vi sono presenti anche i")
    theory = theory2.replace("vi è presente anche le", "vi sono presenti anche le")
    del theory1, theory2

    return theory

def translate_theory(theory):
    theory1 = theory.replace("  ", " ")
    theory2 = theory1.replace("category(1)", "starter")
    theory3 = theory2.replace("category(2)", "complete meal")
    theory4 = theory3.replace("category(3)", "first course")
    theory5 = theory4.replace("category(4)", "second course")
    theory6 = theory5.replace("category(5)", "savory cake")
    theory7 = theory6.replace("with starter", "which is a starter")
    theory8 = theory7.replace("with complete meal", "which is a complete meal")
    theory9 = theory8.replace("with first course", "which is a first course")
    theory10 = theory9.replace("with second course", "which is a second course")
    theory11 = theory10.replace("with savory cake", "which is savory cake")
    theory12 = theory11.replace("with more cost", "which costs more")
    theory13 = theory12.replace("with more prepTime", "which requires more preparation time")
    theory14 = theory13.replace("with more difficulty", "which is more difficult to prepare")
    theory15 = theory14.replace("_", " ")
    translated = GoogleTranslator(source='english', target='italian').translate(theory15)
    translated1 = translated.replace("quello con", "quella con")
    translated2 = translated1.replace("quello che", "quella che")
    translated3 = translated2.replace("Anche se è vero che in genere", "Seppur sia vero che in generale")
    translated4 = translated3.replace("si apprezza", "apprezzi")
    translated5 = translated4.replace("non si apprezza", "non apprezzi")
    translated6 = translated5.replace("c'è anche più bollitura", "la ricetta è preparata tramite bollitura")
    translated7 = translated6.replace("c'è anche più rosolatura", "la ricetta è preparata tramite rosolatura")
    translated8 = translated7.replace("c'è anche più frittura", "la ricetta è preparata tramite frittura")
    translated9 = translated8.replace("c'è anche più marinatura", "la ricetta è preparata tramite marinatura")
    translated10 = translated9.replace("c'è anche più mantecatura", "la ricetta è preparata tramite mantecatura")
    translated11 = translated10.replace("c'è anche più forno", "la ricetta è preparata tramite cottura al forno")
    translated12 = translated11.replace("c'è anche più cottura a fiamma", "la ricetta è preparata tramite cottura a fiamma")
    translated13 = translated12.replace("c'è anche più stufato", "la ricetta è preparata tramite stufato")
    translated14 = translated13.replace("con più bollitura", "in cui la ricetta è preparata tramite bollitura")
    translated15 = translated14.replace("con più rosolatura", "in cui la ricetta è preparata tramite rosolatura")
    translated16 = translated15.replace("con più frittura", "in cui la ricetta è preparata tramite frittura")
    translated17 = translated16.replace("con più marinatura", "in cui la ricetta è preparata tramite marinatura")
    translated18 = translated17.replace("con più mantecatura", "in cui la ricetta è preparata tramite mantecatura")
    translated19 = translated18.replace("con più forno", "in cui la ricetta è preparata tramite cottura al forno")
    translated20 = translated19.replace("con più cottura a fiamma", "in cui la ricetta è preparata tramite cottura a fiamma")
    translated21 = translated20.replace("con più stufato", "in cui la ricetta è preparata tramite stufato")
    translated22 = translated21.replace(" con lo stesso livello di priorità", "")
    translated23 = translated22.replace("è anche vero che questo non è vero", "ciò non è più vero")
    translated24 = translated23.replace("ciò non è più vero quando la ricetta è", "ciò non è più vero quando quella stessa ricetta è")
    translated25 = translated24.replace("ciò non è più vero quando c'è anche", "ciò non è più vero quando in quella stessa ricetta c'è anche")
    translated26 = translated25.replace("primo piatto", "primo")
    translated27 = translated26.replace("secondo piatto", "secondo")
    translated28 = translated27.replace("e che è un antipasto", "ed è un antipasto")
    translated29 = translated28.replace("e che è un pasto completo", "ed è un pasto completo")
    translated30 = translated29.replace("e che è un primo", "ed è un primo")
    translated31 = translated30.replace("e che è un secondo", "ed è un secondo")
    translated32 = translated31.replace("e che è un torta salata", "ed è una torta salata")

    return translated32


def theoriesScoreCalculator(theory, user, scope):
    columns = ["dolcificanti", "farinacei", "erbe_spezie_e_condimenti", "carne", "cereali", "frutta", "funghi_e_tartufi", "latticini", "pasta", "pesce", "uova", "verdure_e_ortaggi", "bollitura", "cottura_a_fiamma", "cottura_a_vapore", "cottura_in_forno", "frittura", "mantecatura", "marinatura", "rosolatura", "stufato", "difficolta", "tempo_di_preparazione", "costo"]
    users_ground_truth = pd.DataFrame(columns=[*columns])
    for dirname, _, filenames in os.walk('./Answers_dataset/'):
        if "answers-of-return" in dirname:
            continue
        for index, filename in enumerate(filenames):
            path = os.path.join(dirname, filename)
            considered_survey = pd.read_csv(path, delimiter=";")
            for i, answer in enumerate(considered_survey.iterrows()):
                to_fill = np.zeros((len(columns)), dtype="float32")
                to_fill[0] = answer[1]["Ti piacciono le ricette in cui sono presenti i dolcificanti?"]
                to_fill[1] = answer[1]["Ti piacciono le ricette in cui è presente la farina?"]
                to_fill[2] = answer[1]["Ti piacciono le ricette speziate?"]
                to_fill[3] = answer[1]["Ti piacciono i cibi a base di carne?"]
                to_fill[4] = answer[1]["Ti piacciono i cibi a base di cereali?"]
                to_fill[5] = answer[1]["Ti piacciono i cibi a base di frutta?"]
                to_fill[6] = answer[1]["Ti piacciono i cibi a base di funghi e tartufo?"]
                to_fill[7] = answer[1]["Ti piacciono i cibi a base di latticini?"]
                to_fill[8] = answer[1]["Ti piacciono i cibi a base di pasta?"]
                to_fill[9] = answer[1]["Ti piacciono i cibi a base di pesce?"]
                to_fill[10] = answer[1]["Ti piacciono i cibi a base di uova?"]
                to_fill[11] = answer[1]["Ti piacciono i cibi a base di verdure e ortaggi?"]
                to_fill[12] = answer[1]["Ti piacciono i piatti che richiedono la bollitura?"]
                to_fill[13] = answer[1]["Ti piacciono i piatti che richiedono la cottura a fiamma?"]
                to_fill[14] = answer[1]["Ti piacciono i piatti che richiedono la cottura a vapore?"]
                to_fill[15] = answer[1]["Ti piacciono i piatti che richiedono la cottura in forno?"]
                to_fill[16] = answer[1]["Ti piacciono i piatti che richiedono la frittura?"]
                to_fill[17] = answer[1]["Ti piacciono i piatti che richiedono la mantecatura?"]
                to_fill[18] = answer[1]["Ti piacciono i piatti che richiedono la marinatura?"]
                to_fill[19] = answer[1]["Ti piacciono i piatti che richiedono la rosolatura?"]
                to_fill[20] = answer[1]["Ti piacciono i piatti che richiedono lo stufato?"]
                to_fill[21] = answer[1]["Preferisci la cucina semplice a quella elaborata?"]
                to_fill[22] = answer[1]["Sei disposto ad aspettare per un piatto con elevato tempo di preparazione?"]
                to_fill[23] = answer[1]["Sei disposto a spendere per un buon piatto?"]
                temp_to_insert = pd.DataFrame([to_fill], columns=[*columns])
                users_ground_truth = users_ground_truth.append(temp_to_insert)
                del temp_to_insert
    # users_ground_truth.hist(bins=len(users_ground_truth.columns), figsize=(20, 20))
    # plt.show()

    users_ground_truth.reset_index(inplace=True)
    values_for_sentence = np.zeros((5, 5), dtype="float32")
    sense_for_sentence = np.zeros((5), dtype="float32")
    counter_values = 0
    for line_index, line in enumerate(theory.split("\n")):
        if scope == "global":
            if line_index <= 4:
                continue
        else:
            if line_index <= 1:
                continue
        counter_vars = 0
        if "priority" in line:
            if "you appreciate less" in line:
                sense_for_sentence[counter_values] = -1
            else:
                sense_for_sentence[counter_values] = 1
            if "dolcificanti" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["dolcificanti"]
                counter_vars += 1
            if "farinacei" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["farinacei"]
                counter_vars += 1
            if "erbe_spezie_e_condimenti" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["erbe_spezie_e_condimenti"]
                counter_vars += 1
            if "carne" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["carne"]
                counter_vars += 1
            if "cereali" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["cereali"]
                counter_vars += 1
            if "frutta" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["frutta"]
                counter_vars += 1
            if "funghi_e_tartufi" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["funghi_e_tartufi"]
                counter_vars += 1
            if "latticini" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["latticini"]
                counter_vars += 1
            if "pasta" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["pasta"]
                counter_vars += 1
            if "pesce" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["pesce"]
                counter_vars += 1
            if "uova" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["uova"]
                counter_vars += 1
            if "verdure_e_ortaggi" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["verdure_e_ortaggi"]
                counter_vars += 1
            if "bollitura" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["bollitura"]
                counter_vars += 1
            if "cottura_a_fiamma" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["cottura_a_fiamma"]
                counter_vars += 1
            if "cottura_a_vapore" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["cottura_a_vapore"]
                counter_vars += 1
            if "cottura_in_forno" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["cottura_in_forno"]
                counter_vars += 1
            if "frittura" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["frittura"]
                counter_vars += 1
            if "mantecatura" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["mantecatura"]
                counter_vars += 1
            if "marinatura" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["marinatura"]
                counter_vars += 1
            if "rosolatura" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["rosolatura"]
                counter_vars += 1
            if "stufato" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["stufato"]
                counter_vars += 1
            if "difficola" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["difficolta"]
                counter_vars += 1
            if "tempo_di_preparazione" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["tempo_di_preparazione"]
                counter_vars += 1
            if "costo" in line:
                values_for_sentence[counter_values, counter_vars] = users_ground_truth.loc[user]["costo"]
                counter_vars += 1
        counter_values += 1
    list_sentences_wrong_sense = []
    list_index_wrong_position = []
    list_index_uncertain_position = []
    for index_sentence, (sentence_value, sentence_sense) in enumerate(zip(values_for_sentence, sense_for_sentence)):
        if sentence_sense == 0:
            continue    # this means that both wc don't exist (we set empty wc to arrive to 5 wc)
        all_zero = True
        for value in sentence_value:
            if value != 0:
                all_zero = False
        if all_zero:
            continue    # this is a special case that could arise since in the ground-truth we don't have rating on categorical data (but we can retrieve sense in the theory)
        if (sentence_sense == 1 and np.nanmean(np.where(sentence_value != 0, sentence_value, np.nan)) < 6) or (sentence_sense == -1 and np.nanmean(np.where(sentence_value != 0, sentence_value, np.nan)) > 5):
            list_sentences_wrong_sense.append(index_sentence)
    for index_sentence_i, sentence_value_i in enumerate(values_for_sentence):
        is_wrong = False
        for index_sentence_j, sentence_value_j in enumerate(values_for_sentence):
            if index_sentence_j <= index_sentence_i:
                continue
            all_zero_i = True
            all_zero_j = True
            for sentence_value_i_element in sentence_value_i:
                if sentence_value_i_element != 0:
                    all_zero_i = False
                    break
            for sentence_value_i_element in sentence_value_j:
                if sentence_value_i_element != 0:
                    all_zero_j = False
                    break
            if all_zero_i and (not all_zero_j):
                if 0 < np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if 0 > np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if 0 == np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if (sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1) or (sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1):
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
            elif (not all_zero_i) and all_zero_j:
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) < 0:
                    if sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) > 0:
                    if sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) == 0:
                    if (sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1) or (sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1):
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
            elif all_zero_i and all_zero_j:
                continue    # this means that both wc don't exist (we set empty wc to arrive to 5 wc)
            else:
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) < np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) > np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1:
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
                if np.nanmean(np.where(sentence_value_i != 0, sentence_value_i, np.nan)) == np.nanmean(np.where(sentence_value_j != 0, sentence_value_j, np.nan)):
                    if (sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == 1) or (sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == -1):
                        list_index_wrong_position.append(index_sentence_i)
                        is_wrong = True
        if is_wrong:
            continue
        for index_sentence_j, sentence_value_j in enumerate(values_for_sentence):
            if index_sentence_j <= index_sentence_i:
                continue
            if sense_for_sentence[index_sentence_i] == 1 and sense_for_sentence[index_sentence_j] == 1:
                list_index_uncertain_position.append(index_sentence_i)
            if sense_for_sentence[index_sentence_i] == -1 and sense_for_sentence[index_sentence_j] == -1:
                list_index_uncertain_position.append(index_sentence_i)
    list_index_wrong_position = list(set(list_index_wrong_position))
    list_index_uncertain_position = list(set(list_index_uncertain_position))
    return len(list_sentences_wrong_sense), len(list_index_wrong_position), len(list_index_uncertain_position)



if __name__ == '__main__':
    columns = ["dolcificanti", "farinacei", "erbe_spezie_e_condimenti", "carne", "cereali", "frutta", "funghi_e_tartufi", "latticini", "pasta", "pesce", "uova", "verdure_e_ortaggi", "bollitura", "cottura_a_fiamma", "cottura_a_vapore", "cottura_in_forno", "frittura", "mantecatura", "marinatura", "rosolatura", "stufato", "difficolta", "tempo_di_preparazione", "costo", "antipasti", "piatto_unico", "primo", "secondo", "torta_salata"]
    collect_data = pd.DataFrame(np.zeros((50, len(columns)), dtype="float32"), columns=[*columns])
    counter_collector = 0
    # GLOBAL PART
    # list_score_wrong_sense_45 = []
    # list_score_wrong_position_45 = []
    # list_score_uncertain_position_45 = []
    # list_score_wrong_sense_105 = []
    # list_score_wrong_position_105 = []
    # list_score_uncertain_position_105 = []
    # list_score_wrong_sense_210 = []
    # list_score_wrong_position_210 = []
    # list_score_uncertain_position_210 = []
    # list_score_wrong_sense_150 = []
    # list_score_wrong_position_150 = []
    # list_score_uncertain_position_150 = []
    # list_number_weak_constraint_150 = []
    # users = [3, 4, 7, 11, 14, 15, 20, 29, 32, 36]
    # couples = [157]
    # for user in users:
    #     for couple in couples:
    #         buffer = StringIO()
    #         sys.stdout = buffer
    #         # printTheory('ILASPcode/Data/testOutput_original/results_zero.csv', user=user, max_v=5, max_p=5, couple=couple, data_collector=collect_data, data_counter_collector = counter_collector)
    #         printTheory('ILASPcode/PCAexperiment/testOutput_original20/results_zero.csv', user=user, max_v=5, max_p=5, couple=couple, data_collector=collect_data, data_counter_collector=counter_collector)
    #         counter_collector += 1
    #         sys.stdout = sys.__stdout__
    #         not_translated_theory = buffer.getvalue()
    #         score_wrong_sense, score_wrong_position, score_uncertain_position = theoriesScoreCalculator(not_translated_theory, user=user, scope="global")
    #         # start only for ILASP as classsifier
    #         counter_wc = 0
    #         for i_line, line in enumerate(not_translated_theory.split("\n")):
    #             if i_line <= 4:
    #                 continue
    #             if i_line > 9:
    #                 break
    #             if "Although" in line:
    #                 continue
    #             if "priority" in line:
    #                 counter_wc+=1
    #
    #
    #         # end part only for ILASP as classifier
    #         # if couple == 45:
    #         #     list_score_wrong_sense_45.append(score_wrong_sense)
    #         #     list_score_wrong_position_45.append(score_wrong_position)
    #         #     list_score_uncertain_position_45.append(score_uncertain_position)
    #         # if couple == 105:
    #         #     list_score_wrong_sense_105.append(score_wrong_sense)
    #         #     list_score_wrong_position_105.append(score_wrong_position)
    #         #     list_score_uncertain_position_105.append(score_uncertain_position)
    #         # if couple == 210:
    #         #     list_score_wrong_sense_210.append(score_wrong_sense)
    #         #     list_score_wrong_position_210.append(score_wrong_position)
    #         #     list_score_uncertain_position_210.append(score_uncertain_position)
    #         if couple == 157:
    #             list_score_wrong_sense_150.append(score_wrong_sense)
    #             list_score_wrong_position_150.append(score_wrong_position)
    #             list_score_uncertain_position_150.append(score_uncertain_position)
    #             list_number_weak_constraint_150.append(counter_wc)
    # collect_data.replace(0, np.nan, inplace=True)
    # collect_data.hist(bins=5, figsize=(20, 20), range=[1, 5])
    # plt.show()
    # # print("On dataset with 45 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_45)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_45)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_45)))
    # # print("On dataset with 105 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_105)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_105)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_105)))
    # # print("On dataset with 190 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_210)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_210)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_210)))
    # print("On dataset with 105 pairs (#wc = " + str(np.mean(list_number_weak_constraint_150)) + "): mean of WSS = " + str(np.mean(list_score_wrong_sense_150)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_150)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_150)))

    # LOCAL PART
    list_score_wrong_sense_45 = []
    list_score_wrong_position_45 = []
    list_score_uncertain_position_45 = []
    list_score_wrong_sense_105 = []
    list_score_wrong_position_105 = []
    list_score_uncertain_position_105 = []
    list_score_wrong_sense_210 = []
    list_score_wrong_position_210 = []
    list_score_uncertain_position_210 = []
    users = [3, 4, 7, 11, 14, 15, 20, 29, 32, 36]
    couples = [45]
    for user in users:
        for couple in couples:
            buffer = StringIO()
            sys.stdout = buffer
            if couple != 210:
                printTheory('ILASPcode/local/local/Data/theories/results_zero_' + str(couple) +'_gauss_std0.1.csv', user=user, max_v=1, max_p=5, couple=couple, data_collector=collect_data, data_counter_collector = counter_collector)
            else:
                printTheory('ILASPcode/local/local/Data/theories/results_zero_190_gauss_std0.1.csv', user=user, max_v=1, max_p=5, couple=couple, data_collector=collect_data, data_counter_collector = counter_collector)
            sys.stdout = sys.__stdout__
            # translated_theory = translate_theory(buffer.getvalue())
            # list_of_theories = list(translated_theory.split("Utente"))
            theory = buffer.getvalue()
            list_of_theories = list(theory.split("User"))
            list_score_wrong_sense_45_single_user = []
            list_score_wrong_position_45_single_user = []
            list_score_uncertain_position_45_single_user = []
            list_score_wrong_sense_105_single_user = []
            list_score_wrong_position_105_single_user = []
            list_score_uncertain_position_105_single_user = []
            list_score_wrong_sense_210_single_user = []
            list_score_wrong_position_210_single_user = []
            list_score_uncertain_position_210_single_user = []
            for single_theory_index, single_theory in enumerate(list_of_theories):
                if single_theory_index == 0:
                    continue
                score_wrong_sense, score_wrong_position, score_uncertain_position = theoriesScoreCalculator(single_theory, user=user, scope="local")
                if couple == 45:
                    list_score_wrong_sense_45_single_user.append(score_wrong_sense)
                    list_score_wrong_position_45_single_user.append(score_wrong_position)
                    list_score_uncertain_position_45_single_user.append(score_uncertain_position)
                if couple == 105:
                    list_score_wrong_sense_105_single_user.append(score_wrong_sense)
                    list_score_wrong_position_105_single_user.append(score_wrong_position)
                    list_score_uncertain_position_105_single_user.append(score_uncertain_position)
                if couple == 210:
                    list_score_wrong_sense_210_single_user.append(score_wrong_sense)
                    list_score_wrong_position_210_single_user.append(score_wrong_position)
                    list_score_uncertain_position_210_single_user.append(score_uncertain_position)
            if couple == 45:
                list_score_wrong_sense_45.append(np.mean(list_score_wrong_sense_45_single_user))
                list_score_wrong_position_45.append(np.mean(list_score_wrong_position_45_single_user))
                list_score_uncertain_position_45.append(np.mean(list_score_uncertain_position_45_single_user))
            if couple == 105:
                list_score_wrong_sense_105.append(np.mean(list_score_wrong_sense_105_single_user))
                list_score_wrong_position_105.append(np.mean(list_score_wrong_position_105_single_user))
                list_score_uncertain_position_105.append(np.mean(list_score_uncertain_position_105_single_user))

            if couple == 210:
                list_score_wrong_sense_210.append(np.mean(list_score_wrong_sense_210_single_user))
                list_score_wrong_position_210.append(np.mean(list_score_wrong_position_210_single_user))
                list_score_uncertain_position_210.append(np.mean(list_score_uncertain_position_210_single_user))
    print("On dataset with 45 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_45)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_45)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_45)))
    print("On dataset with 105 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_105)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_105)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_105)))
    print("On dataset with 190 pairs: mean of WSS = " + str(np.mean(list_score_wrong_sense_210)) + "; mean of WOS = " + str(np.mean(list_score_wrong_position_210)) + "; mean of UOS = " + str(np.mean(list_score_uncertain_position_210)))



    # ---------------------------------------------------------------------------------------------------------------------------------
    # buffer = StringIO()
    # sys.stdout = buffer
    # printTheory('ILASPcode/Data/testOutput/results_zero.csv', user=4, max_v=5, max_p=5, couple=210)
    # sys.stdout = sys.__stdout__
    # translated_theory = translate_theory(buffer.getvalue())
    # print(translated_theory)
    # score1, score2 = theoriesScoreCalculator(translated_theory, user=4)
    # buffer = StringIO()
    # sys.stdout = buffer
    # printTheory('ILASPcode/PCAexperiment/testOutput20/results_zero.csv', user=4, max_v=5, max_p=5, couple=190)  # note, remember that 210 is 190 (it's annoted as 210 for a mistake)
    # sys.stdout = sys.__stdout__
    # translated_theory = translate_theory(buffer.getvalue())
    # print(translated_theory)