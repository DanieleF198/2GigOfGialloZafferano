import os
import ILASPparser
from io import StringIO
import docx
from docx import oxml
import sys
import numpy as np
import re


users_of_survey = [0, 4]   # 0, 4, 10, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 31, 41, 42, 43, 44

f_preferences_test_local_dir = "./ILASPcode/local/local/Data8Component2Std/sampled-recipes-zero/Train105_gauss/std-0.1/"
fNames = os.path.join('./dataset_100/separated_text_data/names.txt')
fLinks = os.path.join('./dataset_100/separated_text_data/links.txt')

fN = open(fNames)
dataNames = fN.read()
fN.close()
fL = open(fLinks)
dataLinks = fL.read()
fL.close()

linesOfN = dataNames.split('\n')
food_data_names = []
for i, line in enumerate(linesOfN):
    food_data_names.append(line)
linesOfL = dataLinks.split('\n')
food_data_links = []
for i, line in enumerate(linesOfL):
    food_data_links.append(line)

for user in users_of_survey:

    fPTest = open(os.path.join(f_preferences_test_local_dir + "couple" + str(user) + ".txt"))
    dataPTest = fPTest.read()
    fPTest.close()

    linesOfPTest = dataPTest.split('\n')
    couples_dataset_test = np.zeros((100, 2), dtype='int32')
    for i, line in enumerate(linesOfPTest):
        if line == "":
            continue
        couple = [x for x in line.split(';')[0:]]
        for j, element in enumerate(couple):
            couples_dataset_test[i, j] = int(element)

    doc = docx.Document()

    title = doc.add_heading('Verifica delle preferenze gastronomiche', level=1)

    introduction_paragraph = doc.add_paragraph()
    introduction_paragraph.add_run("Ciao!")
    introduction_paragraph.add_run("\n")
    introduction_paragraph.add_run("In passato le abbiamo chiesto di rispondere ad un sondaggio dove le venivano chieste alcune informazioni riguardo alle vostre preferenze gastronomiche, questo al fine di raccogliere dati da usare in degli studi relativi ad algoritmi di machine learning e sistemi di apprendimento induttivo basati sulla logica (ILASP).")
    introduction_paragraph.add_run("\n")
    introduction_paragraph.add_run("Come forse ricorderai nel sondaggio era richiesta opzionalmente la vostra mail, questo al fine di ricontattarvi e sottoporvi nuovamente ad un sondaggio. Questa volta l'obiettivo è quello di ottenere dati che ci diano delle conferme (o meno) sugli studi condotti, ed avere un feedback su di essi.")
    introduction_paragraph.add_run("\n")
    introduction_paragraph.add_run("Quello che le chiediamo di fare è semplicemente rispondere nuovamente a delle domande riguardanti le sue preferenze gastronomiche. Non richiederà più di 10 minuti e ci permetterà di condurre degli studi più accurati.")
    introduction_paragraph.add_run("\n")
    introduction_paragraph.add_run("Tutte le ricette presenti sono consultabili sul sito Giallo Zafferano.")
    introduction_paragraph.add_run("\n")
    introduction_paragraph.add_run("La ringraziamo per il tuo tempo.")

    # doc.add_heading("Valutazione delle sentenze - parte 1", level=1)
    # val_sent_1_intr_paragraph = doc.add_paragraph()
    # val_sent_1_intr_paragraph.add_run("Di seguito viene riportata descrizione su quelli che potrebbero essere alcuni dei suoi gusti.")
    # val_sent_1_intr_paragraph.add_run("\n")
    # val_sent_1_intr_paragraph.add_run("La descrizione riporta le vostre preferenze in ordine crescente di priorità (dove 1 è la priorità minima mentre 5 è la priorità massima) e le eccezioni, ovvero ricette in cui una preferenza viene a mancare a causa della presenza di un certa caratteristica nella relativa ricetta.")
    # val_sent_1_intr_paragraph.add_run("\n")
    #
    # buffer = StringIO()
    # sys.stdout = buffer
    # ILASPparser.printTheory('ILASPcode/Data8Component2Std/testOutput/results_zero.csv', user=user, max_v=5, max_p=5, couple=210)
    # sys.stdout = sys.__stdout__
    # translated_theory = ILASPparser.translate_theory(buffer.getvalue())
    # counter_guard = len(translated_theory.split("\n"))
    # val_sent_1 = doc.add_paragraph()
    # for j, line in enumerate(translated_theory.split("\n")):
    #     if j <= 3:
    #         continue
    #     if line == "---------------------------------------------------------------------------------------":
    #         continue
    #     val_sent_1.add_run(line)
    #     if j >= counter_guard - 3:
    #         break
    #     val_sent_1.add_run("\n")
    #
    # for i in range(1, 6):
    #     buffer = StringIO()
    #     sys.stdout = buffer
    #     ILASPparser.printTheory('ILASPcode/Data8Component2Std/testOutput/results_zero.csv', user=user, max_v=i, max_p=i, couple=210)
    #     sys.stdout = sys.__stdout__
    #     global_theory1 = buffer.getvalue()
    #     global_theory2 = global_theory1.replace("  ", " ", )
    #     global_theory3 = global_theory2.replace("you appreciate", "apprezzi")
    #     global_theory4 = global_theory3.replace("you don't appreciate", "non apprezzi")
    #     global_theory5 = global_theory4.replace("proportionally to its presence/importance in the recipe", "proporzionalmente alla sua presenza/importanza nella ricetta")
    #     global_theory6 = global_theory5.replace("preferences", "preferenze")
    #     global_theory7 = global_theory6.replace("conflicts", "conflitti")
    #     global_theory8 = global_theory7.replace("Although it's true that", "Seppur sia vero che")
    #     global_theory9 = global_theory8.replace("and", "e")
    #     global_theory10 = global_theory9.replace("with the same level of priority", "con lo stesso livello di priorità")
    #     global_theory11 = global_theory10.replace("Preferences are written from the one with less priority to the one with high priority", "le preferenze sono scritte in ordine di priorità crescente")
    #     global_theory12 = global_theory11.replace("The conflitti are grouped by the right part of the statement (so wcs x(1), x(2), ..., x(n) which are contradicted by the same wc y) e are written with the same order considered for preferenze referred to wc y", "I conflitti sono raggruppati rispetto alla parte destra della sentenza (quindi wcs x(1), x(2), ..., x(n) i quali sono contraddetti da un wc y) e sono scritti sempre in ordine di priorità crescente")
    #     global_theory13 = global_theory12.replace("User", "Utente")
    #     global_theory14 = global_theory13.replace("Dataset of size", "Dataset di taglia")
    #     global_theory15 = global_theory14.replace("it's also true that", "è anche vero che")
    #     global_theory = global_theory15.replace("them when there is", "esso/i quando vi è")
    #     del global_theory1, global_theory2, global_theory3, global_theory4, global_theory5, global_theory6, global_theory7, global_theory8, global_theory9, global_theory10, global_theory11, global_theory12, global_theory13, global_theory14, global_theory15
    #
    #     counter_guard = len(global_theory.split("\n"))
    #     val_sent_1 = doc.add_paragraph(style="ListNumber")
    #     val_sent_1.add_run("Ti ritrovi in questa descrizione dei tuoi gusti?")
    #     val_sent_1.add_run("\n")
    #     for j, line in enumerate(global_theory.split("\n")):
    #         if j <= 3:
    #             continue
    #         if line == "---------------------------------------------------------------------------------------":
    #             continue
    #         val_sent_1.add_run(line)
    #         if j >= counter_guard - 3:
    #             break
    #         val_sent_1.add_run("\n")
    #     doc.add_paragraph("Sì", style="ListBullet")
    #     doc.add_paragraph("No", style="ListBullet")

    doc.add_heading("Valutazione delle ricette", level=1)

    question_counter = 1

    for i, couple in enumerate(couples_dataset_test):
        name1 = food_data_names[couple[0]].replace("-", " ").replace("à", "a'")
        name2 = food_data_names[couple[1]].replace("-", " ").replace("à", "a'")
        link1 = food_data_links[couple[0]]
        link2 = food_data_links[couple[1]]
        options = doc.add_paragraph()
        options.add_run(str(question_counter) + ". Cosa preferisci tra:\n")
        question_counter += 1
        options.add_run("A. " + name1 + " (" + str(link1) + ")" + "\n")
        options.add_run("B. " + name2 + " (" + str(link2) + ")" + "\n")
        options.add_run("C. indifferente")

    doc.save("./user" + str(user) + ".docx")
