import csv
import numpy as np

users = [10, 13, 23, 24, 25, 28, 42, 43]
first_description = np.zeros((len(users), 6), dtype="int32")
second_description = np.zeros((len(users), 6), dtype="int32")
third_description = np.zeros((len(users), 6), dtype="int32")
fourth_description = np.zeros((len(users), 7), dtype="int32")
final_questions = np.zeros((len(users), 2), dtype="int32")


for u_counter, user in enumerate(users):
    first_description_type = []
    second_description_type = []
    third_description_type = []
    fourth_description_type = []
    with open('./Answers_dataset/answers-of-return/' + str(user) + '.csv', newline='\n', encoding='utf-8') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=';', quotechar='|')
        for i_row, row in enumerate(csvReader):
            if i_row == 0:
                description = 0
                priority = 0
                exception = False
                for element in row:
                    if "priorità" in element:
                        if exception:
                            description += 1
                            exception = False
                        elif priority == 5:
                            description += 1
                            priority = 0
                        if description == 0:
                            first_description_type.append("p")
                            priority += 1
                        elif description == 1:
                            second_description_type.append("p")
                            priority += 1
                        elif description == 2:
                            third_description_type.append("p")
                            priority += 1
                        elif description == 3:
                            fourth_description_type.append("p")
                            priority += 1
                    elif "eccezione" in element:
                        exception = True
                        if description == 0:
                            first_description_type.append("e")
                        elif description == 1:
                            second_description_type.append("e")
                        elif description == 2:
                            third_description_type.append("e")
                        elif description == 3:
                            fourth_description_type.append("e")
                    elif "Ritieni" in element:
                        if description == 3:
                            fourth_description_type.append("q")
                            break
            else:
                insert_counter = 0
                first_description_counter = 0
                second_description_counter = 0
                third_description_counter = 0
                fourth_description_counter = 0
                for element_index, element in enumerate(row):
                    if element_index < 6:
                        continue
                    if insert_counter < len(first_description_type):
                        if first_description_type[first_description_counter] == "p":
                            if element == "sono d'accordo":
                                first_description[u_counter, 0] += 1
                                insert_counter += 1
                                first_description_counter += 1
                            elif element == "sono d'accordo ma gli do maggiore priorità":
                                first_description[u_counter, 1] += 1
                                insert_counter += 1
                                first_description_counter += 1
                            elif element == "sono d'accordo ma gli do minore priorità":
                                first_description[u_counter, 2] += 1
                                insert_counter += 1
                                first_description_counter += 1
                            elif element == "non sono d'accordo":
                                first_description[u_counter, 3] += 1
                                insert_counter += 1
                                first_description_counter += 1
                        elif first_description_type[first_description_counter] == "e":
                            if "non" in element:
                                first_description[u_counter, 5] += 1
                                insert_counter += 1
                                first_description_counter += 1
                            else:
                                first_description[u_counter, 4] += 1
                                insert_counter += 1
                                first_description_counter += 1
                    elif insert_counter < len(first_description_type) + len(second_description_type):
                        if second_description_type[second_description_counter] == "p":
                            if element == "sono d'accordo":
                                second_description[u_counter, 0] += 1
                                insert_counter += 1
                                second_description_counter += 1
                            elif element == "sono d'accordo ma gli do maggiore priorità":
                                second_description[u_counter, 1] += 1
                                insert_counter += 1
                                second_description_counter += 1
                            elif element == "sono d'accordo ma gli do minore priorità":
                                second_description[u_counter, 2] += 1
                                insert_counter += 1
                                second_description_counter += 1
                            elif element == "non sono d'accordo":
                                second_description[u_counter, 3] += 1
                                insert_counter += 1
                                second_description_counter += 1
                        elif second_description_type[second_description_counter] == "e":
                            if "non" in element:
                                second_description[u_counter, 5] += 1
                                insert_counter += 1
                                second_description_counter += 1
                            else:
                                second_description[u_counter, 4] += 1
                                insert_counter += 1
                                second_description_counter += 1
                    elif insert_counter < len(first_description_type) + len(second_description_type) + len(third_description_type):
                        if third_description_type[third_description_counter] == "p":
                            if element == "sono d'accordo":
                                third_description[u_counter, 0] += 1
                                insert_counter += 1
                                third_description_counter += 1
                            elif element == "sono d'accordo ma gli do maggiore priorità":
                                third_description[u_counter, 1] += 1
                                insert_counter += 1
                                third_description_counter += 1
                            elif element == "sono d'accordo ma gli do minore priorità":
                                third_description[u_counter, 2] += 1
                                insert_counter += 1
                                third_description_counter += 1
                            elif element == "non sono d'accordo":
                                third_description[u_counter, 3] += 1
                                insert_counter += 1
                                third_description_counter += 1
                        elif third_description_type[third_description_counter] == "e":
                            if "non" in element:
                                third_description[u_counter, 5] += 1
                                insert_counter += 1
                                third_description_counter += 1
                            else:
                                third_description[u_counter, 4] += 1
                                insert_counter += 1
                                third_description_counter += 1
                    elif insert_counter < len(first_description_type) + len(second_description_type) + len(third_description_type) + len(fourth_description_type):
                        if fourth_description_type[fourth_description_counter] == "p":
                            if element == "sono d'accordo":
                                fourth_description[u_counter, 0] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                            elif element == "sono d'accordo ma gli do maggiore priorità":
                                fourth_description[u_counter, 1] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                            elif element == "sono d'accordo ma gli do minore priorità":
                                fourth_description[u_counter, 2] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                            elif element == "non sono d'accordo":
                                fourth_description[u_counter, 3] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                        elif fourth_description_type[fourth_description_counter] == "e":
                            if "non" in element:
                                fourth_description[u_counter, 5] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                            else:
                                fourth_description[u_counter, 4] += 1
                                insert_counter += 1
                                fourth_description_counter += 1
                        elif fourth_description_type[fourth_description_counter] == "q":
                            fourth_description[u_counter, 6] = int(element)
                            insert_counter += 1
                            fourth_description_counter += 1
                    else:
                        if element_index == len(row) - 1:
                            final_questions[u_counter, 1] = int(element)
                        else:
                            final_questions[u_counter, 0] = int(element)
print("")
print("Total answers: " + str(len(users)))
print("")
print("First description - priorities (" + str(np.sum(first_description[:, 0:4])) + "):")
print("- I agree: " + str(np.sum(first_description[:, 0])))
print("- I agree but I give more priority to it: " + str(np.sum(first_description[:, 1])))
print("- I agree but I give less priority to it: " + str(np.sum(first_description[:, 2])))
print("- I don't agree: " + str(np.sum(first_description[:, 3])))
print("so the users, at average, agree regardless of the priority the " + str(np.round(((np.sum(first_description[:, 0]) + np.sum(first_description[:, 1]) + np.sum(first_description[:, 2]))/np.sum(first_description[:, 0:4]))*100, 2)) + "% of the times")
print("but, when they agree, they don't agree about priority order the " + str(np.round(((np.sum(first_description[:, 1]) + np.sum(first_description[:, 2]))/np.sum(first_description[:, 0:3]))*100, 2)) + "% of the times")
print("while don't agree at all the " + str(np.round(((np.sum(first_description[:, 3]))/np.sum(first_description[:, 0:4]))*100, 2)) + "% of the times")
print("")
print("First description - exceptions (" + str(np.sum(first_description[:, 4:])) + "):")
print("- I agree: " + str(np.sum(first_description[:, 4])) + " (" + str(np.round(((np.sum(first_description[:, 4]))/np.sum(first_description[:, 4:]))*100, 2)) + "%)")
print("- I don't agree: " + str(np.sum(first_description[:, 5])) + " (" + str(np.round(((np.sum(first_description[:, 5]))/np.sum(first_description[:, 4:]))*100, 2)) + "%)")
print("----------------------------------------------------------------------------------------------------------------------")
print("Second description - priorities (" + str(np.sum(second_description[:, 0:4])) + "):")
print("- I agree: " + str(np.sum(second_description[:, 0])))
print("- I agree but I give more priority to it: " + str(np.sum(second_description[:, 1])))
print("- I agree but I give less priority to it: " + str(np.sum(second_description[:, 2])))
print("- I don't agree: " + str(np.sum(second_description[:, 3])))
print("so the users, at average, agree regardless of the priority the " + str(np.round(((np.sum(second_description[:, 0]) + np.sum(second_description[:, 1]) + np.sum(second_description[:, 2]))/np.sum(second_description[:, 0:4]))*100, 2)) + "% of the times")
print("but, when they agree, they don't agree about priority order the " + str(np.round(((np.sum(second_description[:, 1]) + np.sum(second_description[:, 2]))/np.sum(second_description[:, 0:3]))*100, 2)) + "% of the times")
print("while don't agree at all the " + str(np.round(((np.sum(second_description[:, 3]))/np.sum(second_description[:, 0:4]))*100, 2)) + "% of the times")
print("")
print("Second description - exceptions (" + str(np.sum(second_description[:, 4:])) + "):")
print("- I agree: " + str(np.sum(second_description[:, 4])) + " (" + str(np.round(((np.sum(second_description[:, 4]))/np.sum(second_description[:, 4:]))*100, 2)) + "%)")
print("- I don't agree: " + str(np.sum(second_description[:, 5])) + " (" + str(np.round(((np.sum(second_description[:, 5]))/np.sum(second_description[:, 4:]))*100, 2)) + "%)")
print("----------------------------------------------------------------------------------------------------------------------")
print("Third description - priorities (" + str(np.sum(third_description[:, 0:4])) + "):")
print("- I agree: " + str(np.sum(third_description[:, 0])))
print("- I agree but I give more priority to it: " + str(np.sum(third_description[:, 1])))
print("- I agree but I give less priority to it: " + str(np.sum(third_description[:, 2])))
print("- I don't agree: " + str(np.sum(third_description[:, 3])))
print("so the users, at average, agree regardless of the priority the " + str(np.round(((np.sum(third_description[:, 0]) + np.sum(third_description[:, 1]) + np.sum(third_description[:, 2]))/np.sum(third_description[:, 0:4]))*100, 2)) + "% of the times")
print("but, when they agree, they don't agree about priority order the " + str(np.round(((np.sum(third_description[:, 1]) + np.sum(third_description[:, 2]))/np.sum(third_description[:, 0:3]))*100, 2)) + "% of the times")
print("while don't agree at all the " + str(np.round(((np.sum(third_description[:, 3]))/np.sum(third_description[:, 0:4]))*100, 2)) + "% of the times")
print("")
print("Third description - exceptions (" + str(np.sum(third_description[:, 4:])) + "):")
print("- I agree: " + str(np.sum(third_description[:, 4])) + " (" + str(np.round(((np.sum(third_description[:, 4]))/np.sum(third_description[:, 4:]))*100, 2)) + "%)")
print("- I don't agree: " + str(np.sum(third_description[:, 5])) + " (" + str(np.round(((np.sum(third_description[:, 5]))/np.sum(third_description[:, 4:]))*100, 2)) + "%)")
print("----------------------------------------------------------------------------------------------------------------------")
print("Fourth description - priorities (" + str(np.sum(fourth_description[:, 0:4])) + "):")
print("- I agree: " + str(np.sum(fourth_description[:, 0])))
print("- I agree but I give more priority to it: " + str(np.sum(fourth_description[:, 1])))
print("- I agree but I give less priority to it: " + str(np.sum(fourth_description[:, 2])))
print("- I don't agree: " + str(np.sum(fourth_description[:, 3])))
print("so the users, at average, agree regardless of the priority the " + str(np.round(((np.sum(fourth_description[:, 0]) + np.sum(fourth_description[:, 1]) + np.sum(fourth_description[:, 2]))/np.sum(fourth_description[:, 0:4]))*100, 2)) + "% of the times")
print("but, when they agree, they don't agree about priority order the " + str(np.round(((np.sum(fourth_description[:, 1]) + np.sum(fourth_description[:, 2]))/np.sum(fourth_description[:, 0:3]))*100, 2)) + "% of the times")
print("while don't agree at all the " + str(np.round(((np.sum(fourth_description[:, 3]))/np.sum(fourth_description[:, 0:4]))*100, 2)) + "% of the times")
print("")
print("Fourth description - exceptions (" + str(np.sum(fourth_description[:, 4:6])) + "):")
if np.sum(fourth_description[:, 4:6]) == 0:
    print("- sono d'accordo: 0 (0.0%)")
    print("- non sono d'accordo: 0 (0.0%)")
else:
    print("- I agree: " + str(np.sum(fourth_description[:, 4])) + " (" + str(np.round(((np.sum(fourth_description[:, 4])) / np.sum(fourth_description[:, 4:6])) * 100, 2)) + "%)")
    print("- I don't agree: " + str(np.sum(fourth_description[:, 5])) + " (" + str(np.round(((np.sum(fourth_description[:, 5])) / np.sum(fourth_description[:, 4:6])) * 100, 2)) + "%)")
print("")
print("pertinence rating: " + str(np.round(np.mean(fourth_description[:, 6]), 2)))
print("----------------------------------------------------------------------------------------------------------------------")
print("final questions: ")
print("")
print("- clarity: " + str(np.round(np.mean(final_questions[:, 0]), 2)))
print("- well written: " + str(np.round(np.mean(final_questions[:, 1]), 2)))
