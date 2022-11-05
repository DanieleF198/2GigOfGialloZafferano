# importing libraries

import csv

# importing data

listOfData = []

with open('./dataset_100/NewDataset.csv', newline='', encoding='utf-8') as csvFile:
    csvReader = csv.reader(csvFile, delimiter=';', quotechar='|')
    for row in csvReader:
        listOfData.append(list(row))

ingredientsIdPositions = []
preparationsIdPositions = []

for rowIndex, row in enumerate(listOfData):
    if rowIndex >= 1:
        break
    for index, element in enumerate(row):
        if (8 <= index <= 61) and (element == "Ingredient ID"):
            ingredientsIdPositions.append(index)
        if (62 <= index <= 76) and (element == "ID"):
            preparationsIdPositions.append(index)

ingredientsIds = []
preparationsIds = []

for rowIndex, row in enumerate(listOfData):
    if rowIndex == 0:
        continue
    for index, element in enumerate(row):
        if index in ingredientsIdPositions:
            if element not in ingredientsIds and element is not '':
                ingredientsIds.append(element)
        if index in preparationsIdPositions:
            if element not in preparationsIds and element is not '':
                preparationsIds.append(element)

# total number of ingredients (max granularity): 136
# total number of preparation: 8

ingredientsIds.sort(key=int)
preparationsIds.sort(key=int)

finalDataset = []
for index_row, row in enumerate(listOfData):
    if index_row == 0:
        continue
    row_of_dataset = []
    ID_of_food = None
    name_of_food = ""
    category_of_food = None
    cost_of_food = None
    difficulty_of_food = None
    preparation_time_of_food = None
    list_of_ingredients_of_food = []
    list_of_preparations_of_food = []
    for i in range(0, 136):
        list_of_ingredients_of_food.append(0)
    for i in range(0, 8):
        list_of_preparations_of_food.append(0)
    link_of_food = ""
    for index_element, element in enumerate(row):
        if index_element == 0:
            name_of_food = element
        if index_element == 1:
            ID_of_food = int(element)
        if index_element == 2:
            link_of_food = element
        if index_element == 3:
            continue
        if index_element == 4:
            category_of_food = int(element)
        if index_element == 5:
            cost_of_food = int(element)
        if index_element == 6:
            difficulty_of_food = int(element)
        if index_element == 7:
            preparation_time_of_food = int(element)
        if index_element in ingredientsIdPositions:
            if element == '':
                continue
            else:
                for i in range(0, 135):
                    if element == ingredientsIds[i]:
                        list_of_ingredients_of_food[i] = int(row[index_element+1])
        if index_element in preparationsIdPositions:
            if element == '':
                continue
            else:
                for i in range(0, 8):
                    if element == preparationsIds[i]:
                        list_of_preparations_of_food[i] = int(row[index_element + 1])
        if index_element >= 77:
            row_of_dataset.append(ID_of_food)
            row_of_dataset.append(name_of_food)
            row_of_dataset.append(category_of_food)
            row_of_dataset.append(cost_of_food)
            row_of_dataset.append(difficulty_of_food)
            row_of_dataset.append(preparation_time_of_food)
            row_of_dataset.append(list_of_ingredients_of_food)
            row_of_dataset.append(list_of_preparations_of_food)
            # for ingredient in list_of_ingredients_of_food:
            #     row_of_dataset.append(ingredient)
            # for preparation in list_of_preparations_of_food:
            #     row_of_dataset.append(preparation)
            row_of_dataset.append(link_of_food)
            finalDataset.append(row_of_dataset)
            break

for row in finalDataset:
    print(row)
