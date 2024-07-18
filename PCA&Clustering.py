# importing of libraries

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
from numpy.random._generator import default_rng
from sklearn.decomposition import PCA
from sklearn import preprocessing, decomposition
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import norm

# importing of data
# in scalars.txt there is only preparation time
# in categories.txt there are category, cost and difficulty, but then cost and difficulty will be treated as scalars
# in ingredients there are the ingredients in a one-hot-encoding array
# in preparation there are the preparations in a one-hot-encoding array
# in names there are the name of foods.

data_dir = "./dataset_100/separated_text_data/"
fScalar = os.path.join(data_dir, 'scalars.txt')
fCategories = os.path.join(data_dir, 'categories.txt')
fIngredients = os.path.join(data_dir, 'ingredients.txt')
fPreparation = os.path.join(data_dir, 'preparations.txt')
fnames = os.path.join(data_dir, 'names.txt')
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
fN = open(fnames)
dataN = fN.read()
fN.close()

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

linesOfP = dataP.split('\n')
food_data_preparation = np.zeros((len(linesOfP), 8), dtype='float32')
for i, line in enumerate(linesOfP):
    values = [x for x in line.split(' ')[1:]]
    food_data_preparation[i, :] = values

linesOfN = dataN.split('\n')
food_data_names = ["" for x in range(101)]
for i, line in enumerate(linesOfN):
    food_data_names[i] = line

# manipulate categorical data
category1HE = food_data_categories[:, 0]
food_data_categories = np.delete(food_data_categories, 0, 1)
category1HE = np.reshape(category1HE, (101, 1))
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(category1HE)
finalCategory = enc.transform(category1HE).toarray()

# remove ingredients never used
food_data_ingredients = np.delete(food_data_ingredients, [1, 16, 23, 26, 32], axis=1)

# normalize scaled ingredients and scaled preparation

normalized_food_data_ingredients = np.zeros((101, 31), dtype="float32")
normalized_food_data_preparation = np.zeros((101, 8), dtype="float32")

for i, row in enumerate(food_data_ingredients):
    sum_of_elements = 0
    for element in row:
        sum_of_elements = sum_of_elements + element
    for j, element in enumerate(row):
        normalized_food_data_ingredients[i, j] = food_data_ingredients[i, j]/sum_of_elements

for i, row in enumerate(food_data_preparation):
    sum_of_elements = 0
    for element in row:
        sum_of_elements = sum_of_elements + element
    for j, element in enumerate(row):
        normalized_food_data_preparation[i, j] = food_data_preparation[i, j]/sum_of_elements

# scaling preparation time, cost, difficulty, ingredients and preparations
scaler = preprocessing.StandardScaler()
scaled_food_data_finalCategory = scaler.fit_transform(finalCategory)
scaled_food_data_scalars = scaler.fit_transform(food_data_scalars)
scaled_food_data_categories = scaler.fit_transform(food_data_categories)
scaled_food_data_ingredients = scaler.fit_transform(normalized_food_data_ingredients)
scaled_food_data_preparation = scaler.fit_transform(normalized_food_data_preparation)

# concatenate all data in a numpy tensor
all_data = np.concatenate([scaled_food_data_finalCategory, scaled_food_data_categories, scaled_food_data_scalars, scaled_food_data_ingredients, scaled_food_data_preparation], axis=1)

# preparation of labels for pandas dataframe
foodsLabel = ['F' + str(i) for i in range(1, 101)]
categoryLabel = ['CATEGORY' + str(i) for i in range(1, 6)]
ingredientsLabel = ['INGREDIENTS' + str(i) for i in range(1, 32)]
preparationsLabel = ['PREPARATIONS' + str(i) for i in range(1, 9)]

# creating pandas dataframe
final_data = pd.DataFrame(columns=[*categoryLabel, 'COST', 'DIFFICULTY', 'PREPARATION', *ingredientsLabel, *preparationsLabel], index=foodsLabel)

# fill pandas dataframe with data that we concatenated in line 83
for i, food in enumerate(final_data.index):
    final_data.loc[food] = all_data[i]

# instantiate pca object, then fit on final data
pca = PCA()  # (we used this before to understand the number of component to use)
# pca = decomposition.PCA(n_components=17)
pca.fit(final_data)

# apply pca on data in pandas_dataframe
pca_data = pca.transform(final_data)

# COMMENTED TO NOT OVERWRITE

# prepare output files
# pca_dir = "./PCA_data/"
# fnameOutput = os.path.join(pca_dir, 'output.txt')
# fnameOutput2 = os.path.join(pca_dir, 'output2.txt')
# # print eigenvalue on first output file:
# f = open(fnameOutput)
# sys.stdout = open(fnameOutput, 'w')
# print("EigenValues")
# for i, element in enumerate(pca.explained_variance_):
#     j = i + 1
#     if j > 9:
#         print("PC" + str(j) + ": " + str(element))
#     else:
#         print("PC" + str(j) + " : " + str(element))
# sys.stdout = sys.__stdout__
# f.close()
#
# temp = 0
# for element in pca.explained_variance_:
#     temp += element
# temp /= 101
#
# temp2 = 0
# for i, element in enumerate(pca.explained_variance_ratio_):
#     temp2 += element
#     print("PC" + str(i+1) + ": " + str(temp2) + ".")

# plotting matrix principal component x feature where M[i, j] is the weight of feature j on PC i
# matrixToPlot = np.zeros((len(pca.components_), 47))
# for i, row in enumerate(pca.components_):
#     matrixToPlot[i, :] = abs(row)
# plt.matshow(matrixToPlot)
# plt.show()

# plotting PC with relative percentage of explained variance in a bar plots
per_var = np.round(pca.explained_variance_ratio_*101, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

# COMMENTED TO NOT OVERWRITE

# plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
# plt.ylabel('percentage of explained variance')
# plt.xlabel('principal component')
# plt.title('scree plots')
# plt.show()


feature_weights = pca.components_
feature_weights = abs(feature_weights)
labelsFeature = [str(i) for i in range(1, len(feature_weights[0])+1)]

# COMMENTED TO NOT OVERWRITE

plot_ordered_weight_dir = "PCA_data/plots/weight_distr/"
for i in range(0, 47):
    data = sorted(feature_weights[i])
    mu = np.mean(data)
    sigma = np.std(data)
    sum = mu+sigma
    sum2 = mu+(sigma*2)
    fit = norm.pdf(data, mu, sigma)
    pl.hist(data, density=True, edgecolor="black", label='distribution')
    # pl.plot(data, fit, '-o', color="orange", label='standardized distribution')
    plt.axvline(x=sum, color="green", label='mean+std')
    plt.axvline(x=sum2, color="red", label='mean+2*(std)')
    plt.ylabel('frequency')
    plt.xlabel('weight value')
    plt.title('distribution of weights in PC' + str(i+1))
    plt.legend(loc="upper right")
    plt.savefig(plot_ordered_weight_dir + "PC" + str(i) + ".png", dpi=300)
    plt.clf()
    # data = sorted(feature_weights[i])
    # fit = norm.pdf(data, np.mean(data), np.std(data))
    # pl.plot(data, fit, '-o')
    # pl.hist(data, density=True)
    # pl.show()
# plot_weight_dir = "./PCA_data/plots/weight/"
# for i in range(0, 47):
#     plt.bar(x=range(1, len(feature_weights[i])+1), height=feature_weights[i], tick_label=labelsFeature)
#     plt.ylabel('weight on feature')
#     plt.xlabel('features')
#     plt.title('scree plots')
#     plt.savefig(plot_weight_dir + "PC" + str(i) + ".png", dpi=300)
#     plt.clf()
#
#
# # output matrix component x feature in third output file
# f = open(fnameOutput2)
# sys.stdout = open(fnameOutput2, 'w')
# for line in feature_weights:
#     newLine = np.around(line, 4)
#     print(*newLine)
# sys.stdout = sys.__stdout__
# f.close()

# # extracting feature with weight greater than the mean + std of the weight of the same component for the first 17 component
# interested_feature_one_std_dir = "./PCA_data/InterestedFeature/one_std/"
# interested_feature_two_std_dir = "./PCA_data/InterestedFeature/two_std/"
#
# for i, line in enumerate(feature_weights):
#     list_one_std = {}
#     list_two_std = {}
#     mean_line = np.mean(line)
#     std_line = np.std(line)
#     count_one = 0
#     count_two = 0
#     for j, element in enumerate(line):
#         if element > mean_line + std_line:
#             list_one_std[count_one] = [j, element]
#             count_one = count_one + 1
#         if element > mean_line + (2*std_line):
#             list_two_std[count_two] = [j, element]
#             count_two = count_two + 1
#     sorted_features_one = dict(sorted(list_one_std.items(), key=lambda item: item[1][1], reverse=True))
#     sorted_features_two = dict(sorted(list_two_std.items(), key=lambda item: item[1][1], reverse=True))
#     f_one_std = os.path.join(interested_feature_one_std_dir, 'feature' + str(i+1) + '.txt')     # cause i start from 0 and i want index that start from 1.
#     f_two_std = os.path.join(interested_feature_two_std_dir, 'feature' + str(i+1) + '.txt')
#     f = open(f_one_std, "w")
#     sys.stdout = open(f_one_std, 'w')
#     print("mean = " + str(mean_line) + ", std = " + str(std_line) + ". So feature with weight greater than " + str(mean_line + std_line))
#     for element in sorted_features_one.items():
#         print("feature: " + final_data.columns[element[1][0]] + " with value = " + str(element[1][1]))
#     sys.stdout = sys.__stdout__
#     f.close()
#     f = open(f_two_std, "w")
#     sys.stdout = open(f_two_std, 'w')
#     print("mean = " + str(mean_line) + ", std = " + str(std_line) + ". So feature with weight greater than " + str(mean_line + (std_line*2)))
#     for element in sorted_features_two.items():
#         print("feature: " + final_data.columns[element[1][0]] + " with value = " + str(element[1][1]))
#     sys.stdout = sys.__stdout__
#     f.close()



# preparing dataframe for plotting result in n differents 2D graph
pca_df = pd.DataFrame(pca_data, columns=labels)

# COMMENTED TO NOT OVERWRITE

# plot_labeled_dir = "./PCA_data/plots/labeled/"
# plot_no_labeled_dir = "./PCA_data/plots/no-labeled/"
#
#
# # plotting the 2D graph mentioned in line 185
# plt.scatter(pca_df.PC1, pca_df.PC2)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC2.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC3)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC3.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC1 PC8.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC3)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC3.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC2 PC8.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC3 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC3 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC3 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC3 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_no_labeled_dir + "PC3 PC8.png", dpi=300)
# plt.clf()
#
# # also plotted with label
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC2)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC2.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC3.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC3)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC3.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC4.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC4.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC5.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC5.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC6.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC6.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC7.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC7.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC8.loc[sample]))
#
# plt.scatter(pca_df.PC1, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC1 PC8.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC3.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC3)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC3.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC4.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC4.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC5.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC5.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC6.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC6.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC7.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC7.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC2.loc[sample], pca_df.PC8.loc[sample]))
#
# plt.scatter(pca_df.PC2, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC2 PC8.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC4.loc[sample]))
#
# plt.scatter(pca_df.PC3, pca_df.PC4)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC3 PC4.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC5.loc[sample]))
#
# plt.scatter(pca_df.PC3, pca_df.PC5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC3 PC5.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC6.loc[sample]))
#
# plt.scatter(pca_df.PC3, pca_df.PC6)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC3 PC6.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC7.loc[sample]))
#
# plt.scatter(pca_df.PC3, pca_df.PC7)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC3 PC7.png", dpi=300)
# plt.clf()
#
# for sample in pca_df.index:
#     plt.annotate(sample, (pca_df.PC3.loc[sample], pca_df.PC8.loc[sample]))
#
# plt.scatter(pca_df.PC3, pca_df.PC8)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig(plot_labeled_dir + "PC3 PC8.png", dpi=300)
# plt.clf()

# in the next 5 lines I store in an array the sum of square within cluster of k-means with k going from 1 to 20
# we do this to plots the result inside array and understanding which k is better using elbow method (we also use
# other criteria such us the cardinality of the clusters)

cluster_dir = "./cluster_data/"
wcss = []
for i in range(1, 21):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(pca_data)
    wcss.append(kmeans_pca.inertia_)

# COMMENTED TO NOT OVERWRITE

# plots mentioned in line 531
# plt.plots(range(1, 21), wcss, marker='o', linestyle='--')
# plt.xlabel('Number of cluster')
# plt.ylabel('WCSS - within cluster sum of squares')
# plt.title('K-means with PCA clustering')
# plt.show()

# plotting clustering for k = 2, 3, 4, 5 and 20
for i in range(2, 6):
    kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_pca.fit(pca_data)
    kmeans_predict = kmeans_pca.predict(pca_data)

# COMMENTED TO NOT OVERWRITE

#     plt.scatter(pca_df.PC1, pca_df.PC2, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC2.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC3, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC3.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC4.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC5.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC6.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC7.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC1, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
#     plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC1 PC8.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC3, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC3.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC4.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC5.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC6.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC7.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC2, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
#     plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC2 PC8.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC3, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC3 PC4.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC3, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC3 PC5.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC3, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC3 PC6.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC3, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC3 PC7.png", dpi=300)
#     plt.clf()
#
#     plt.scatter(pca_df.PC3, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
#     centers = kmeans_pca.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
#     plt.title('PCA graph')
#     plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
#     plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
#     plt.savefig(cluster_dir + str(i) + "-cluster/PC3 PC8.png", dpi=300)
#     plt.clf()
#
# kmeans_pca = KMeans(n_clusters=20, init='k-means++', random_state=42)
# kmeans_pca.fit(pca_data)
# kmeans_predict = kmeans_pca.predict(pca_data)
#
# plt.scatter(pca_df.PC1, pca_df.PC2, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC2.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC3, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC3.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC1, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC1 - {:.1f}%'.format(per_var[0]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.savefig(cluster_dir + "20-cluster/PC1 PC8.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC3, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC3.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC2, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC2 - {:.1f}%'.format(per_var[1]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.savefig(cluster_dir + "20-cluster/PC2 PC8.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC4, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC4 - {:.1f}%'.format(per_var[3]))
# plt.savefig(cluster_dir + "20-cluster/PC3 PC4.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC5, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC5 - {:.1f}%'.format(per_var[4]))
# plt.savefig(cluster_dir + "20-cluster/PC3 PC5.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC6, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC6 - {:.1f}%'.format(per_var[5]))
# plt.savefig(cluster_dir + "20-cluster/PC3 PC6.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC7, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC7 - {:.1f}%'.format(per_var[6]))
# plt.savefig(cluster_dir + "20-cluster/PC3 PC7.png", dpi=300)
# plt.clf()
#
# plt.scatter(pca_df.PC3, pca_df.PC8, c=kmeans_predict, s=50, cmap='viridis')
# centers = kmeans_pca.cluster_centers_
# plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
# plt.title('PCA graph')
# plt.xlabel('PC3 - {:.1f}%'.format(per_var[2]))
# plt.ylabel('PC8 - {:.1f}%'.format(per_var[7]))
# plt.savefig(cluster_dir + "20-cluster/PC3 PC8.png", dpi=300)
# plt.clf()

# print set of food that we'll use in survey (they are printed in order to make less intersection as possible)

kmeans_pca = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans_pca.fit(pca_data)
kmeans_predict = kmeans_pca.predict(pca_data)

occOne = 0
occTwo = 0
occThree = 0

for element in kmeans_pca.labels_:
    if element == 0:
        occOne = occOne + 1
    elif element == 1:
        occTwo = occTwo + 1
    elif element == 2:
        occThree = occThree + 1


clusterOne = np.zeros(occOne, dtype='float32')
clusterTwo = np.zeros(occTwo, dtype='float32')
clusterThree = np.zeros(occThree, dtype='float32')

one = 0
two = 0
three = 0

# da prendere randomicamente con reimmissione tra i diversi sondaggi (ma senza nello stesso sondaggio)

for i, element in enumerate(kmeans_pca.labels_):
    if element == 0:
        clusterOne[one] = i
        one = one + 1
    if element == 1:
        clusterTwo[two] = i
        two = two + 1
    if element == 2:
        clusterThree[three] = i
        three = three + 1

for i in range(0, 10):
    print("set" + str(i))
    
    listFirstCluster = {}
    listSecondCluster = {}
    listThirdCluster = {}

    random_food_first_cluster = np.random.choice(clusterOne, 7, replace=False)
    random_food_second_cluster = np.random.choice(clusterTwo, 7, replace=False)
    random_food_third_cluster = np.random.choice(clusterThree, 7, replace=False)

    print("from first cluster:", end='')
    for index, j in enumerate(random_food_first_cluster):
        if index == 6:
            print(food_data_names[int(j)] + ".")
            listFirstCluster[index] = food_data_names[int(j)]
        else:
            print(food_data_names[int(j)] + ",", end='')
            listFirstCluster[index] = food_data_names[int(j)]
                

    print("from second cluster:", end='')
    for index, j in enumerate(random_food_second_cluster):
        if index == 6:
            print(food_data_names[int(j)] + ".")
            listSecondCluster[index] = food_data_names[int(j)]
        else:
            print(food_data_names[int(j)] + ", ", end='')
            listSecondCluster[index] = food_data_names[int(j)]

    print("from third cluster:", end='')
    for index, j in enumerate(random_food_third_cluster):
        if index == 6:
            print(food_data_names[int(j)] + ".")
            listThirdCluster[index] = food_data_names[int(j)]
        else:
            print(food_data_names[int(j)] + ", ", end='')
            listThirdCluster[index] = food_data_names[int(j)]

    X_food = ""
    A_food = ""
    B_food = ""

    Y_food = ""
    C_food = ""
    D_food = ""

    Z_food = ""
    E_food = ""
    F_food = ""

    counter_first_cluster = 0
    counter_second_cluster = 0
    counter_third_cluster = 0

    print("")
    
    for j in range(0, 3):
        print(str(j+1) + " subset to order: ", end='')
        if j == 0:
            for first_cluster_first_subset in range(0, 3):
                if first_cluster_first_subset == 0:
                    X_food = listFirstCluster[counter_first_cluster]
                    print(X_food + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
                elif first_cluster_first_subset == 1:
                    print(listFirstCluster[counter_first_cluster] + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
                else:
                    B_food = listFirstCluster[counter_first_cluster]
                    print(B_food + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
            for second_cluster_first_subset in range(0, 3):
                if second_cluster_first_subset == 0:
                    Y_food = listSecondCluster[counter_second_cluster]
                    print(Y_food + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
                elif second_cluster_first_subset == 1:
                    print(listSecondCluster[counter_second_cluster] + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
                else:
                    D_food = listSecondCluster[counter_second_cluster]
                    print(D_food + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
            for third_cluster_first_subset in range(0, 4):
                if third_cluster_first_subset == 0:
                    print(listThirdCluster[counter_third_cluster] + ", ", end='')
                    counter_third_cluster = counter_third_cluster + 1
                elif third_cluster_first_subset == 1:
                    Z_food = listThirdCluster[counter_third_cluster]
                    print(Z_food + ", ", end='')
                    counter_third_cluster = counter_third_cluster + 1
                elif third_cluster_first_subset == 2:
                    print(listThirdCluster[counter_third_cluster] + ", ", end='')
                    counter_third_cluster = counter_third_cluster + 1
                else:
                    F_food = listThirdCluster[counter_third_cluster]
                    print(F_food + ".")
                    counter_third_cluster = counter_third_cluster + 1
        elif j == 1:
            for first_cluster_second_subset in range(0, 3):
                if first_cluster_second_subset == 0:
                    print(X_food + ", ", end='')
                elif first_cluster_second_subset == 1:
                    A_food = listFirstCluster[counter_first_cluster]
                    print(A_food + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
                else:
                    print(listFirstCluster[counter_first_cluster] + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
            for second_cluster_second_subset in range(0, 4):
                if second_cluster_second_subset == 0:
                    print(Y_food + ", ", end='')
                elif second_cluster_second_subset == 1:
                    C_food = listSecondCluster[counter_second_cluster]
                    print(C_food + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
                else:
                    print(listSecondCluster[counter_second_cluster] + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
            for third_cluster_second_subset in range(0, 3):
                if third_cluster_second_subset == 0:
                    print(Z_food + ", ", end='')
                elif third_cluster_second_subset == 1:
                    E_food = listThirdCluster[counter_third_cluster]
                    print(E_food + ", ", end='')
                    counter_third_cluster = counter_third_cluster + 1
                else:
                    print(listThirdCluster[counter_third_cluster] + ".")
                    counter_third_cluster = counter_third_cluster + 1
        else:
            for first_cluster_third_subset in range(0, 4):
                if first_cluster_third_subset == 0 or first_cluster_third_subset == 3:
                    print(listFirstCluster[counter_first_cluster] + ", ", end='')
                    counter_first_cluster = counter_first_cluster + 1
                elif first_cluster_third_subset == 1:
                    print(A_food + ", ", end='')
                elif first_cluster_third_subset == 2:
                    print(B_food + ", ", end='')
            for second_cluster_third_subset in range(0, 3):
                if second_cluster_third_subset == 0:
                    print(C_food + ", ", end='')
                elif second_cluster_third_subset == 1:
                    print(D_food + ", ", end='')
                else:
                    print(listSecondCluster[counter_second_cluster] + ", ", end='')
                    counter_second_cluster = counter_second_cluster + 1
            for third_cluster_third_subset in range(0, 3):
                if third_cluster_third_subset == 0:
                    print(listThirdCluster[counter_third_cluster] + ", ", end='')
                    counter_third_cluster = counter_third_cluster + 1
                elif third_cluster_third_subset == 1:
                    print(E_food + ", ", end='')
                else:
                    print(F_food + ".")
    print("")
