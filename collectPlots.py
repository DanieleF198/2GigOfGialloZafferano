import os
import shutil

# little program to collect plots of all user in a unique folder
# now = datetime.now()
# year = now.strftime("%Y")
# month = now.strftime("%m")
# day = now.strftime("%d")
year = "2022"
month = "01"
day = "23"
data_dir = "./dataset_100/separated_text_data/"
input_layer = "tanh"
first_hidden_layer = "relu"
second_hidden_layer = "linear"
output_layer = "softmax"
lr_value = "0.0005"
opt = "SGD"
nodes = "64"
for dirname, _, filenames in os.walk('./NN_data/plots/'):
    for index, filename in enumerate(filenames):
        path = os.path.join(dirname, filename)
        if "accuracy" in path:
            user_number = int(path[path.find("User") + 4: path.find("\\accuracy")])
            for i in range(0, 48):
                if i == user_number:
                    if (i == 2) or (i == 5) or (i == 23):
                        if "tanh-relu-linear-Yes_Drop0.1_Yes_batch-softmax_LR-0.0005_OPT-SGD_Nodes-64" in path:
                            shutil.copyfile(path, "./NN_data/allPlots/accuracy/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
                    else:
                        shutil.copyfile(path, "./NN_data/allPlots/accuracy/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
        elif "loss" in path:
            user_number = int(path[path.find("User") + 4: path.find("\\loss")])
            for i in range(0, 48):
                if i == user_number:
                    if (i == 2) or (i == 5) or (i == 23):
                        if "tanh-relu-linear-Yes_Drop0.1_Yes_batch-softmax_LR-0.0005_OPT-SGD_Nodes-64" in path:
                            shutil.copyfile(path, "./NN_data/allPlots/loss/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
                    else:
                        shutil.copyfile(path, "./NN_data/allPlots/loss/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
        elif "precision" in path:
            user_number = int(path[path.find("User") + 4: path.find("\\precision")])
            for i in range(0, 48):
                if i == user_number:
                    if (i == 2) or (i == 5) or (i == 23):
                        if "tanh-relu-linear-Yes_Drop0.1_Yes_batch-softmax_LR-0.0005_OPT-SGD_Nodes-64" in path:
                            shutil.copyfile(path, "./NN_data/allPlots/precision/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
                    else:
                        shutil.copyfile(path, "./NN_data/allPlots/precision/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
        elif "recall" in path:
            user_number = int(path[path.find("User") + 4: path.find("\\recall")])
            for i in range(0, 48):
                if i == user_number:
                    if (i == 2) or (i == 5) or (i == 23):
                        if "tanh-relu-linear-Yes_Drop0.1_Yes_batch-softmax_LR-0.0005_OPT-SGD_Nodes-64" in path:
                            shutil.copyfile(path, "./NN_data/allPlots/recall/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
                    else:
                        shutil.copyfile(path, "./NN_data/allPlots/recall/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
        else:
            user_number = int(path[path.find("User") + 4: path.find("\\f1score")])
            for i in range(0, 48):
                if i == user_number:
                    if (i == 2) or (i == 5) or (i == 23):
                        if "tanh-relu-linear-Yes_Drop0.1_Yes_batch-softmax_LR-0.0005_OPT-SGD_Nodes-64" in path:
                            shutil.copyfile(path, "./NN_data/allPlots/f1score/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
                    else:
                        shutil.copyfile(path, "./NN_data/allPlots/f1score/User" + str(i) + "-" + input_layer + "-" + first_hidden_layer + "-" + second_hidden_layer + "-Yes_Drop0.1_Yes_batch-" + output_layer + "_LR-" + lr_value + "_OPT-" + opt + "_Nodes-" + nodes + ".jpg")
