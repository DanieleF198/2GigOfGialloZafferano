import os

max_v_list = [1, 2, 3, 4, 5]
max_p_list = [1, 2, 3, 4, 5]
choices = [0, 1]

# for choice in choices:
    # USERS = [str(i) for i in range(0, 54)]
    # for USER in USERS:
    #     if int(choice) == 0:
    #         train_data_dir = "./Data8Component2Std/final_original/users/no_zero/train/150Couples/User" + str(USER) + "/"
    #         # test_data_dir = "./Data8Component2Std/final_original/users/no_zero/test/150Couples/User" + str(USER) + "/"
    #     else:
    #         train_data_dir = "./Data8Component2Std/final_original/users/zero/train/150Couples/User" + str(USER) + "/"
            # test_data_dir = "./Data8Component2Std/final_original/users/zero/test/150Couples/User" + str(USER) + "/"
        # fileToCreate = open(train_data_dir + "script_ilasp_commands_equals_and_less_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v > 3 or max_p > 3:
        #             continue
        #         if max_v == max_p:
        #             fileToCreate.write("ILASP --version=4 ./trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(train_data_dir + "script_ilasp_commands_not_equals_and_less_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v > 3 or max_p > 3:
        #             continue
        #         if max_v != max_p:
        #             fileToCreate.write("ILASP --version=4 ./trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(train_data_dir + "script_ilasp_commands_equals_and_great_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v <= 3 or max_p <= 3:
        #             continue
        #         if max_v == max_p:
        #             fileToCreate.write("ILASP --version=4 ./trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(train_data_dir + "script_ilasp_commands_not_equals_and_great_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v <= 3 or max_p <= 3:
        #             continue
        #         if max_v != max_p:
        #             fileToCreate.write("ILASP --version=4 ./trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(train_data_dir + "script_ilasp_commands_all.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         fileToCreate.write("ILASP --version=4 ./trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()


        # fileToCreate = open(test_data_dir + "script_ilasp_commands_equals_and_less_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v > 3 or max_p > 3:
        #             continue
        #         if max_v == max_p:
        #             fileToCreate.write("ILASP --version=4 ./testFiles/test_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTest/outputTest_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(test_data_dir + "script_ilasp_commands_not_equals_and_less_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v > 3 or max_p > 3:
        #             continue
        #         if max_v != max_p:
        #             fileToCreate.write("ILASP --version=4 ./testFiles/test_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTest/outputTest_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(test_data_dir + "script_ilasp_commands_equals_and_great_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v <= 3 or max_p <= 3:
        #             continue
        #         if max_v == max_p:
        #             fileToCreate.write("ILASP --version=4 ./testFiles/test_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTest/outputTest_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
        # fileToCreate.flush()
        # fileToCreate.close()
        # fileToCreate = open(test_data_dir + "script_ilasp_commands_not_equals_and_great_three.sh", "w+")
        # for max_v in max_v_list:
        #     for max_p in max_p_list:
        #         if max_v <= 3 or max_p <= 3:
        #             continue
        #         if max_v != max_p:
        #             fileToCreate.write("ILASP --version=4 ./testFiles/test_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./outputTest/outputTest_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")

second_list_of_user = [0, 10, 12, 13, 19, 21, 22, 23, 24, 25, 26, 28, 31, 41, 42, 43, 44] # 15, 3, 32, 7, 36, 4, 20, 29, 14, 11
# second_list_of_user = [str(i) for i in range(0, 54)]
no_zero_data_dir = "./Data17Component2Std/final/users/no_zero/train/210Couples/"
zero_data_dir = "./Data17Component2Std/final/users/zero/train/210Couples/"
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_three.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 3:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_four.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 4:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_five.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 5:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_three.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 3:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_four.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 4:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_five.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 5:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_three.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 3:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_four.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 4:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_until_five.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 5:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_three.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 3:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_four.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 4:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_commands_equals_only_five.sh", "w+")
for USER in second_list_of_user:
    for max_v in max_v_list:
        for max_p in max_p_list:
            if max_v == max_p:
                if max_v == 5:
                    fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
fileToCreate.flush()
fileToCreate.close()
#
# # version maxp = 2, maxv from 1 to 10
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_test_users_all_combination.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 continue
#             fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
# #
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_all_combination.sh", "w+")
# for USER in second_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 continue
#             fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()

# HERE
# fileToCreate = open(zero_data_dir + "script_ilasp_for_test_users_on_founded_max_v_max_p.sh", "w+")
# for USER in second_list_of_user:
#     if USER in ['15', '3', '32', '7', '36', '4', '20', '29', '14', '11']:
#         continue
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == 3 and max_p == 5:
#                 fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()

# other_list_of_user = [x for x in range(0, 48)]
# for USER in other_list_of_user[::-1]:
#     if USER in second_list_of_user:
#         other_list_of_user.remove(USER)
#
# no_zero_data_dir = "./PCAexperiment/final_original/users/no_zero/train/150Couples/"
# zero_data_dir = "./PCAexperiment/final_original/users/zero/train/150Couples/"
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_three.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 3:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_four.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 4:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_five.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 5:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_three.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 3:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_four.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 4:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(no_zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_five.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 5:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_three.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 3:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_four.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 4:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_until_five.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v > 5:
#                     continue
#                 else:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_three.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 3:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_four.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 4:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()
#
# fileToCreate = open(zero_data_dir + "script_ilasp_for_other_users_commands_equals_only_five.sh", "w+")
# for USER in other_list_of_user:
#     for max_v in max_v_list:
#         for max_p in max_p_list:
#             if max_v == max_p:
#                 if max_v == 5:
#                     fileToCreate.write("ILASP --version=4 ./User" + str(USER) + "/trainFiles/train_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).las > ./User" + str(USER) + "/outputTrain/outputTrain_max-v\(" + str(max_v) + "\)-max_p\(" + str(max_p) + "\).txt;\n")
# fileToCreate.flush()
# fileToCreate.close()