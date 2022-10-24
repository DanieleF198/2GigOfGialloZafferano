import ilaspReadWriteUtils as ilasp
import time
from random import shuffle
from datetime import datetime
import os

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--user', type=int, default=1, help='User preferences.')
    # parser.add_argument('--prefers', type=int, default=-1, help='prefers')
    # parser.add_argument('--to', type=int, default=-1, help='to')
    # parser.add_argument('--mode', type=str, default='prediction', help='mode')
    # parser.add_argument('--data', type=str, default='USER', help='user')
    # args = parser.parse_args()

    # Open output and write preamble

    while True:
        print("insert max_p and max_v (integers, if 0 is passed then they won't be set)")
        max_p = input()
        max_v = input()

        if not max_p.isdigit() or not max_v.isdigit():
            print("input must be integers")
            continue
        if int(max_v) < 0 or int(max_p) < 0:
            print("input must be >= 0")
            continue
        if int(max_v) > 5 or int(max_p) > 5:
            print("input must be <= 5")
            continue
        break

    while True:
        print("choice --max-wc-length parameter (integer, passing 0 means set default value 3)")
        max_wc = input()

        if not max_wc.isdigit():
            print("input must be integers")
            continue
        if int(max_wc) < 0:
            print("input must be >= 0")
            continue
        if int(max_wc) == 0:
            max_wc = 3
        break

    while True:
        print("insert 0 if you want only relation of preference such that A>B or A<B, 1 if you want also case in which A=B")
        choice = input()

        if not choice.isdigit():
            print("input must be 0 or 1")
            continue
        if int(choice) < 0 or int(choice) > 1:
            print("input must be 0 or 1")
            continue
        break

    OUTPUT_PATH = "Output/output" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".csv"
    f_output = open(OUTPUT_PATH, "w")
    f_output.write("USERID;MAXV;MAXP;MAXWC;TRAIN_SIZE;TEST_SIZE;CORRECT;UNCERTAIN;INCORRECT;CORRECTP;UNCERTAINP;INCORRECTP;CORRECT_UDISCARDEDP;TRAIN_TIME;THEORY\n")

    if int(max_v) > 0 and int(max_p) > 0:
        items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
        language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(" + str(max_p) + ").las")
    elif int(max_v) > 0 or int(max_p) > 0:
        if int(max_v) > 0:
            items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
            language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(" + str(max_v) + ")-max_p(default).las")
        else:
            items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(default)-max_p(" + str(max_p) +").las")
            language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(default)-max_p(" + str(max_p) +").las")
    else:
        items = ilasp.itemsFromFile("Data/recipes/recipes_max_v(default)-max_p(default).las")
        language_bias = ilasp.languageBiasFromFile("Data/recipes/recipes_max_v(default)-max_p(default).las")

    # USERS = [str(i) for i in range(0,48)]
    USERS = [15, 3, 32, 7, 36, 4, 20, 29, 14, 11]
    # shuffle(USERS)

    # Main loop
    for USER in USERS:
        # loads preference data
        if int(choice) == 0:
            preferences = ilasp.preferencesFromFile("Data/users/no_zero/user" + str(USER) + ".txt")
        else:
            preferences = ilasp.preferencesFromFile("Data/users/zero/user" + str(USER) + ".txt")

        # Percentage of data used for training
        training_fraction = 0.7
        shuffle(preferences)
        training_set = preferences[0:round(len(preferences) * training_fraction)]
        test_set = preferences[round(len(preferences) * training_fraction):]

        MAXWC = int(max_wc)

        # Open temporary file
        TEMP_FILE = "Temp/temp.las"
        temp_file = open(TEMP_FILE, "w")

        # Write theory in temporary file
        temp_file.write(ilasp.itemsToPos(items) + "\n")
        temp_file.write(language_bias)
        temp_file.write(ilasp.preferencesToBraveOrderings(training_set))
        temp_file.flush()
        temp_file.close()

        # Train
        start_time = time.time()
        output_theory = ilasp.train(TEMP_FILE, options="--max-wc-length={}".format(MAXWC))
        ilasp_train_time = time.time() - start_time

        # Test
        results = ilasp.test(output_theory, items, test_set)
        print(results)
        print(output_theory)
        print(ilasp_train_time)

        f_output.write(USER + ";" + str(max_v) + str(max_p) + ";" + str(MAXWC) + ";" + str(len(training_set)) + ";" + str(len(test_set)) + ";" + str(results["correct"]) + ";" + str(results["uncertain"]) + ";" + str(results["incorrect"]) + ";" + str(results["correct"] / len(test_set)) + ";" + str(results["uncertain"] / len(test_set)) + ";" + str(results["incorrect"] / len(test_set)) + ";" + str(results["correct"] / (len(test_set) - results["uncertain"])) + ";" + str(ilasp_train_time) + ";" + output_theory.replace("\n", " ") + "\n")
        f_output.flush()

        # Delete temp file
        temp_file.close()
        if os.path.exists(TEMP_FILE):
            os.remove(TEMP_FILE)

        print("user" + str(USER) + " done")
    f_output.close()