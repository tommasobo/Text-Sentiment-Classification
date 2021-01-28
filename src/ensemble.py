from pathlib import Path
import numpy
from collections import Counter
import csv
import argparse
import datetime

import constants

'''
    Run the Ensemble Model by using submission files. Each file gives the same weighted contribution. If there is an even number
    of models, it is possible that the value will be chosen randomly if half of the models say A while the rest B.
    Finally the results and a debugging file are saved.
'''
def run_ensemble():
    # Print alert to the user
    print(f"\nPrediction files inside {constants.ENSEMBLE_FOLDER} will be used to run the ensemble.")
    constants.ENSEMBLE_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Cycle each file inside ensemble folder
    num_csv_files = len(list(Path(constants.ENSEMBLE_FOLDER).glob('*.csv')))
    if num_csv_files == 0:
        print("No .csv files detected, please add them manually")
        return

    # Load predictions from all files into a matrix
    result_matrix = list()
    input_files_list = list()
    for prediction_csv in Path(constants.ENSEMBLE_FOLDER).glob('*.csv'):
        print(f"Processing {prediction_csv}")
        input_files_list.append(prediction_csv)

        # Read predictions, remove header, remove empty lines and get only predictions values
        predictions = prediction_csv.read_text(encoding = "utf8").split("\n")
        del(predictions[0])  
        predictions = [sing_pred[sing_pred.find(",") + 1:] for sing_pred in predictions]
        predictions = list(filter(None, predictions))
        result_matrix.append(predictions)

    # Compute most agreed element and save it to file (majority voting part)
    output_list = list()
    for i in range (constants.NUM_TEST_SAMPLES):
        single_prediction_models = list()
        for j in range (num_csv_files):
            single_prediction_models.append(result_matrix[j][i])

        most_common, count = Counter(single_prediction_models).most_common(1)[0]
        output_list.append(most_common)

    # Saving results as csv and saving debugging info
    print(f"\nSaving results to {constants.ENSEMBLE_FOLDER_SUBMISSION}")
    constants.ENSEMBLE_FOLDER_SUBMISSION.mkdir(parents=True, exist_ok=True)
    write_to_file(output_list, constants.ENSEMBLE_FOLDER_SUBMISSION / "latest_ensemble_submission.csv")
    filename_sub = str(datetime.datetime.utcnow()) + "_ensemble_submission.csv"
    filename_debug =  str(datetime.datetime.utcnow()) + "_ensemble_info.txt"
    write_to_file(output_list, constants.ENSEMBLE_FOLDER_SUBMISSION / filename_sub)
    write_debug_file(input_files_list, constants.ENSEMBLE_FOLDER_SUBMISSION / filename_debug)


'''
    Validate the ensemble using a validation set.
'''
def validate_ensemble():

    # Print alert to the user
    print(f"\nPrediction files inside {constants.VALIDATE_ENSEMBLE} will be used to validate the ensemble.")
    constants.ENSEMBLE_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Cycle each file inside folder
    num_csv_files = len(list(Path(constants.VALIDATE_ENSEMBLE).glob('*.csv')))
    if num_csv_files == 0:
        print("No .csv files detected, please add them manually")
        return

    # Load predictions from all files into a matrix
    input_files_list = list()
    for prediction_csv in Path(constants.VALIDATE_ENSEMBLE).glob('*.csv'):
        print(f"Processing {prediction_csv}")
        input_files_list.append(prediction_csv)

        #Read labels
        labels_val = read_from_file(constants.INPUT_DATA_FOLDER / "labels")
        

        # Read predictions, remove header, remove empty lines and get only predictions values
        predictions = prediction_csv.read_text(encoding = "utf8").split("\n")
        del(predictions[0])  
        predictions = [sing_pred[sing_pred.find(",") + 1:] for sing_pred in predictions]
        predictions = list(filter(None, predictions))

        print(len(labels_val))
        print(len(predictions))

        print("ElmoAfter Split1:" + str(predictions[10]))
        print("ElmoAfter Split2:" + str(predictions[10000]))
        print("After Split3:" + str(predictions[11]))
        print("After Split4:" + str(predictions[10001]))

        print("Split1:" + str(labels_val[10]))
        print("Split2:" + str(labels_val[10000]))
        print("Split3:" + str(labels_val[11]))
        print("Split4:" + str(labels_val[10001]))

        # Computing matches
        match = 0
        for index, ele in enumerate(predictions):
            if (int(labels_val[index + 1]) == 0):
                labels_val[index + 1] = -1

            if (int(labels_val[index + 1]) == int(predictions[index])):
                match += 1

    # Printing result
    perc_acc = match / len(predictions) * 100
    print(f"Percentage of accurate predictions: {perc_acc} ({match} of {len(predictions)} predictions)")


'''
    Writes to file the predictions made. Also builds the right header and structure.
    Args:
        prediction: list of predictions.
        filename: name to use to save the file.
'''
def write_to_file(prediction, filename):
    with open(filename, 'w', newline='') as file:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        i = 1
        for elem in prediction:
            if (float(elem) >= 0.5):
                writer.writerow({'Id': i, 'Prediction': "1"})
            else:
                writer.writerow({'Id': i, 'Prediction': "-1"})
            i += 1


'''
    Writes to file some debugging info about the models used to generate the ensemble.
    Args:
        debug_info: list of models used.
        filename: name to use to save the file.
'''
def write_debug_file(debug_info, filename):
    with open(filename, 'w', newline='') as f:
        f.write("Files used to generate ensemble with filename %s\n" % filename)
        for item in debug_info:
            f.write("%s\n" % item)


def main():
    parser = argparse.ArgumentParser(description='Run Ensemble')
    parser.add_argument("--validate", action='store_true', help="Validate Results produced by Ensemble")
    args = parser.parse_args()

    if (args.validate):
        print("Validating the model")
        validate_ensemble()
        exit()

    run_ensemble()


if __name__ == "__main__":
    main()