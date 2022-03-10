from datetime import datetime
from os.path import join as path_join, basename
from glob import glob
from sklearn.linear_model import LogisticRegression


from tqdm import tqdm
import numpy as np
from joblib import dump as model_dump

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

data_dir = "data"
results_dir = "results"
models_dir = "models"


def train_model(model, cv, label):
    run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

    output = open(path_join("results", "%s.csv" % label), "w")
    output.write(
        "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
    )

    all_train_accuracies = []
    all_test_accuracies = []

    for folder in glob(path_join(data_dir, "*")):
        crop_type = basename(folder)
        crop_accuracy = {}

        csvs = glob(path_join(folder, "**", "*.csv"), recursive=True)
        for csv in tqdm(csvs, desc="Training with %s" % crop_type):
            *x, y = np.loadtxt(
                csv,
                delimiter=",",
                skiprows=1,
                dtype={
                    "names": ["R", "G", "B", "l", "a", "b", "Class"],
                    "formats": [
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "float",
                        "bool",
                    ],
                },
                converters={6: lambda value: value == b"True"},
                unpack=True,
            )
            x = np.transpose(x)

            try:
                scores = cross_validate(
                    model,
                    x,
                    y,
                    scoring="accuracy",
                    return_train_score=True,
                    cv=cv,
                    error_score="raise",
                    n_jobs=-1,
                )
            except Exception as error:
                print(error)
                print(csv)
                exit()

            train_accuracy = np.mean(scores["train_score"])
            test_accuracy = np.mean(scores["test_score"])

            all_train_accuracies.append(train_accuracy)
            all_test_accuracies.append(test_accuracy)

            output.write(
                "%s,%s,%s,%s\n"
                % (
                    crop_type,
                    basename(csv),
                    train_accuracy,
                    test_accuracy,
                )
            )

    model_dump(model, path_join(models_dir, "%s.model" % label))

    output.close()

    return [all_train_accuracies, all_test_accuracies]


def print_summary(
    label,
    model_name,
    configuration,
    train_accs,
    test_accs,
    param_grid=None,
    best_params=None,
):
    summary_strings = (
        [model_name, label]
        + [f"{key}: {value}" for key, value in configuration.items()]
        + [
            "",
            "Overall training accuracy: %f (%f)"
            % (np.mean(train_accs), np.std(train_accs)),
            "Overall test accuracy: %f (%f)" % (np.mean(test_accs), np.std(test_accs)),
            "",
        ]
    )

    if param_grid and best_params:
        summary_strings += ["Best Parameters:"] + [
            f"{key}: {value} of {param_grid[key]}" for key, value in best_params.items()
        ]

    summary_strings += ["=" * 10, "", ""]

    print()
    for line in summary_strings:
        print(line)

    with open(path_join("results", "summary.txt"), "a") as summary:
        summary.write("\n".join(summary_strings))
