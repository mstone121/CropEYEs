from os.path import join as path_join, basename
from glob import glob
from tqdm import tqdm
from joblib import dump as model_dump
from pprint import pprint
import numpy as np

from sklearn.model_selection import cross_validate

data_dir = "data"
results_dir = "results"
models_dir = "models"


def train_model(
    model,
    cv,
    label,
    collect_best_params=False,
    params_are_distributions=False,
    use_numeric_labels=False,
    break_early=False,
):
    output = open(path_join("results", "%s.csv" % label), "w")
    output.write(
        "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
    )

    all_train_accs = []
    all_test_accs = []
    best_params = {}

    for folder in glob(path_join(data_dir, "*")):
        crop_type = basename(folder)

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
                converters={
                    6: (lambda value: value == b"True")
                    if not use_numeric_labels
                    else (lambda value: 1 if value == b"True" else 0)
                },
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
                    return_estimator=collect_best_params,
                    cv=cv,
                    error_score="raise",
                    n_jobs=-1,
                )
            except Exception as error:
                print(error)
                print(csv)
                exit()

            train_accs = scores["train_score"]
            test_accs = scores["test_score"]

            all_train_accs.extend(train_accs)
            all_test_accs.extend(test_accs)

            output.write(
                "%s,%s,%s,%s\n"
                % (
                    crop_type,
                    basename(csv),
                    np.mean(train_accs),
                    np.mean(test_accs),
                )
            )

            if collect_best_params:
                for estimator in scores["estimator"]:
                    for param, value in estimator.best_params_.items():
                        if params_are_distributions:
                            if not param in best_params:
                                best_params[param] = []

                            best_params[param].append(value)

                        else:  # Discrete param grid
                            if not param in best_params:
                                best_params[param] = {}

                            if not value in best_params[param]:
                                best_params[param][value] = 1
                            else:
                                best_params[param][value] += 1

        if break_early:
            print("Warning: breaking early")
            break

    model_dump(model, path_join(models_dir, "%s.model" % label))

    output.close()

    if collect_best_params:
        if params_are_distributions:
            avg_best_params = {}
            for param, values in best_params.items():
                avg_best_params[param] = np.mean(values)

            return [all_test_accs, all_test_accs, avg_best_params]

        return [all_train_accs, all_test_accs, best_params]

    return [all_train_accs, all_test_accs]


def get_best_param_value(values):
    if type(values) is dict:
        best_value = None
        max_count = 0
        for value, count in values.items():
            if count > max_count:
                max_count = count
                best_value = value

        return best_value
    else:  # probably a distribution
        return values


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
        summary_strings += (
            ["Best Parameters:"]
            + [
                f"{key}: {get_best_param_value(values)} of {param_grid[key]}"
                for key, values in best_params.items()
            ]
            + [""]
        )

    summary_end = "\n".join(["=" * 10, "", ""])

    print()
    for line in summary_strings:
        print(line)

    if best_params:
        pprint(best_params)

    print(summary_end)

    with open(path_join("results", "summary.txt"), "a") as summary:
        summary.write("\n".join(summary_strings))
        if best_params:
            pprint(best_params, summary)

        summary.write(summary_end)
