from datetime import datetime
from os.path import join as path_join, basename
from glob import glob
from sklearn.decomposition import PCA

from tqdm import tqdm
import numpy as np
from joblib import dump as model_dump

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.ensemble import RandomForestClassifier

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = RandomForestClassifier


class Configuration:
    random_seed = 27

    cv_splits = 5
    cv_repeats = 3
    n_estimators = 200


def get_bool_value(value):
    return value == b"True"


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=Configuration.cv_splits,
    n_repeats=Configuration.cv_repeats,
    random_state=Configuration.random_seed,
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            classifier(n_jobs=-1),
        ),
    ],
)


output = open(path_join("results", "%s.csv" % run_datetime), "w")
output.write(
    "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
)

all_train_accuracies = []
all_test_accuracies = []
best_params = {
    param: {value: 0 for value in param_values}
    for param, param_values in param_grid.items()
}

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
            converters={6: get_bool_value},
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


model_dump(model, path_join(models_dir, "%s.model" % run_datetime))

output.close()

summary_strings = (
    ["%s Model" % classifier.__name__, run_datetime]
    + [
        f"{key}: {value}"
        for key, value in vars(Configuration).items()
        if not key.startswith("__")
    ]
    + [
        "",
        "Overall training accuracy: %f (%f)"
        % (np.mean(all_train_accuracies), np.std(all_train_accuracies)),
        "Overall test accuracy: %f (%f)"
        % (np.mean(all_test_accuracies), np.std(all_test_accuracies)),
        "",
        "=" * 10,
        "",
        "",
    ]
)

print()
for line in summary_strings:
    print(line)

with open(path_join("results", "summary.txt"), "a") as summary:
    summary.write("\n".join(summary_strings))
