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
from sklearn.linear_model import LogisticRegression

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = LogisticRegression


class Configuration:
    random_seed = 27

    hyper_cv_splits = 3
    hyper_cv_repeats = 3

    test_cv_splits = 3
    test_cv_repeats = 3

    solver = "liblinear"


param_grid = {
    "polynomial__degree": [3, 5],
    "pca__n_components": [0.8, 0.9, 0.95],
    "classifier__C": [0.1, 1, 10],
}


def get_bool_value(value):
    return value == b"True"


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


hyper_cv = RepeatedStratifiedKFold(
    n_splits=Configuration.hyper_cv_splits,
    n_repeats=Configuration.hyper_cv_repeats,
    random_state=Configuration.random_seed,
)

test_cv = RepeatedStratifiedKFold(
    n_splits=Configuration.test_cv_splits,
    n_repeats=Configuration.test_cv_repeats,
    random_state=Configuration.random_seed,
)

model = GridSearchCV(
    Pipeline(
        [
            ("polynomial", PolynomialFeatures()),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            (
                "classifier",
                classifier(solver=Configuration.solver, max_iter=10000),
            ),
        ],
    ),
    param_grid=param_grid,
    cv=hyper_cv,
    n_jobs=-1,
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
                return_estimator=True,
                cv=test_cv,
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

        for estimator in scores["estimator"]:
            for param, value in estimator.best_params_.items():
                best_params[param][value] += 1

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


def get_best_param_value(values: dict):
    best_value = None
    max_count = 0
    for value, best_count in values.items():
        if best_count > max_count:
            best_value = value

    return best_value


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
        "Best Parameters:",
    ]
    + [
        f"{key}: {get_best_param_value(values)} of {param_grid[key]}"
        for key, values in best_params.items()
    ]
    + ["", "=" * 10, "", ""]
)

print()
for line in summary_strings:
    print(line)

with open(path_join("results", "summary.txt"), "a") as summary:
    summary.write("\n".join(summary_strings))