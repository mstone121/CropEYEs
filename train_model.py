from datetime import datetime
from os.path import join as path_join, basename
from glob import glob
from tqdm import tqdm
import numpy as np
from joblib import dump as model_dump

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = LogisticRegression


class Configuration:
    random_seed = 27
    polynomial_degrees = 3
    cv_splits = 2
    cv_repeats = 1


param_grid = {}


def get_bool_value(value):
    return value == b"True"


def date_filename():
    return datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=Configuration.cv_splits,
    n_repeats=Configuration.cv_repeats,
    random_state=Configuration.random_seed,
)

model = GridSearchCV(
    Pipeline(
        (
            ["polynomial", PolynomialFeatures(degree=Configuration.polynomial_degrees)],
            ["scaler", StandardScaler()],
            ["classifier", classifier(max_iter=1000)],
        ),
    ),
    param_grid=param_grid,
    cv=cv,
    return_train_score=True,
)


accuracy = {}

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

        model.fit(x, y)
        crop_accuracy[basename(csv)] = [
            model.cv_results_["mean_train_score"],
            model.cv_results_["mean_test_score"],
        ]

    accuracy[crop_type] = crop_accuracy


model_dump(model, path_join(models_dir, "%s.model" % date_filename()))


all_train_accuracies = []
all_test_accuracies = []
with open(path_join("results", "%s.csv" % date_filename()), "w") as output:
    output.write(
        "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
    )

    for crop_type, crop_accuracy in accuracy.items():
        for csv, csv_accuracy in crop_accuracy.items():
            train_accuracy = csv_accuracy[0][model.best_index_]
            test_accuracy = csv_accuracy[1][model.best_index_]

            all_train_accuracies.append(train_accuracy)
            all_test_accuracies.append(test_accuracy)

            output.write(
                "%s,%s,%s,%s\n" % (crop_type, csv, train_accuracy, test_accuracy)
            )


summary_strings = (
    ["%s Model" % classifier.__name__, date_filename()]
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
    + [f"{key}: {value}" for key, value in model.best_params_]
    + ["", "=" * 10, "", ""]
)

print()
for line in summary_strings:
    print(line)

with open(path_join("results", "summary.txt"), "a") as summary:
    summary.write("\n".join(summary_strings))
