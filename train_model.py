from datetime import datetime
from os.path import join as path_join, basename
from glob import glob


from tqdm import tqdm
import numpy as np
from joblib import dump as model_dump


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = LogisticRegression


class Configuration:
    polynomial_degrees = 1
    cv_splits = 2
    cv_repeats = 1


def configuration_lines():
    return [
        f"{key}: {value}"
        for key, value in vars(Configuration).items()
        if not key.startswith("__")
    ]


def get_bool_value(value):
    return value == b"True"


def date_filename():
    return datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


output = open(path_join("results", "%s.csv" % date_filename()), "w")
output.write(
    "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
)

model = Pipeline(
    (
        ["polynomial", PolynomialFeatures(degree=Configuration.polynomial_degrees)],
        ["scaler", StandardScaler()],
        ["model", classifier()],
    )
)

cv = RepeatedStratifiedKFold(
    n_splits=Configuration.cv_splits, n_repeats=Configuration.cv_repeats
)

all_training_accuracies = []
all_test_accuracies = []

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
            )
        except Exception as error:
            print(error)
            print(csv)
            exit()

        train_accuracy = np.mean(scores["train_score"])
        test_accuracy = np.mean(scores["test_score"])

        all_training_accuracies.append(train_accuracy)
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

model_dump(model, path_join(models_dir, "%s.model" % date_filename()))

summary_string = (
    ["%s Model" % classifier.__name__, date_filename()]
    + configuration_lines()
    + [
        "Overall training accuracy: %f (%f)"
        % (np.mean(all_training_accuracies), np.std(all_training_accuracies)),
        "Overall test accuracy: %f (%f)"
        % (np.mean(all_test_accuracies), np.std(all_test_accuracies)),
        "=" * 10,
        "",
    ]
)

print()
for line in summary_string:
    print(line)

summary = open(path_join("results", "summary.txt"), "a")
summary.write("\n".join(summary_string))

output.close()
summary.close()
