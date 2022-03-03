from datetime import datetime
from os.path import join as path_join, basename
from glob import glob
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

data_dir = "data"
results_dir = "results"


def get_bool_value(value):
    return value == b"True"


output = open(path_join("results", "%s.csv" % datetime.now()), "w")
output.write(
    "%s,%s,%s,%s\n" % ("Crop Type", "Filename", "Accuracy", "Validation Accuray")
)

model = LinearDiscriminantAnalysis()

for folder in glob(path_join(data_dir, "*")):
    crop_type = basename(folder)

    csvs = glob(path_join(folder, "**", "*.csv"), recursive=True)
    csv_count = len(csvs)
    for index in tqdm(range(csv_count), desc="Training with %s" % crop_type):
        csv = csvs[index]

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

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

        model.fit(x_train, y_train)

        accuracy = model.score(x_train, y_train)
        val_accuracy = model.score(x_test, y_test)

        output.write(
            "%s,%s,%s,%s\n" % (crop_type, basename(csv), accuracy, val_accuracy)
        )
