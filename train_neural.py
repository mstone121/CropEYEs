from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold

from keras.models import Sequential, Input
from keras.layers import Dense

from scikeras.wrappers import KerasClassifier


from train import print_summary, train_model


data_dir = "data"
results_dir = "results"
models_dir = "models"


configuration = {
    "random_seed": 27,
    "cv_splits": 5,
    "cv_repeats": 3,
    "layer1": 500,
    "layer2": 100,
    "optimizer": "nadam",
}


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=configuration["cv_splits"],
    n_repeats=configuration["cv_repeats"],
    random_state=configuration["random_seed"],
)


def build_model(meta):
    n_features_in_ = meta["n_features_in_"]

    model = Sequential(
        [
            Input(shape=(n_features_in_)),
            Dense(configuration["layer1"], activation="relu"),
            Dense(configuration["layer2"], activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy")

    return model


model = Pipeline(
    [
        ("poly", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        (
            "classifier",
            KerasClassifier(model=build_model, verbose=0),
        ),
    ],
)


train_accs, test_accs = train_model(
    model=model, cv=cv, label=run_datetime, use_numeric_labels=True
)

print_summary(
    label=run_datetime,
    model_name="Neural Network",
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
)
