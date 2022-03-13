from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold

from keras.models import Sequential, Input
from keras.layers import Dense

from scikeras.wrappers import KerasClassifier


from train import print_summary, train_model


# class KerasClassifierWrapper(KerasClassifier):
#     def fit(self, *args, **kwargs):
#         return super().fit(*args, **kwargs, verbose=0)


data_dir = "data"
results_dir = "results"
models_dir = "models"


configuration = {
    "random_seed": 27,
    "cv_splits": 2,
    "cv_repeats": 1,
    "hyper_cv_splits": 2,
    "hyper_cv_repeats": 1,
}

param_grid = {
    # "poly__degree": [1, 3, 5],
    "classifier__model__layer1": [100, 200, 500],
    "classifier__model__layer2": [50, 100, 200],
    "classifier__model__optimizer": ["nadam", "sgd"],
}


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=configuration["cv_splits"],
    n_repeats=configuration["cv_repeats"],
    random_state=configuration["random_seed"],
)


hyper_cv = RepeatedStratifiedKFold(
    n_splits=configuration["hyper_cv_splits"],
    n_repeats=configuration["hyper_cv_repeats"],
    random_state=configuration["random_seed"],
)


def build_model(meta, layer1, layer2, optimizer):
    n_features_in_ = meta["n_features_in_"]

    model = Sequential(
        [
            Input(shape=(n_features_in_)),
            Dense(layer1, activation="relu"),
            Dense(layer2, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(loss="binary_crossentropy", optimizer=optimizer)

    return model


model = GridSearchCV(
    Pipeline(
        [
            # ("poly", PolynomialFeatures()),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            (
                "classifier",
                KerasClassifier(
                    model=build_model,
                    verbose=0,
                    random_state=configuration["random_seed"],
                ),
            ),
        ],
    ),
    param_grid,
    cv=hyper_cv,
)


train_accs, test_accs, best_params = train_model(
    model=model,
    cv=cv,
    label=run_datetime,
    use_numeric_labels=True,
    collect_best_params=True,
)

print_summary(
    label=run_datetime,
    model_name="Neural Network",
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
    param_grid=param_grid,
    best_params=best_params,
)
