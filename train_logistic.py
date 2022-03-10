from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

from train import print_summary, train_model

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = LogisticRegression


configuration = {
    "random_seed": 27,
    "cv_splits": 5,
    "cv_repeats": 3,
    "poly_degree": 3,
    "solver": "liblinear",
    "C": 10,
}


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=configuration["cv_splits"],
    n_repeats=configuration["cv_repeats"],
    random_state=configuration["random_seed"],
)

model = Pipeline(
    [
        ("polynomial", PolynomialFeatures(degree=configuration["poly_degree"])),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
        (
            "classifier",
            classifier(solver=configuration["solver"], C=configuration["C"]),
        ),
    ],
)


train_accs, test_accs = train_model(model=model, cv=cv, label=run_datetime)

print_summary(
    label=run_datetime,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
)
