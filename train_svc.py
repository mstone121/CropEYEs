from datetime import datetime
from distutils.command.config import config

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.svm import SVC

from train import print_summary, train_model

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = SVC


configuration = {
    "random_seed": 27,
    "cv_splits": 5,
    "cv_repeats": 3,
    "pca_variance": 0.95,
    "C": 1,
    "gamma": 1,
    "kernel": "poly",
    "degree": 3,
}


run_datetime = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")


cv = RepeatedStratifiedKFold(
    n_splits=configuration["cv_splits"],
    n_repeats=configuration["cv_repeats"],
    random_state=configuration["random_seed"],
)

model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=configuration["pca_variance"])),
        (
            "classifier",
            classifier(
                kernel=configuration["kernel"],
                C=configuration["C"],
                gamma=configuration["gamma"],
                degree=configuration["degree"],
            ),
        ),
    ],
)


train_accs, test_accs = train_model(model=model, cv=cv, label=run_datetime)

print_summary(
    label=run_datetime,
    model_name=classifier.__name__,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
)
