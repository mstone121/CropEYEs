from datetime import datetime

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
    "cv_splits": 2,
    "cv_repeats": 1,
    "hyper_cv_splits": 2,
    "hyper_cv_repeats": 1,
    "pca_variance": 0.95,
}


param_grid = {
    "classifier__C": [1, 10, 100, 1000],
    "classifier__gamma": [1, 0.1, 0.001, 0.0001],
    "classifier__kernel": ["poly", "rbf"],
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

model = GridSearchCV(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=configuration["pca_variance"])),
            ("classifier", classifier()),
        ],
    ),
    param_grid,
    cv=hyper_cv,
)


train_accs, test_accs, best_params = train_model(
    model=model, cv=cv, label=run_datetime, collect_best_params=True
)

print_summary(
    label=run_datetime,
    model_name=classifier.__name__,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
    best_params=best_params,
    param_grid=param_grid,
)
