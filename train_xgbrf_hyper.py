from datetime import datetime
from scipy.stats import uniform, randint

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from xgboost import XGBRFClassifier


from train import print_summary, train_model

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = XGBRFClassifier


configuration = {
    "random_seed": 27,
    "cv_splits": 3,
    "cv_repeats": 3,
    "hyper_cv_splits": 3,
    "hyper_cv_repeats": 3,
    "objective": "binary:logistic",
}

param_grid = {
    "classifier__n_estimators": randint(100, 500),
    "classifier__learning_rate": uniform(0.01, 0.5),
    "classifier__subsample": uniform(0.6, 0.4),
    "classifier__colsample_bynode": uniform(0.6, 0.4),
    "classifier__reg_lambda": uniform(1e-05, 1e-03),
    "classifier__max_depth": randint(2, 6),
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

model = RandomizedSearchCV(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95)),
            (
                "classifier",
                classifier(
                    objective=configuration["objective"],
                    eval_metric="auc",
                    use_label_encoder=False,
                    random_state=configuration["random_seed"],
                    n_jobs=1,
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
    params_are_distributions=True,
)

print_summary(
    label=run_datetime,
    model_name=classifier.__name__,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
    param_grid=param_grid,
    best_params=best_params,
)
