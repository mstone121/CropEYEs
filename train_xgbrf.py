from datetime import datetime

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
    "cv_splits": 5,
    "cv_repeats": 3,
    "objective": "binary:logistic",
    "n_estimators": 300,
    "learning_rate": 0.2606759988756412,
    "subsample": 0.8297720412850104,
    "colsample_bynode": 0.7943780972507282,
    "reg_lambda": 0.000511315193617651,
    "max_depth": 4,
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
        ("pca", PCA(n_components=0.95)),
        (
            "classifier",
            classifier(
                objective=configuration["objective"],
                eval_metric="auc",
                use_label_encoder=False,
                random_state=configuration["random_seed"],
                n_jobs=1,
                n_estimators=configuration["n_estimators"],
                learning_rate=configuration["learning_rate"],
                subsample=configuration["subsample"],
                colsample_bynode=configuration["colsample_bynode"],
                reg_lambda=configuration["reg_lambda"],
                max_depth=configuration["max_depth"],
            ),
        ),
    ],
)


train_accs, test_accs = train_model(
    model=model,
    cv=cv,
    label=run_datetime,
    use_numeric_labels=True,
)

print_summary(
    label=run_datetime,
    model_name=classifier.__name__,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
)
