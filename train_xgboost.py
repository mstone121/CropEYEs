from datetime import datetime
from scipy.stats import uniform, randint

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold

from xgboost import XGBClassifier


from train import print_summary, train_model

data_dir = "data"
results_dir = "results"
models_dir = "models"

classifier = XGBClassifier


configuration = {
    "random_seed": 27,
    "cv_splits": 3,
    "cv_repeats": 5,
    "objective": "binary:logistic",
    "colsample_bytree": 0.8493071412088892,
    "gamma": 0.2466921548262684,
    "learning_rate": 0.19308562020455775,
    "max_depth": 4,
    "n_estimators": 124,
    "subsample": 0.8368410436197263,
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
                colsample_bytree=configuration["colsample_bytree"],
                gamma=configuration["gamma"],
                learning_rate=configuration["learning_rate"],
                max_depth=configuration["max_depth"],
                n_estimators=configuration["n_estimators"],
                subsample=configuration["subsample"],
                n_jobs=1,
            ),
        ),
    ],
)


train_accs, test_accs = train_model(
    model=model,
    cv=cv,
    label=run_datetime,
    use_numeric_labels=True,
    break_early=True,
)

print_summary(
    label=run_datetime,
    model_name=classifier.__name__,
    configuration=configuration,
    train_accs=train_accs,
    test_accs=test_accs,
)
