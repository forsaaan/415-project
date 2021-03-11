from pathlib import Path
import pickle

import optuna
from src.ml_train import train_model, test_model
from src.ml_helper import get_train_test_data

LOCAL_FOLDER = Path(__file__).resolve().parent


class Objective(object):
    def __init__(self, tr_i, tr_t, te_i, te_t):
        self.tr_i = tr_i
        self.tr_t = tr_t
        self.te_i = te_i
        self.te_t = te_t

    def __call__(self, trial):
        max_depth = trial.suggest_int('max_depth', 2, 32)
        estimators = trial.suggest_int('estimators', 2, 20)
        model = train_model(self.tr_i, self.tr_t, max_depth, estimators)
        score = test_model(model, self.te_i, self.te_t)

        with open(LOCAL_FOLDER.joinpath("models/{}.pkl".format(trial.number)), "wb") as f:
            pickle.dump(model, f)

        return score


if __name__ == "__main__":
    train_input, train_target, test_input, test_target = get_train_test_data(.7)
    obj = Objective(train_input, train_target, test_input, test_target)

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=10)
    # user_id = 1
    # score, movie = suggest_to_user(study, user_id=user_id)
    # print("User ID:", user_id, "\nMovie:", movie, "\nPredicted score: %.2f" % score)
    fig = optuna.visualization.plot_contour(study, params=["max_depth", "estimators"])
    fig.show()
    study.optimize(obj, n_trials=10)
    fig = optuna.visualization.plot_contour(study, params=["max_depth", "estimators"])
    fig.show()
    study.optimize(obj, n_trials=10)
    fig = optuna.visualization.plot_contour(study, params=["max_depth", "estimators"])
    fig.show()
    study.optimize(obj, n_trials=10)
    fig = optuna.visualization.plot_contour(study, params=["max_depth", "estimators"])
    fig.show()
    study.optimize(obj, n_trials=10)