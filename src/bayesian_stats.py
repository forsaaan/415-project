from random import randint
from sklearn.ensemble import RandomForestClassifier
import optuna

TRAIN_SET_LIMIT = 1000
TRAIN_SET_COUNT = 200

TRAIN_INPUT = list()
TRAIN_OUTPUT = list()
for i in range(TRAIN_SET_COUNT):
    a = randint(0, TRAIN_SET_LIMIT)
    b = randint(0, TRAIN_SET_LIMIT)
    c = randint(0, TRAIN_SET_LIMIT)
    op = a + (2*b) + (3*c) + a*b + (a*b*c)**2
    TRAIN_INPUT.append([a, b, c])
    TRAIN_OUTPUT.append(op)

split_input = TRAIN_INPUT[:-40]
split_output = TRAIN_OUTPUT[:-40]


# 1. Define an objective function to be maximized.
def objective(trial):

    # 2. Suggest values for the hyperparameters using a trial object.
    rf_max_depth = int(trial.suggest_loguniform('rf_max_depth', 2, 32))
    rf_estimators = int(trial.suggest_int('rf_estimators', 2, 20))
    classifier_obj = RandomForestClassifier(max_depth=rf_max_depth, n_estimators=rf_estimators, max_features="sqrt")
    classifier_obj.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)
    accuracy = classifier_obj.score(split_input, split_output)
    return accuracy


# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# we can view the progression of optimization process this way
# plot_contour shows how 2 parameters' gradients relates to the scoring of the BOpt, helpful for showing how over time the parameter ranges changes
# fig = optuna.visualization.plot_contour(study, params=["rf_max_depth", "rf_estimators"])
# fig.show()
# study.optimize(objective, n_trials=10)
#
# fig = optuna.visualization.plot_contour(study, params=["rf_max_depth", "rf_estimators"])
# fig.show()
# study.optimize(objective, n_trials=10)
#
# fig = optuna.visualization.plot_contour(study, params=["rf_max_depth", "rf_estimators"])
# fig.show()

# shows the overall progression (not related to the specific parameters) of BOpt, good to show whether or not the model converges
# fig = optuna.visualization.plot_optimization_history(study)
# fig.show()

# shows the importance of each parameter after a number of trials, this is like a hyper prior distribution for the probability distribution of each parameter
# fig = optuna.visualization.plot_param_importances(study)
# fig.show()

# pretty much a replica of the plot_contour but with each variable on their own
# fig = optuna.visualization.plot_slice(study, params=["x", "y"])
# fig.show()

# shows where the majority of the score of the model lies based on the steps made by BOpt
# fig = optuna.visualization.plot_edf([study])
# fig.show()
