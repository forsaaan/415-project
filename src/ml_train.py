from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from src.find_corr import load_movies_and_users, find_corr_movie
from src.ml_helper import get_user_stats, get_favorite_movie, genre_indices

LOCAL_FOLDER = Path(__file__).resolve().parent


def standardized_scores(score_lists):
    temp = []
    for score_list in score_lists:
        np_list = np.array(score_list[:20])
        standardized = (np_list - np_list.mean()) / np_list.std()
        temp.append(standardized.tolist() + score_list[20:])
    return temp


def train_model(inputs, target, max_depth, estimators):
    model = RandomForestRegressor(max_depth=max_depth, n_estimators=estimators, max_features='sqrt')
    cv_scores = cross_val_score(
        model,
        inputs,
        target,
        cv=5
    )
    model.fit(inputs, target)
    print("CV score:", cv_scores.mean())

    return model


def test_model(model, inputs, target):
    return model.score(inputs, target)


def suggest_to_user(user_id, folder_name="data"):
    movies, users, movie_id_map, id_movie_map = load_movies_and_users(folder_name, ids=True)
    user_stats = get_user_stats(user_id, users, movies, id_movie_map)
    with open(LOCAL_FOLDER.joinpath("best_model.pkl"), "rb") as f:
        best_clf = pickle.load(f)
    favorite_movie_id, seen = get_favorite_movie(user_id, users, id_movie_map)
    suggestions = find_corr_movie(favorite_movie_id, movies, 10)

    best_pred_score = 0
    best_movie = suggestions[0]
    for suggestion in suggestions:
        movie_ids = [0] * len(list(genre_indices.keys()))
        for genre in movies[suggestion]["genres"]:
            movie_ids[genre_indices[genre]] = 1
        if suggestion not in seen:
            pred_score = best_clf.predict([user_stats + movie_ids])
            if pred_score > best_pred_score:
                best_pred_score = pred_score
                best_movie = best_movie

    return best_pred_score[0], movies[best_movie]


if __name__ == "__main__":
    user_id = 1
    score, movie = suggest_to_user(user_id=user_id)
    print("User ID:", user_id, "\nMovie:", movie, "\nPredicted score: %.2f" % score)
