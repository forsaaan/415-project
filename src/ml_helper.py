from pathlib import Path
import pickle

from sklearn.model_selection import train_test_split

LOCAL_FOLDER = Path(__file__).resolve().parent


def get_train_test_data(train_prop):
    with open(LOCAL_FOLDER.joinpath("std_data.pkl"), "rb") as file:
        model_input = pickle.load(file)
    with open(LOCAL_FOLDER.joinpath("target.pkl"), "rb") as file:
        model_target = pickle.load(file)

    train_index, test_index = train_test_split(
        list(range(len(model_target))), train_size=train_prop
    )

    train_input, train_target, test_input, test_target = [], [], [], []
    for index in train_index:
        train_input.append(model_input[index])
        train_target.append(model_target[index])
    for index in test_index:
        test_input.append(model_input[index])
        test_target.append(model_target[index])

    return train_input, train_target, test_input, test_target


def get_favorite_movie(user_id, users, id_movie_map):
    highest = index = 0
    seen = set()
    for i, score in enumerate(users[user_id]):
        if score != 0:
            seen.add(id_movie_map[i])
        if score > highest:
            highest = score
            index = i
    return id_movie_map[index], seen


def get_user_stats(user_id, users, movies, id_movie_map):
    user_stats = [None] * len(list(genre_indices.keys()))
    for i, score in enumerate(users[user_id]):
        if score > 0:
            for genre in movies[id_movie_map[i]]["genres"]:
                if user_stats[genre_indices[genre]] is not None:
                    user_stats[genre_indices[genre]][0] += score
                    user_stats[genre_indices[genre]][1] += 1
                else:
                    user_stats[genre_indices[genre]] = [score, 1]
    temp = []
    for stat in user_stats:
        if stat is not None and stat[1] > 0:
            temp.append(stat[0] / stat[1])
        else:
            temp.append(0)
    return temp


def get_data():
    pass
    # movie_info, user_ratings = load_movies_and_users(folder_name)
    # movies = []
    # data = []
    # target = []
    # index_id = {}
    # genre_indices = {}
    # i = j = 0
    # for movie_id in movie_info.keys():
    #     index_id[i] = movie_id
    #     i += 1
    #     for genre in movie_info[movie_id]["genres"]:
    #         if not genre in genre_indices:
    #             genre_indices[genre] = j
    #             j += 1
    #
    # for movie_id in movie_info.keys():
    #     temp = [0] * len(list(genre_indices.keys()))
    #     for genre in movie_info[movie_id]["genres"]:
    #         temp[genre_indices[genre]] = 1
    #     movies.append(temp)
    #
    # for i in range(len(movies)):
    #     for user_id in user_ratings.keys():
    #         user_genres = {}
    #         if user_ratings[user_id][i] == 0:
    #             continue
    #         for j, movie_score in enumerate(user_ratings[user_id]):
    #             if movie_score > 0 and i != j:
    #                 movie_id = index_id[j]
    #                 for genre in movie_info[movie_id]["genres"]:
    #                     if genre in user_genres:
    #                         user_genres[genre][0] += movie_score
    #                         user_genres[genre][1] += 1
    #                     else:
    #                         user_genres[genre] = [movie_score, 1]
    #         user_avg = [0] * len(list(genre_indices.keys()))
    #         for genre in user_genres.keys():
    #             user_avg[genre_indices[genre]] = user_genres[genre][0] / user_genres[genre][1]
    #         data.append(user_avg + movies[i])
    #         target.append(user_ratings[user_id][i])


genre_indices = {'adventure': 0, 'animation': 1, 'children': 2, 'comedy': 3, 'fantasy': 4,
                 'romance': 5, 'drama': 6, 'action': 7, 'crime': 8, 'thriller': 9, 'horror': 10,
                 'mystery': 11, 'sci-fi': 12, 'war': 13, 'musical': 14, 'documentary': 15,
                 'imax': 16, 'western': 17, 'film-noir': 18, '(no genres listed)': 19}
