# built in libraries
from pathlib import Path

# external libraries
import pandas as pd
import numpy as np
from scipy import spatial
from nltk.stem import WordNetLemmatizer

LOCAL_FOLDER = Path(__file__).resolve().parent
LEMMATIZER = WordNetLemmatizer()


def convert_dict_to_df(movie_dict):
    # format in a pandas read-able way
    new_movies = {"movieId": [], "title": [], "genres": [], "tags": []}
    for i, movie_id in enumerate(movie_dict.keys()):
        new_movies["movieId"].append(movie_id)
        new_movies["title"].append(movie_dict[movie_id]["title"])
        new_movies["genres"].append(movie_dict[movie_id]["genres"])
        new_movies["tags"].append(list(movie_dict[movie_id]["tags"]))

    return pd.DataFrame.from_dict(new_movies)


def users_dict_to_df(users_dict):
    # format in a pandas read-able way
    new_users = {"userId": [], "ratings": []}
    for user_id in users_dict:
        new_users["userId"].append(user_id)
        new_users["ratings"].append(users_dict[user_id])

    return pd.DataFrame.from_dict(new_users)


def merge_movies_and_tags(movies, movie_tags):
    df = movies.merge(movie_tags, on="movieId", how="left")
    df = df.to_dict()
    new_df = {}
    for row in df["movieId"].keys():
        movie_id = df["movieId"][row]
        if movie_id in new_df:
            new_df[movie_id]["tags"].add(LEMMATIZER.lemmatize(df["tag"][row]))
        else:
            new_df[movie_id] = {}
            new_df[movie_id]["title"] = df["title"][row]
            new_df[movie_id]["genres"] = df["genres"][row]
            if pd.isna(df["tag"][row]):
                new_df[movie_id]["tags"] = {"no tag"}
            else:
                new_df[movie_id]["tags"] = {LEMMATIZER.lemmatize(df["tag"][row])}

    return new_df


def process_users(users, num_movies, movie_id_map):
    df = users.to_dict()
    new_users = {}
    for row in df["userId"].keys():
        user_id = df["userId"][row]
        if user_id in new_users:
            new_users[user_id][movie_id_map[df["movieId"][row]]] = df["rating"][row]
        else:
            new_users[user_id] = np.array([0]*num_movies)
            new_users[user_id][movie_id_map[df["movieId"][row]]] = df["rating"][row]

    return new_users


def load_movies_and_users(folder_name):
    def split_genres(string):
        return string.split("|")

    folder_path = LOCAL_FOLDER.joinpath(folder_name)

    # get and process movie_tags
    movie_tags = pd.read_csv(folder_path.joinpath("tags.csv"))
    movie_tags["tag"] = movie_tags["tag"].str.lower()
    movie_tags = movie_tags.drop(columns=["timestamp"])

    # get and process movies
    movies = pd.read_csv(folder_path.joinpath("movies.csv"))

    # this is because movie ids are not a sequence
    movie_id_map = {}
    for i, movie_id in enumerate(movies["movieId"]):
        movie_id_map[movie_id] = i

    movies["genres"] = movies["genres"].str.lower().apply(split_genres)
    movies = merge_movies_and_tags(movies, movie_tags)

    # get and process users
    users = pd.read_csv(folder_path.joinpath("ratings.csv"))
    users = process_users(users, len(movies.keys()), movie_id_map)

    return movies, users


def find_corr_movie(movie_id, movies, num_similar):
    movie_id_genres = set(movies[movie_id]["genres"])  # I use sets here because the lookup of a set is O(1) instead of a list which is O(n)
    movie_id_tags = movies[movie_id]["tags"]
    similar_movies = []

    for movie in movies.keys():
        if movie != movie_id:
            genre_overlap = tag_overlap = 0
            movie_genres = movies[movie]["genres"]
            movie_tags = movies[movie]["tags"]

            for genre in movie_genres:
                if genre in movie_id_genres:
                    genre_overlap += 1
            for tag in movie_tags:
                if tag in movie_id_tags:
                    if tag == "no tag":  # it's much easier to get a no tag overlap for many movies
                        tag_overlap += 0.1
                    else:
                        tag_overlap += 1

            genre_overlap_ratio = genre_overlap / len(movie_id_genres)
            tag_overlap_ratio = tag_overlap / len(movie_id_tags)
            similarity = genre_overlap_ratio + tag_overlap_ratio  # abstract weight here, this can be optimized
            similar_movies.append((movie, similarity))

    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)
    return [movie[0] for movie in similar_movies[:num_similar]]


def find_corr_users(user_id, users, num_similar):
    user_id_ratings = users[user_id]
    similar_users = []

    for user in users.keys():
        if user != user_id:
            user_ratings = users[user]
            similarity = 1 - spatial.distance.cosine(user_id_ratings, user_ratings)
            similar_users.append((user, similarity))

    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    return [user[0] for user in similar_users[:num_similar]]


if __name__ == "__main__":
    movies, users = load_movies_and_users("data")
    sim = find_corr_movie(1, movies, 5)
    print("Base movie id:", 1, "description:", movies[1])
    print("Similar movies:")
    for i in sim:
        print("Movie id:", i, "description:", movies[i])

    sim = find_corr_users(1, users, 5)
    print("\nBase user id:", 1, "description:", users[1])
    print("Similar users:")
    for i in sim:
        print("User id:", i, "description:", users[i])
