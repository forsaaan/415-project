from src.find_corr import *

LOCAL_FOLDER = Path(__file__).resolve().parent


def get_avg_genre_ratings(user_id, users, num_similar, folder_name):
    def split_genres(string):
        return string.split("|")

    def avg_ratings(user_id, ratings):
        ratings = ratings[ratings['userId'] == user_id]
        ratings = ratings.merge(movies, on='movieId')
        ratings = ratings.drop(columns=['movieId', 'timestamp', 'title'])
        ratings['genres'] = ratings['genres'].apply(split_genres)
        ratings_long = ratings.copy()
        ratings_long['genres'] = ratings_long['genres'].explode('genres')
        user_ratings = ratings_long.groupby(['userId', 'genres']).mean().reset_index()

        genre_ratings = [0] * len(genres)
        for j, genre in enumerate(genres):
            if genre in user_ratings['genres'].tolist():
                genre_ratings[j] = user_ratings[user_ratings['genres'] == genre]['rating'].iloc[0]
        return [genre_ratings, ratings]

    def get_movie_genres(data):
        user_all_movies = []
        for a, curr_movie_genres in enumerate(data[1].iloc[:, 2]):
            movie_genres = [0] * len(genres)
            for b, genre in enumerate(genres):
                if genre in curr_movie_genres:
                    movie_genres[b] = 1
            user_all_movies.append([data[0], movie_genres])
        return user_all_movies

    folder_path = LOCAL_FOLDER.joinpath(folder_name)
    movies = pd.read_csv(folder_path.joinpath("movies.csv"))
    ratings = pd.read_csv(folder_path.joinpath("ratings.csv"))
    genres = movies['genres'].apply(split_genres).explode('genres').unique().tolist()

    user_data = avg_ratings(user_id, ratings)
    user_data = get_movie_genres(user_data)

    similar_users = find_corr_users(user_id, users, num_similar)

    for k in range(num_similar):
        similar_data = avg_ratings(similar_users[k], ratings)
        user_data.extend(get_movie_genres(similar_data))
    return pd.DataFrame(user_data)


if __name__ == "__main__":
    movies, users = load_movies_and_users("data")
    sim = find_corr_users(1, users, 5)
    test_data = get_avg_genre_ratings(1, users, 3, 'data')
    print(test_data)