from src.find_corr import *

LOCAL_FOLDER = Path(__file__).resolve().parent


def get_avg_genre_ratings(user_id, movie_id, similar_users, ratings, folder_name):
    def split_genres(string):
        return string.split("|")

    def avg_ratings(user_id, ratings):
        ratings = ratings[ratings['userId'] == user_id]
        ratings = ratings.drop(columns=['timestamp', 'title'])
        ratings['genres'] = ratings['genres'].apply(split_genres)
        ratings_long = ratings.copy()
        ratings = ratings.drop(columns='movieId')
        ratings_long = ratings_long[ratings_long['movieId'] != movie_id]
        ratings_long = ratings_long.drop(columns='movieId')
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
    genres = movies['genres'].apply(split_genres).explode('genres').unique().tolist()

    user_data = avg_ratings(user_id, ratings)
    user_data = get_movie_genres(user_data)

    for k in range(len(similar_users)):
        similar_data = avg_ratings(similar_users[k], ratings)
        user_data.extend(get_movie_genres(similar_data))
    return pd.DataFrame(user_data)


def get_user_reviews(user_id, similar_users, ratings):
    def user_reviews(user_id, ratings):
        user_ratings = ratings[ratings['userId'] == user_id]
        user_ratings = user_ratings['rating'].tolist()
        return user_ratings

    reviews = user_reviews(user_id, ratings)
    for user in similar_users:
        reviews.extend(user_reviews(user, ratings))
    return reviews


if __name__ == "__main__":
    movies, users = load_movies_and_users("data")
    test_data = get_avg_genre_ratings(1, 1, 3, 'data')
    test_ratings = get_user_reviews(1, find_corr_users(1, users, 3), 'data')
    print(test_data)
    print(len(test_ratings))
