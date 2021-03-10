from src.test_data import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

LOCAL_FOLDER = Path(__file__).resolve().parent


def get_train_test_data(user_id, movie_id, num_similar, folder_name, train_size):
    folder_path = LOCAL_FOLDER.joinpath(folder_name)

    movies = pd.read_csv(folder_path.joinpath("movies.csv"))
    ratings = pd.read_csv(folder_path.joinpath("ratings.csv"))
    ratings = ratings.merge(movies, on='movieId')
    ratings_train, ratings_test = train_test_split(ratings, train_size=train_size)

    movie_info, user_ratings = load_movies_and_users(folder_name)
    similar_users = find_corr_users(user_id, user_ratings, num_similar)

    train_features = get_avg_genre_ratings(user_id, movie_id, similar_users, ratings_train, folder_name)
    train_target = get_user_reviews(user_id, similar_users, ratings_train)

    test_features = get_avg_genre_ratings(user_id, movie_id, similar_users, ratings_test, folder_name)
    test_target = get_user_reviews(user_id, similar_users, ratings_test)

    return train_features, train_target, test_features, test_target


def train_model():

if __name__ == "__main__":
    a, b, c, d = get_train_test_data(1, 1, 3, 'data', .9)
    print(a)

