import numpy as np
from lightfm import LightFM
import pandas as pd
import requests, zipfile, io
import os.path
import scipy.sparse as sp
from sklearn.preprocessing import MultiLabelBinarizer


def download_dataset(url, local_download=True):
    if local_download:
        if not os.path.exists("ml-1m"):
            if not (
                    os.path.isfile("ml-1m/movies.dat") and os.path.isfile("ml-1m/ratings.dat")
                    and os.path.isfile("ml-1m/users.dat")):
                print("Downloading files...")
                file = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(file.content))
                z.extractall()


def build_interaction_matrix(data, min_rating):
    # select only the rows with a rating of at least min_rating then set the UserID and MovieID as indices
    data = data[data[2] >= min_rating].set_index([0, 1])
    # create COO type matrix using UserID and MovieID as coordinates and Rating as value
    matrix = sp.coo_matrix((data[2], (data.index.codes[0], data.index.codes[1])))

    return matrix


def fetch_ml(min_rating):
    url = 'https://github.com/Ascerte/Movielens-Dataset/releases/download/1.0/ml-1m.zip'
    download_dataset(url)

# features in movies.dat : "MovieID", "Title", "Genres"
    movie_data = pd.read_csv("ml-1m/movies.dat", sep="::", header=None,
                             engine='python', usecols=[1, 2])
# features in ratings.dat : "UserID", "MovieID", "Rating", "Timestamp"
    matrix_data = pd.read_csv("ml-1m/ratings.dat", sep="::", header=None,
                              engine='python', usecols=[0, 1, 2])

# features in users.dat : "UserID", "Gender", "Age", "Occupation", "Zip-code"
    user_data = pd.read_csv("ml-1m/users.dat", sep="::", header=None,
                            engine='python', usecols=[1, 2, 3, 4])

    movie_label = np.array(movie_data[movie_data.columns[0]])

    mlb = MultiLabelBinarizer()

    movie_data = movie_data.join(pd.DataFrame(mlb.fit_transform(movie_data.pop(2).str.split("|")),
                                              columns=mlb.classes_,
                                              index=movie_data.index))

    data = {
        'train': build_interaction_matrix(matrix_data, min_rating=min_rating),
        'user_data': user_data,
        'movie_label': movie_label,
        'movie_data': movie_data
    }
    return data


data = fetch_ml(min_rating=4.0)

model = LightFM(loss='warp')
print("Training model...")
model.fit(data['train'], epochs=30, num_threads=4)
print("Model finished training.")


def recommendation(model, user_ids, data):
    n_items = data['train'].shape[1]
    print("items",n_items)
    for uid in user_ids:
        known_positives = data['movie_label'][data['train'].tocsr()[uid].indices]

        scores = model.predict(uid, np.arange(n_items))

        top_items = data['movie_label'][np.argsort(-scores)]

        print("User ", uid)
        print("Known positives:")
        for x in known_positives[:3]:
            print(x)
        print("Recommended:")

        for x in top_items[:3]:
            print(x)
        print("\n")


recommendation(model, [35, 89, 233], data)
