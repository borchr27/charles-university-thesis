from sklearn import metrics
from PostgresDatabase import PostgresDatabase 
from Website import Website

import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--seed", default=42, type=int, help="Random seed")

class Dataset:
    def __init__(self):
        self.websites = []
        self.data = []
        self.target = []

        db = PostgresDatabase()
        db.connect()
        if db.connection:
            db.execute('SELECT * FROM site_data;')
            result = db.cursor.fetchall()
            db.close()

            for row in result:
                w = Website(name=row[1], i_url=row[2], v_url=row[3], text=row[4], category=row[5], tags=row[6])
                self.websites.append(w)
                self.data.append(w.text)
                self.target.append(w.category)

def main(args: argparse.Namespace):
    # TODO: fix duplicate info in db
    # TODO: separate data by language when training, just english, try different, gradient boost, random forest, svc

    # import data from db
    train = Dataset()
    train_data = train.data
    train_target = train.target

    # one hot encode 
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(np.array(train_target).reshape(-1, 1))
    train_target = enc.transform(np.array(train_target).reshape(-1, 1)).toarray()

    # split data into train and test
    train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.2, random_state=args.seed)
    
    # train the model
    features_1 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=500) #, stop_words="english")
    features_2 = TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=500) #, stop_words="english")
    features = FeatureUnion([("tfidf1", features_1), ("tfidf2", features_2)])
    # features = FeatureUnion([("tfidf2", features_2)])
    clf = MLPClassifier(random_state=args.seed, max_iter=1000, hidden_layer_sizes=(500)) 
    pipe = Pipeline(steps=[("features", features), ("clf", clf)])
    model = pipe.fit(train_data, train_target)

    # predict 
    pred = model.predict(test_data)
    # print(train_data[0], train_target[0])
    print("Accuracy:", metrics.accuracy_score(test_target, pred))


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
