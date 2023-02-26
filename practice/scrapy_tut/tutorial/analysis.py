# coding=utf8
import lzma
import pickle
from sklearn import feature_extraction, metrics
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import thesis_utils as tu

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--seed", default=42, type=int, help="Random seed")

def site_data_analysis(args: argparse.Namespace):
    # import data from db
    train = tu.Dataset("site_data")
    train_data = train.data
    train_target = train.target

    selected_language_indicies, lang_array = tu.site_data_filter(train_data)
    train_data = np.array(train_data)[selected_language_indicies]
    train_target = np.array(train_target)[selected_language_indicies]
    y_labels = np.unique(train_target)
    tu.plot_histograms(lang_array, train_target, show=True)

def translated_data_analysis(args: argparse.Namespace):
    # import data from db
    train = tu.Dataset("translated_data")
    train_data = train.data
    train_target = train.target

    lang_array = []
    for i in train._class:
        lang_array.append(i.original_language)
    tu.plot_histograms(lang_array, train_target, show=True)

def test_model(args: argparse.Namespace):
    train = tu.Dataset("translated_data")
    train_data = np.array(train.data)
    train_target = np.array(train.target)
    # selected_language_indicies, lang_array = tu.site_data_filter(train_data)
    selected_indicies = np.where((train_target != 'Atm') & (train_target != 'Children'))[0] #becuase atm and children each have only 1 sample
    train_data = train_data[selected_indicies]
    train_target = train_target[selected_indicies]

    y_labels = np.unique(train_target)

    if False:
        # one hot encoder (for mlp)
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(np.array(train_y).reshape(-1, 1))
        train_y = enc.transform(np.array(train_y).reshape(-1, 1)).toarray()
    else:
        # lable encoder (for gbdt)
        le = LabelEncoder()
        le.fit(train_target)
        train_target = le.transform(train_target)

    # split data into train and test
    train_X, test_X, train_y, test_y = train_test_split(train_data, train_target, test_size=0.33, random_state=args.seed, stratify=train_target)

    # train the model
    # features_1 = TfidfVectorizer(ngram_range=(3, 5), analyzer="char", max_features=300, stop_words="english")
    # features_2 = TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=800,stop_words="english")
    # features = FeatureUnion([("tfidf1", features_1), ("tfidf2", features_2)])
    # features = FeatureUnion([("tfidf2", features_2)])
    # clf = MLPClassifier(random_state=args.seed, max_iter=5000, hidden_layer_sizes=(2000))
    # clf = GradientBoostingClassifier(random_state=args.seed, max_depth=5, n_estimators=100) 
    # pipe = Pipeline(steps=[("features", features), ("clf", clf)])
    pipe = Pipeline([
        # ("features", feature_extraction.text.CountVectorizer(analyzer="word", stop_words="english", max_features=2800, strip_accents= 'unicode' )),
        ("features", tu.tfidf_vectorizer()),
        #("neural_network", sklearn.neural_network.MLPClassifier(hidden_layer_sizes=1000, solver='lbfgs', max_iter= 100)),
        #("decision_tree", sklearn.tree.DecisionTreeClassifier())
        # ("mlp", MLPClassifier(verbose=1, hidden_layer_sizes=1000, max_iter=1000, solver='lbfgs'))
        ("gbdt", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
    ])

    tu.tfidf_to_csv(train_X, train_y, "website_tfidf_data.csv")

    try:
        print("trying to load model")
        with lzma.open("model.model" , "rb") as modelFile:
            model = pickle.load(modelFile)
    except:
        print("training model bc couldnt load it")
        model = pipe.fit(train_X, train_y)
        with lzma.open("model.model", "wb") as model_file: 
            pickle.dump(model , model_file)

    # predict 
    pred = model.predict(test_X)
    clf = model.named_steps["gbdt"]
    print("Accuracy:", metrics.accuracy_score(test_y, pred))
    tu.show_confusion_matrix(clf, y_labels, pred, test_y, train_y)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # site_data_analysis(args)
    # translated_data_analysis(args)
    test_model(args)
