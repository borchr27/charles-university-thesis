# coding=utf8
#!/usr/bin/env python3
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import lzma
import pickle
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import thesis_utils as tu
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--model", default="tf_NN.model", type=str, help="Output model path.")
parser.add_argument("--hidden_layer", default=1000, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.01, type=float, help="Final learning rate.")


##########################################################################################
# This code is used to run a simple analysis on the data that I collected from the websites
# using histograms, confusion matrices, etc. on either the site_data table or the 
# translated_data table. I also ran a simple gradient boosting classifier on the original 
# (not additional) translated_data.


def build_and_test_model(args: argparse.Namespace, X:np.ndarray, y:np.ndarray, vectorizer:TfidfVectorizer, csv:bool=False, unigrams:bool=False, cm:bool=False, clf_report:bool=False, clf_name:str="LinearSVC"):
    data_name = "original"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    weights = tu.get_weights(y)

    if unigrams:
        # make a dictionary with y labels as keys and their index as values
        category_to_id = dict(zip(y_labels, range(len(y_labels))))
        tu.table_correlated_unigrams(X, y, vectorizer, category_to_id, f"table_correlated_unigrams_{data_name}")
    if csv:
        tu.tfidf_to_csv(X.toarray(), y, "tfidf_data.csv")

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    if clf_name == "LinearSVC":
        pipe = Pipeline([
            # (f"{clf_name}", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
            (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights, fit_intercept=False))
        ])
    elif clf_name == "KNN":
        pipe = Pipeline([
            # (f"{clf_name}", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
            (f"{clf_name}", KNeighborsClassifier(n_neighbors=8, metric='cosine'))
        ])
    elif clf_name == "GBDT":
        pipe = Pipeline([
            (f"{clf_name}", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
        ])
    else:
        raise ValueError("Please choose a valid classifier (LinearSVC, KNN, GBDT)")

    try:
        print("Loading the model...")
        with lzma.open("model.model" , "rb") as modelFile:
            model = pickle.load(modelFile)
    except:
        print("Model could not be found, training a new model...")
        model = pipe.fit(train_X, train_y)
        with lzma.open("model.model", "wb") as model_file: 
            pickle.dump(model , model_file)

    pred = model.predict(test_X)
    clf = model.named_steps[clf_name]
    print("Error:", 1 - metrics.accuracy_score(test_y, pred, normalize=True))
    if cm:
        tu.plot_confusion_matrix(y_labels, pred, test_y, clf_name)
    if clf_report:
        tu.table_classification_report(test_y, pred, y_labels, clf_name)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = tu.Dataset()
    X, y, vectorizer = tu.data_prep(data)
    tu.table_variable_importance(X, y, vectorizer)
    # build_and_test_model(args, X, y, vectorizer, cm=True, clf_name="KNN")
    # tu.build_tensor_flow_NN(args, X, y)
    # tu.plot_all_histograms(data, "plot_all_hist")
    # tu.plot_original_histograms(data, "plot_original_english_counts")
    # tu.plot_all_results_from_probal()
    # tu.table_data_category_counts(data, "table_category_counts")
    # tu.plot_explore_classifiers(args, X, y)
    # tu.build_LSVC_models(args, X, y)