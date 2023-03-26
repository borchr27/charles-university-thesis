# coding=utf8
#!/usr/bin/env python3
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default
import lzma
import pickle
import re
import warnings
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
import thesis_utils as tu
import matplotlib.pyplot as plt
from itertools import cycle

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--model", default="model_NN.model", type=str, help="Output model path.")
parser.add_argument("--hidden_layer", default=1000, type=int, help="Size of the hidden layer.")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Initial learning rate.")
parser.add_argument("--learning_rate_final", default=0.01, type=float, help="Final learning rate.")

parser.add_argument("--csv", default=False, type=bool, help="Output a csv file with the tfidf data for probal/.")
parser.add_argument("--plot", default=None, type=str, help=["all_histograms", "original_en_hist", "all_results_from_probal", "test_results", "explore_classifiers", "pr_curve", "test_results_averaged"])
parser.add_argument("--table", default=None, type=str, help=["correlated_unigrams", "classification_report", "data_category_counts", "variable_importance"])
parser.add_argument("--clf", default=None, type=str, help="Choose a classifier for the test model function. [KNN, LinearSVC, GBDT]")

##########################################################################################
# This code is used to run a simple analysis on the data that I collected from the websites
# using histograms, confusion matrices, etc. on either the site_data table or the 
# translated_data table. I also ran a simple gradient boosting classifier on the original 
# (not additional) translated_data.


def test_model(args, X:np.ndarray, y:np.ndarray, vectorizer:TfidfVectorizer, unigrams:bool=False, cm:bool=False, clf_name:str=None) -> float:
    assert clf_name in ["LinearSVC", "KNN", "GBDT"], "Please choose a valid classifier (LinearSVC, KNN, GBDT)"
    data_name = "original"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)

    weights = tu.get_weights(y)

    if unigrams:
        # make a dictionary with y labels as keys and their index as values
        category_to_id = dict(zip(y_labels, range(len(y_labels))))
        tu.table_correlated_unigrams(X, y, vectorizer, category_to_id, f"table_correlated_unigrams_{data_name}")
    if args.csv:
        tu.tfidf_to_csv(X.toarray(), y, "tfidf_data.csv")

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    if clf_name == "LinearSVC":
        pipe = Pipeline([
            (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights))
        ])
    elif clf_name == "KNN":
        pipe = Pipeline([
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
        with lzma.open(f"model_{clf_name}.model" , "rb") as modelFile:
            model = pickle.load(modelFile)
    except:
        print("Model could not be found, training a new model...")
        model = pipe.fit(train_X, train_y)
        with lzma.open(f"model_{clf_name}.model", "wb") as model_file: 
            pickle.dump(model , model_file)

    pred = model.predict(test_X)
    clf = model.named_steps[clf_name]
    error = 1 - metrics.accuracy_score(test_y, pred, normalize=True)
    print(f"{clf_name} Error:", error)
    if cm:
        tu.plot_confusion_matrix(y_labels, pred, test_y, clf_name)
    if args.table == "classification_report":
        tu.table_classification_report(test_y, pred, y_labels, clf_name)

    return error

def table_compare_top_three_models(args, X, y, vectorizer) -> None:
    nn_error = tu.build_tensor_flow_NN(args, X, y)
    knn_error = test_model(args, X, y, vectorizer, clf_name="KNN")
    lsvc_error = test_model(args, X, y, vectorizer, clf_name="LinearSVC")

    errors = pd.DataFrame({"Model": ["Neural Network", "KNN", "LinearSVC"], "Error": [nn_error, knn_error, lsvc_error]})
    errors = errors.sort_values(by="Error", ascending=True)
    errors.to_latex(f"{tu.IMG_FILE_PATH}table_best_errors.tex", index=False, float_format="%.3f")

def optimize_classifier(args, X, y):
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    weights = tu.get_weights(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    
    clf = LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights)
    param_grid = {
        'intercept_scaling': [0.5,1],
        }
    grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=2)
    grid_search.fit(train_X, train_y)
    
    # Print the best parameters found by GridSearchCV
    print(f"best params: {grid_search.best_params_}")
    print(f"best score: {grid_search.best_score_}")
    print(f"best estimator: {grid_search.best_estimator_}")
    print(f"best index: {grid_search.best_index_}")

    best_estimator = grid_search.best_estimator_
    pred = best_estimator.predict(test_X)
    error = 1 - metrics.accuracy_score(test_y, pred, normalize=True)
    print(f"Error: {error}")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # data, X, y, vectorizer = None, None, None, None
    data = tu.Dataset()
    X, y, vectorizer = tu.data_prep(data)
    # test_model(args, X, y, vectorizer, cm=True, clf_name=args.clf)
    # tu.build_LSVC_models(args, X, y)
    # table_compare_top_three_models(args, X, y, vectorizer)
    optimize_classifier(args, X, y)


    if args.plot == "all_histograms":
        tu.plot_all_histograms(data, "plot_all_hist")
    if args.plot == "original_en_hist":
        tu.plot_original_en_histograms(data, "plot_original_english_counts")
    if args.plot == "all_results_from_probal":
        tu.plot_all_results_from_probal()
    if args.plot == "test_results":
        tu.plot_test_results()
    if args.plot == "test_results_averaged":
        tu.plot_test_results_averaged()
    if args.plot == "explore_classifiers":
        tu.plot_explore_classifiers(args, X, y)
    if args.plot == "pr_curve":
        tu.plot_pr_curve(args, X, y)
    
    if args.table == "correlated_unigrams":
        tu.table_correlated_unigrams(X, y, vectorizer, "table_correlated_unigrams")
    if args.table == "data_category_counts":
        tu.table_category_counts(data, "table_category_counts")
    if args.table == "variable_importance":
        tu.table_variable_importance(args, X, y, "table_variable_importance")