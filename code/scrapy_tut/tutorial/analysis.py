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
parser.add_argument("--filter", default="test", type=str, help="Choose the data to select for building csv.")
parser.add_argument("--plot", default=None, type=str, help=["all_histograms", "original_en_hist", "all_results_from_probal", "test_results", "explore_classifiers", "pr_curve", "test_results_averaged"])
parser.add_argument("--table", default=None, type=str, help=["correlated_unigrams", "classification_report", "data_category_counts", "variable_importance"])
parser.add_argument("--clf", default=None, type=str, help="Choose a classifier for the test model function. [KNN, LinearSVC, GBDT]")

##########################################################################################
# This code is used to run a simple analysis on the data that I collected from the websites
# using histograms, confusion matrices, etc. on either the site_data table or the 
# translated_data table. I also ran a simple gradient boosting classifier on the original 
# (not additional) translated_data.


def test_model(args, X:np.ndarray, y:np.ndarray, data_name:str, cm:bool=False, clf_name:str=None) -> float:
    assert clf_name in ["LinearSVC", "KNN", "GBDT"], "Please choose a valid classifier (LinearSVC, KNN, GBDT)"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    # weights = tu.get_weights(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    # print(np.unique(train_y, return_counts=True))
    # vectorize
    vect = tu.tfidf_vectorizer()
    train_X = vect.fit_transform(train_X)
    test_X = vect.transform(test_X)

    if clf_name == "LinearSVC":
        pipe = Pipeline([
            (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, )) #class_weight=weights))
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

    # commented out if running experiments where we build the model each run
    # try:
    #     print("Loading the model...")
    #     with lzma.open(f"model_{clf_name}.model" , "rb") as modelFile:
    #         model = pickle.load(modelFile)
    # except:
    #     print("Model could not be found, training a new model...")
    #     model = pipe.fit(train_X, train_y)
    #     with lzma.open(f"model_{clf_name}.model", "wb") as model_file: 
    #         pickle.dump(model , model_file)

    model = pipe.fit(train_X, train_y)

    pred = model.predict(test_X)
    clf = model.named_steps[clf_name]
    # if clf_name == "LinearSVC":
    #     # weight_array = np.array([weights[i] for i in test_y])
    #     error = 1 - metrics.accuracy_score(test_y, pred,)# sample_weight=weight_array)
    # else:
    #     error = 1 - metrics.accuracy_score(test_y, pred,)
    error = 1 - metrics.accuracy_score(test_y, pred,)
    print(f"{clf_name} Error:", error)
    if cm:
        tu.plot_confusion_matrix(y_labels, pred, test_y, clf_name+'_'+data_name)
    if args.table == "classification_report":
        tu.table_classification_report(test_y, pred, y_labels, clf_name)

    return error

def table_compare_top_three_models(args, X, y, data_name:str) -> None:
    nn_error = tu.build_tensor_flow_NN(args, X, y)
    knn_error = test_model(args, X, y, clf_name="KNN", data_name=data_name)
    lsvc_error = test_model(args, X, y, clf_name="LinearSVC", data_name=data_name)

    errors = pd.DataFrame({"Model": ["Tensor Flow Neural Network", "K Neighbors Classifier", "LinearSVC"], "Error": [nn_error, knn_error, lsvc_error]})
    errors = errors.sort_values(by="Error", ascending=True)
    errors.to_latex(f"{tu.IMG_FILE_PATH}table_{data_name}_data_best_errors.tex", index=False, float_format="%.3f")

def optimize_classifier(args, X, y, data_name:str):
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    # vectrorize the data
    vectorizer = tu.tfidf_vectorizer()
    train_X = vectorizer.fit_transform(train_X)
    test_X = vectorizer.transform(test_X)

    intercepts = np.linspace(0.1, 1, 20)
    c_vals = np.linspace(0.1, 1000, 20)
    base_classifier = LinearSVC(random_state=args.seed, max_iter=10000)
    bagging_classifier = BaggingClassifier(base_estimator=base_classifier, n_estimators=10, random_state=args.seed)
    params = {
        'base_estimator__random_state': [args.seed],
        'base_estimator__intercept_scaling': np.concatenate((intercepts, [1])),
        'base_estimator__loss': ['hinge', 'squared_hinge'],
        'base_estimator__penalty': ['l2'],
        'base_estimator__C': np.concatenate((c_vals, [1])),
        'base_estimator__multi_class': ['ovr', 'crammer_singer']
        }
    grid_search = GridSearchCV(estimator=bagging_classifier, param_grid=params, cv=5, verbose=2, scoring='accuracy', n_jobs=6)
    grid_search.fit(train_X, train_y)
    
    # Print the best parameters found by GridSearchCV
    print(f"best params: {grid_search.best_params_}")
    print(f"best score: {grid_search.best_score_}")
    print(f"best estimator: {grid_search.best_estimator_}")
    print(f"best index: {grid_search.best_index_}")
    # save best estimator data params to text file

    best_estimator = grid_search.best_estimator_
    pred = best_estimator.predict(test_X)    
    error = 1 - metrics.accuracy_score(test_y, pred)
    print(f"Error: {error}")
    results_df = pd.DataFrame.from_dict(grid_search.best_params_, orient='index', columns=['Value'])
    # append error to dataframe
    results_df = results_df.append(pd.DataFrame({'Value': [error]}, index=['test_error']))
    # results_df.to_csv(f"{tu.IMG_FILE_PATH}table_{data_name}_data_grid_search_results.csv", header=True, index_label='Parameter')
    results_df.to_latex(f"{tu.IMG_FILE_PATH}table_{data_name}_data_grid_search_results.tex", index=True)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data, X, y, vect = None, None, None, None
    data = tu.Dataset()
    
    X, y = tu.data_prep_fixed(data, origin_filter=None) # origin_filter=None for all data
    # X, y, vect = tu.data_prep(data, origin_filter='original')
    # tu.table_fisher_exact(args, X, y, 'original')
    # tu.plot_category_reduction_lscv(args, X, y, 'all')
    
    # optimize_classifier(args, X, y, 'all')
    # tu.plot_category_reduction_lscv(args, X, y, 'all')
    # table_compare_top_three_models(args, X, y, data_name='original')
    # test_model(args, X, y, data_name='all', clf_name="LinearSVC", cm=True)
    # tu.plot_probal_selection_dist("text_data_original_proper_vectorizer")

    # tu.build_tensor_flow_NN_upgraded(args, X, y)
    # tu.build_tensor_flow_LSTM(args, X, y)
    # tu.export_text_data_to_csv(X, y, None)
    # tu.plot_LSCV_varying_min_category(args, X, y)
    # tu.plot_pr_curve(args, X, y, data_name="data_filtered_155")
    # test_model(args, X, y, vectorizer, data_name='original', clf_name="LinearSVC")
    # tu.build_LSVC_models(args, X, y)
    # tu.build_tensor_flow_LSTM(args, X, y)
    # tu.plot_test_results_averaged(folder='all_data_experiments')
    # tu.plot_data_length_grid_search(args, data)


    if args.plot == "all_histograms":
        tu.plot_all_histograms(data, "plot_all_hist")
    if args.plot == "original_en_hist":
        tu.plot_original_en_histograms(data, "plot_original_english_counts")
    if args.plot == "all_results_from_probal":
        tu.plot_all_results_from_probal()
    if args.plot == "test_results":
        tu.plot_probal_test_results()
    if args.plot == "test_results_averaged":
        folder_name = "text_data_original_proper_vectorizer_50_st_filter"
        print(f'Make sure the correct folder has been selected. [{folder_name}] folder selected.')
        tu.plot_probal_test_results_averaged(folder=folder_name)
    if args.plot == "explore_classifiers":
        tu.plot_explore_classifiers(args, X, y)
    if args.plot == "pr_curve":
        tu.plot_pr_curve(args, X, y)
    if args.plot == "data_length_grid_search":
        tu.plot_data_length_grid_search(args, data)
    if args.plot == "data_text_analysis":
        # make sure to use fixed_data_prep method
        tu.plot_data_text_analysis(args, X, y)
    if args.plot == "probal_selection_dist":    
        tu.plot_probal_selection_dist("text_data_all_proper_vectorizer")
    if args.plot == "category_reduction_probal":
        tu.plot_category_reduction_probal(args)
    

    if args.table == "category_reduction_lscv":
        tu.table_category_reduction_lsvc(args, X, y, 'all')
    if args.table == "correlated_unigrams":
        data_name = "original"
        tu.table_correlated_unigrams(args, X, y, None, data_name)
    if args.table == "data_category_counts":
        tu.table_category_counts(data, "table_category_counts")
    if args.table == "variable_importance":
        # use with fixed data prep
        tu.table_variable_importance(args, X, y)
    if args.table == 'cosine_decay_weights':
        tu.table_cosine_decay_weights(args, y)

    if args.csv:
        tu.build_csv(args, X, y, 'all_data_filtered')