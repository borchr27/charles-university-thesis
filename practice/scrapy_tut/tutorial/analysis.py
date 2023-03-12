# coding=utf8
import re
import lzma
import math
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
import thesis_utils as tu
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")

##########################################################################################
# This code is used to run a simple analysis on the data that I collected from the websites
# using histograms, confusion matrices, etc. on either the site_data table or the 
# translated_data table. I also ran a simple gradient boosting classifier on the original 
# (not additional) translated_data.

def data_prep(data: tu.Dataset) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    sd_df = data.site_data
    td_df = data.translated_data
    data_df = pd.merge(td_df[['site_data_id', 'english_text']], \
                       sd_df[['id', 'category', 'origin']], left_on='site_data_id', right_on='id')
    filtered_df = data_df[data_df['origin'] == 'original']
    # print(filtered_df.value_counts('category'))
    train_data = filtered_df['english_text'].to_numpy()
    # for each row in train data replace it with this regex (separates camel case words like howAreYou to how Are You)
    train_data = np.array([re.sub(r'([a-z])([A-Z])', r'\1 \2', row) for row in train_data])
    vectorizer = tu.tfidf_vectorizer()
    X = vectorizer.fit_transform(train_data)
    y = filtered_df['category'].to_numpy()
    return X, y, vectorizer

def variable_importance(X:np.ndarray, y:np.ndarray, vectorizer:TfidfVectorizer) -> None:
    """
    This function is used to find the most important features in the dataset using a random forest regressor.
    
    Parameters:
    X: The tfidf vectorized data
    y: The labels for the data
    vectorizer: The vectorizer used to vectorize the data
    """

    # lable encoder (for gbdt)
    y_labels = np.unique(y)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X.toarray(), y)
    X = pd.DataFrame(X.toarray())
    importances = pd.Series(data=rf_model.feature_importances_, index=X.columns)
    importances_sorted = importances.sort_values(ascending=False)
    # Turn the importances back into words with tdidf vectorizer
    top = {}
    bottom = {}
    for i in range(20):
        top[vectorizer.get_feature_names()[importances_sorted.index[i]]] = importances_sorted.values[i]
        bottom[vectorizer.get_feature_names()[importances_sorted.index[-i-1]]] = importances_sorted.values[-i-1]
    # create a pandas df for the top 20 and one for the bottom 20
    top_df = pd.DataFrame.from_dict(top, orient='index', columns=['importance'])
    bottom_df = pd.DataFrame.from_dict(bottom, orient='index', columns=['importance'])
    top_df.to_latex(f"{tu.FILE_PATH}table_top_20_features.tex")
    bottom_df.to_latex(f"{tu.FILE_PATH}table_bottom_20_features.tex")


def get_weights(y:np.ndarray) -> dict:
    """
    This function is used to get the weights for the imbalanced data.
    """
    bin_counts = np.bincount(y)
    # Sort the bin counts in ascending order
    sorted_indices = np.argsort(bin_counts)
    sorted_bin_counts = bin_counts[sorted_indices]
    data_dict = {}
    for i in range(len(sorted_bin_counts)):
        data_dict[sorted_indices[i]] = sorted_bin_counts[i]
    
    def cosine_decay_weight(step, total_steps, initial_weight, final_weight):
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_steps))
        decayed = (1 - initial_weight) * cosine_decay + initial_weight
        return decayed * final_weight
    
    # Define the initial and final weights and the total number of steps/categories
    initial_weight = 0.1
    final_weight = 1.0
    total_steps = 23

    # Create a dictionary to store the weights
    weights = {}

    # Compute the weight for each step using the cosine decay function
    for n, i in enumerate(data_dict.keys()):
        weight = cosine_decay_weight(n, total_steps, initial_weight, final_weight)
        weights[i] = weight

    # Print the dictionary of weights
    # for w in weights:
    #     print(f"Step {w}: {weights[w]}")
    return weights

def test_LSVC_models(args: argparse.Namespace, X:np.ndarray, y:np.ndarray):
    clf_name = "LinearSVC"
    data_name = "original"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    weights = get_weights(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    pipe1 = Pipeline([
        (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights))
    ])
    pipe2 = Pipeline([
        (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, class_weight='balanced'))
    ])
    pipe3 = Pipeline([
        (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000))
    ])

    model1 = pipe1.fit(train_X, train_y)
    model2 = pipe2.fit(train_X, train_y)
    model3 = pipe3.fit(train_X, train_y)
    models = [model3, model2, model1]

    for model in models:
        pred = model.predict(test_X)
        print("Error:", 1 - metrics.accuracy_score(test_y, pred, normalize=True))


def build_and_test_model(args: argparse.Namespace, X:np.ndarray, y:np.ndarray, vectorizer:TfidfVectorizer, csv:bool=False, unigrams:bool=False, cm:bool=False, clf_report:bool=False):
    clf_name = "LinearSVC"
    data_name = "original"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)

    weights = get_weights(y)

    if unigrams:
        # make a dictionary with y labels as keys and their index as values
        category_to_id = dict(zip(y_labels, range(len(y_labels))))
        tu.table_correlated_unigrams(X, y, vectorizer, category_to_id, f"table_correlated_unigrams_{data_name}")
    if csv:
        tu.tfidf_to_csv(X.toarray(), y, "tfidf_data.csv")

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    pipe = Pipeline([
        # ("features", tu.tfidf_vectorizer()),
        # (f"{clf_name}", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
        (f"{clf_name}", LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights))
    ])

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
        tu.plot_confusion_matrix(clf, y_labels, pred, test_y, train_y, clf_name)
    if clf_report:
        tu.table_classification_report(test_y, pred, y_labels, clf_name)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = tu.Dataset()
    X, y, vectorizer = data_prep(data)
    # variable_importance(X, y, vectorizer)
    build_and_test_model(args, X, y, vectorizer, cm=True)
    # tu.plot_all_histograms(data, "plot_all_hist")
    # tu.plot_original_histograms(data, "plot_og_en_hist")
    # TODO: combine histogram plotting into one function
    # tu.plot_all_results_from_probal()
    # tu.table_data_category_counts(data)
    # tu.plot_explore_classifiers(args, X, y)
    # test_LSVC_models(args, X, y)