# coding=utf8
import lzma
import pickle
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import thesis_utils as tu
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
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

def build_and_test_model(args: argparse.Namespace, data: tu.Dataset):
    sd_df = data.site_data
    td_df = data.translated_data
    data_df = pd.merge(td_df[['site_data_id', 'english_text']], \
                       sd_df[['id', 'category', 'origin']], left_on='site_data_id', right_on='id')
    filtered_df = data_df[data_df['origin'] == 'original']
    # print(filtered_df.value_counts('category'))
    train_data = filtered_df['english_text'].to_numpy()
    train_target = filtered_df['category'].to_numpy()
    y_labels = np.unique(train_target)
    # make a dictionary with y labels as keys and their index as values
    category_to_id = dict(zip(y_labels, range(len(y_labels))))

    # lable encoder (for gbdt)
    le = LabelEncoder()
    le.fit(train_target)
    train_target = le.transform(train_target)

    # for each row in train data replace it with this regex (separates camel case words like howAreYou to how Are You)
    train_data = np.array([re.sub(r'([a-z])([A-Z])', r'\1 \2', row) for row in train_data])
    vectorizer = tu.tfidf_vectorizer()
    X = vectorizer.fit_transform(train_data)
    clf_name = "gbdt"
    data_name = "original"

    tu.table_correlated_unigrams(X, train_target, vectorizer, category_to_id, f"table_correlated_unigrams_{data_name}")
    tu.tfidf_to_csv(X.toarray(), train_target, "tfidf_data.csv")
    train_X, test_X, train_y, test_y = train_test_split(X, train_target, test_size=0.3, random_state=args.seed, stratify=train_target)
    pipe = Pipeline([
        # ("features", tu.tfidf_vectorizer()),
        (f"{clf_name}", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
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
    clf = model.named_steps["gbdt"]
    print("Accuracy:", metrics.accuracy_score(test_y, pred, normalize=True))
    tu.plot_confusion_matrix(clf, y_labels, pred, test_y, train_y)
    tu.table_classification_report(test_y, pred, y_labels, f"table_classification_report_{clf_name}")




if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = tu.Dataset()
    # build_and_test_model(args, data)
    # tu.plot_all_histograms(data, "plot_all_hist")
    # tu.plot_original_histograms(data, "plot_og_en_hist")
    tu.plot_all_results_one()
    # tu.plot_all_results_individual()