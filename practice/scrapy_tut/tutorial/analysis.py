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

# These arguments will be set appropriately by ReCodEx, even if you change them.
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed")

##########################################################################################
# This code is used to run a simple analysis on the data that I collected from the websites
# using histograms, confusion matrices, etc. on either the site_data table or the 
# translated_data table. I also ran a simple gradient boosting classifier on the original 
# (not additional) translated_data.

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

def build_and_test_model(args: argparse.Namespace):
    train = tu.Dataset("translated_data")    
    og_data_ind, add_data_ind = train.get_indicies()
    train_data = np.array(train.data)[og_data_ind]
    train_target = np.array(train.target)[og_data_ind]
    y_labels = np.unique(train_target)
    # selected_indicies = np.where((train_target != 'Atm') & (train_target != 'Children'))[0] #bc atm and children each have only 1 sample

    # lable encoder (for gbdt)
    le = LabelEncoder()
    le.fit(train_target)
    train_target = le.transform(train_target)

    # split data into train and test
    train_X, test_X, train_y, test_y = train_test_split(train_data, train_target, test_size=0.3, random_state=args.seed, stratify=train_target)

    # train the model
    pipe = Pipeline([
        ("features", tu.tfidf_vectorizer()),
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

    pred = model.predict(test_X)
    clf = model.named_steps["gbdt"]
    print("Accuracy:", metrics.accuracy_score(test_y, pred))
    tu.show_confusion_matrix(clf, y_labels, pred, test_y, train_y)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    # site_data_analysis(args)
    # translated_data_analysis(args)
    build_and_test_model(args)
