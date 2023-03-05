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

    # lable encoder (for gbdt)
    le = LabelEncoder()
    le.fit(train_target)
    train_target = le.transform(train_target)

    # transform the data with tfidf
    train_data = tu.tfidf_vectorizer().fit_transform(train_data)
    tu.tfidf_to_csv(train_data.toarray(), train_target, "tfidf_data.csv")

    # split data into train and test
    train_X, test_X, train_y, test_y = train_test_split(train_data, train_target, test_size=0.3, random_state=args.seed, stratify=train_target)

    # train the model
    pipe = Pipeline([
        # ("features", tu.tfidf_vectorizer()),
        ("gbdt", GradientBoostingClassifier(random_state=args.seed, max_depth=4, n_estimators=200, verbose=1))
    ])

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
    tu.plot_confusion_matrix(clf, y_labels, pred, test_y, train_y)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    data = tu.Dataset()
    build_and_test_model(args, data)
    tu.plot_all_histograms(data, "plot_all_hist")
    tu.plot_original_histograms(data, "plot_og_en_hist")
    tu.plot_all_results_individual()
    tu.plot_all_results_one()