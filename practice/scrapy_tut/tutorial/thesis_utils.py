from typing import List
from matplotlib import pyplot as plt
import psycopg2
import re
import numpy as np
import pycld2 as cld2
from sklearn import metrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from sqlalchemy import create_engine
import os
from sklearn.feature_selection import chi2
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelEncoder

import configparser
import requests
from dotenv import load_dotenv

# Settings to create pdf plots for thesis
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

FILE_PATH = "/Users/mitchellborchers/Documents/git/charles-university-thesis/thesis/vzor-dp/img/"

logging.basicConfig(filename='thesis.log', level=logging.DEBUG, format='%(asctime)s:%(levelname)s:%(message)s')

##########################################################################################
# This file has methods and classes that I used throughout the project. I have tried to
# organize them into sections based on what they are used for.

class Website:
    def __init__(self, id, name, i_url, v_url, text, category, tags, origin, language=None):
        self.id = id
        self.name = name
        self.i_url = i_url
        self.v_url = v_url
        self.text = text
        self.category = category
        self.tags = tags
        self.origin = origin
        self.language = language

class Dataset:
    def __init__(self):
        self.translated_data = None
        self.site_data = None

        db = PostgresDatabase()
        db.connect()
        if db.connection:
            self.translated_data = pd.read_sql("SELECT * FROM translated_data;", db.connection)
            self.site_data = pd.read_sql("SELECT * FROM site_data;", db.connection)
        db.close()            
    
    def get_indicies(self) -> tuple[list[int], list[int]]:
        """Get the indicies of the original data and additional data."""
        og_data_indicies = []
        add_data_indicies = []
        for item in self._class:
            if item.data_category == "Original Data":
                og_data_indicies.append(item.id)
            else:
                add_data_indicies.append(item.id)
        return og_data_indicies, add_data_indicies


class PostgresDatabase:
    def __init__(self) -> None:
        self.connection = None
        self.cursor = None

    def connect(self):
        """
        Connect to the postgres database. To view the database or debug open up the shell for the 
        database image then use the 'psql -U docker maindb' command to enter into the database bash. Then
        use the '\l' command to list the databases.
        """
        try:
            conn = psycopg2.connect(database="postgres", user="postgres", password="postgres", host="localhost")
            self.connection = conn
        except:
            pass

        # Create quotes table if none exists
        # self.execute("""
        # CREATE TABLE IF NOT EXISTS site_data (
        #     id serial PRIMARY KEY not null, 
        #     name varchar not null,
        #     initial_url varchar not null, 
        #     visited_url varchar not null, 
        #     text varchar not null,
        #     category varchar not null,
        #     tags varchar not null,
        #     timestamp timestamp default current_timestamp
        #     );
        # """)

    def close(self) -> None:
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def commit(self) -> None:
        """Commit the changes to the database."""
        assert self.connection, 'Database connection is not established'
        if self.connection:
            self.connection.commit()

    def execute(self, command, values=None) -> None:
        """Execute a command on the database."""
        assert self.connection, 'Database connection is not established'
        if self.connection:
            self.cursor = self.connection.cursor()
            self.cursor.execute(command, values)

    def is_translated(self, id:int) -> tuple[bool, int]:
        """Check if the ID already exists in the database."""
        self.execute("""SELECT site_data_id FROM translated_data WHERE site_data_id = %s;""", (id,))
        result = self.cursor.fetchone()
        if result: return True, id
        return False, None


    def add_translation(self, site:Website) -> None:
        """Add a translation to the translated_data table in the database."""
        result, id = self.is_translated(site.id)
        if result:
            # If the ID already exists, log an error message
            logging.info(f"Error: ID {result} already exists in the database")
        else:
            # If the ID does not exist, proceed with the insert  
            self.execute("""INSERT INTO translated_data (original_language, english_text, site_data_id) VALUES (%s, %s, %s);""", (
                site.language,
                site.text,
                site.id,))
        self.commit()

    def add_website(self, site: Website) -> None:
        """Add a website to the site_data table in the database. """
        if site.i_url == "missing note" or site.v_url == "" or site.text == "":
            return
        
        # Check to see if text is already in database
        self.execute(
            "SELECT * FROM site_data WHERE initial_url = %s;", (site.i_url,))
        result = self.cursor.fetchone()

        # If it is in DB, create log message
        if result:
            print(f"Item already in database: {site.name}")
        else:
            # Define insert statement
            self.execute("""INSERT INTO site_data (name, initial_url, visited_url, text, category, tags, origin) VALUES (%s,%s,%s,%s,%s,%s,%s);""", (
                site.name,
                site.i_url,
                site.v_url,
                site.text,
                site.category,
                site.tags,
                site.origin,
            ))
        self.commit()


def get_language(data:np.ndarray, language_array:np.ndarray = np.array([None])) -> np.ndarray:
    """Try and get language of each data point in the given list of data points."""
    a = len(data)
    for i in range(len(data)):
        isReliable, textBytesFound, details, vectors = cld2.detect(data[i], returnVectors=True)
        if len(vectors) != 0 and isReliable:
            for v in vectors:
                if v[3] != 'un':
                    language_array[i] = v[3]
                    break
        if language_array[i] == 'un' or language_array[i] == None:
            language_array[i] = 'un'
    return language_array

def clean_text_data(data:np.ndarray) -> np.ndarray:
    """Cleans the data by removing all non-alphanumeric characters and making all characters lowercase."""
    # preprocess data with regex, remove numbers and symbols
    for i in range(len(data)):
        text = data[i]
        text = re.sub(r'[^\w\s]+', ' ', text, flags=re.UNICODE)
        text = re.sub('\d', ' ', text)
        text = re.sub("\s{2,}", ' ', text)
        # text = re.sub(r'(?<!^)(?=[A-Z])', ' ', text)
        data[i] = text[:999]
    return data

def site_data_filter(train_data:np.ndarray, languages:np.ndarray = None, debugging:bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses the data and returns the indicies of the selected languages and the language array."""
    lang_array = [None] * len(train_data)
    lang_array = np.array(lang_array)

    # train_data = clean_text_data(train_data)
    lang_array = get_language(train_data, lang_array)

    # get indicies of specified languages data
    selected_language_indicies = []
    if languages == None:
        selected_language_indicies = [i for i in range(len(lang_array))]
    else:
        for l in languages:
            selected_language_indicies += np.where(lang_array == l)[0].tolist()
        selected_language_indicies = np.array(sorted(selected_language_indicies))
    unknown_indicies = np.where(lang_array == 'un')

    # prints unknown language train data
    if debugging:    
        unk = np.array(train_data)[unknown_indicies]
        for d in unk:
            print(d)
            print(80*"-")

    # returning lang_array just to use in histogram
    return selected_language_indicies, lang_array

def tfidf_vectorizer() -> TfidfVectorizer:
    """Function to return a TF-IDF vectorizer."""
    return TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=5000, stop_words="english")

def tfidf_to_csv(text_data:np.ndarray, y:np.ndarray, file_name:str = "tfidf_data.csv") -> None:
    """Converts TF-IDF data to a CSV file to be used in the xPAL and other algorithms."""
    # get the tfidf vectorizer
    # vectorizer = tfidf_vectorizer()
    file_path = '/Users/mitchellborchers/Documents/git/probal/data/' + file_name
    tfidf = text_data
    # Fit and transform the text data
    # tfidf = vectorizer.fit_transform(text_data)
    # tfidf = tfidf.toarray()
    y = np.array(y)
    # concatenate tfidf and y
    tfidf = np.concatenate((tfidf, y.reshape(-1, 1)), axis=1)

    # Get the feature names
    # feature_names = vectorizer.get_feature_names_out()
    # create feature names in the form of x_0 to x_n
    feature_names_xs = ["x_" + str(i) for i in range(len(text_data[1]))]
    # append y to feature names
    feature_names_xs = np.append(feature_names_xs, "y")

    # Convert the TF-IDF data to a Pandas DataFrame
    tfidf_df = pd.DataFrame(tfidf, columns=feature_names_xs)

    # Export the DataFrame to a CSV file
    tfidf_df.to_csv(file_path, index=False)


def azure_translate(text:str, source_language:str = None, target_language:str ='en') -> str:
    """Use free Azure Cognitive Services to translate text to English.
    text: text to translate
    source_language: language of the text to translate
    target_language: language to translate the text to

    returns: translated text

    Helpful link https://techcommunity.microsoft.com/t5/educator-developer-blog/translate-your-notes-with-azure-translator-and-python/ba-p/3267201
    """

    config = configparser.ConfigParser()
    config.read('/Users/mitchellborchers/.config/azure/my_config_file.ini')
    auth_key = config['cognitive_services']['auth_key']
    location = config['cognitive_services']['location']

    load_dotenv()
    key = auth_key
    region = location
    endpoint = 'https://api.cognitive.microsofttranslator.com/'

    def _translate(text, source_language, target_language, key, region, endpoint):
        # Use the Translator translate function
        url = endpoint + '/translate'
        # Build the request
        params = {
            'api-version': '3.0',
            'from': source_language,
            'to': target_language
        }
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': region,
            'Content-type': 'application/json'
        }
        body = [{
            'text': text
        }]
        # Send the request and get response
        request = requests.post(url, params=params, headers=headers, json=body)
        response = request.json()
        # Get translation
        translation = response[0]["translations"][0]["text"]
        return translation
    
    return _translate(text, source_language, target_language, key, region, endpoint)


def save_plot_image(plot:plt, file_name:str) -> None:
    """Saves the given plot to a file with the given filename to the thesis directory.
    
    plot: plot to save
    filename: name of the file to save the plot to
    """
    plt.savefig(f'{FILE_PATH}{file_name}.pdf')


def plot_confusion_matrix(clf, y_labels, pred, test_target, train_target, name:str) -> None:
    """Plots a confusion matrix for the given classifier.
    
    clf: classifier
    y_labels: labels for the y axis
    pred: predictions
    test_target: test target
    train_target: train target   
    """
    cm = metrics.confusion_matrix(test_target, pred, labels=clf.classes_)
    unique_categories, category_counts = np.unique(train_target, return_counts=True)
    category_bins = [i for i in range(len(category_counts))]
    # category_labels = [i[:8] for i in unique_categories]
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
    disp.plot()
    # make x axis labels smaller
    plt.xticks(category_bins, y_labels, rotation='vertical', fontsize=8)
    plt.yticks(category_bins, y_labels, fontsize=8)
    plt.tight_layout()
    # plt.show()
    save_plot_image(plt, f"plot_confusion_matrix_{name}")
    plt.close()


def plot_all_histograms(data:Dataset, filename:str) -> None:
    """Plot the language and categorical histograms for the original site data AND the translated data.
    
    data: Dataset object
    filename: filename to save the plot to
    """
    td_df = data.translated_data
    sd_df = data.site_data
    m_df = pd.merge(td_df[['site_data_id', 'original_language']], sd_df[['id', 'category','origin']], left_on='site_data_id', right_on='id')
    original = m_df[m_df['origin']=='original']
    additional = m_df[m_df['origin']=='additional']
    categories = m_df['category'].unique()

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    bar_width = 1
    opacity = 0.5

    category_labels = sorted(categories, key=lambda s: s.split()[0])
    df1_c = original['category'].value_counts()
    df2_c = additional['category'].value_counts()
    # Add missing categories and fill with zeros
    missing_categories = set(categories) - set(df1_c.index)
    for category in missing_categories:
        # df1_c = df1_c.append(pd.Series([0], index=[category]))
        df1_c = pd.concat([df1_c, pd.Series([0], index=[category])])
    missing_categories = set(categories) - set(df2_c.index)
    for category in missing_categories:
        # df2_c = df2_c.append(pd.Series([0], index=[category]))
        df2_c = pd.concat([df2_c, pd.Series([0], index=[category])])
    # Sort by category
    df1_c = df1_c.sort_index()
    df2_c = df2_c.sort_index()
    # print(df1_c)
    ax[0].bar(df1_c.index, df1_c.values, bar_width, alpha=opacity, color='b', label='Original Data')
    ax[0].bar(df2_c.index, df2_c.values, bar_width, alpha=opacity, color='r', label='Additional Data', bottom=df1_c.values)
    ax[0].set_xlabel('Categories')
    ax[0].set_ylabel('Count')
    ax[0].set_ylim([0, max(df1_c.max(), df2_c.max()) * 1.1])
    ax[0].set_xticks(np.arange(len(categories)))
    ax[0].set_xticklabels(category_labels, rotation=90, fontsize=8)
    ax[0].legend()

    languages = m_df['original_language'].unique()
    language_labels = sorted(languages, key=lambda s: s.split()[0])
    df1_l = original['original_language'].value_counts()
    df2_l = additional['original_language'].value_counts()
    # Add missing categories and fill with zeros
    missing_languages = set(languages) - set(df1_l.index)
    for language in missing_languages:
        df1_l = pd.concat([df1_l, pd.Series([0], index=[language])])
    missing_languages = set(languages) - set(df2_l.index)
    for language in missing_languages:
        df2_l = pd.concat([df2_l, pd.Series([0], index=[language])])
    # Sort by category
    df1_l = df1_l.sort_index()
    df2_l = df2_l.sort_index()
    ax[1].bar(df1_l.index, df1_l.values, bar_width, alpha=opacity, color='b', label='Original Data')
    ax[1].bar(df2_l.index, df2_l.values, bar_width, alpha=opacity, color='r', label='Additional Data', bottom=df1_l.values)
    ax[1].set_xlabel('Languages')
    ax[1].set_ylabel('Count')
    ax[1].set_xticks(np.arange(len(languages)))
    ax[1].set_xticklabels(language_labels, fontsize=7, ha='center')
    ax[1].legend()
    fig.tight_layout()
    # plt.show()
    save_plot_image(plt, filename)
    plt.close(fig)


def plot_original_histograms(data:Dataset, filename:str) -> None:
    """Plot the lannguage and categorical histograms for the original english site data.
    
    data: Dataset object
    filename: filename to save the plot to

    """
    td_df = data.translated_data
    sd_df = data.site_data
    m_df = pd.merge(td_df[['site_data_id', 'original_language']], sd_df[['id', 'category','origin']], left_on='site_data_id', right_on='id')
    original = m_df[(m_df['origin'] == 'original') & (m_df['original_language'] == 'en')]
    categories = m_df['category'].unique()

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 1
    opacity = 0.5

    df1_c = original['category'].value_counts()
    # Add missing categories and fill with zeros
    missing_categories = set(categories) - set(df1_c.index)
    for category in missing_categories:
        # df1_c = df1_c.append(pd.Series([0], index=[category]))
        df1_c = pd.concat([df1_c, pd.Series([0], index=[category])])
    # Sort by category
    df1_c = df1_c.sort_index()
    category_labels = sorted(categories, key=lambda s: s.split()[0])
    ax.bar(df1_c.index, df1_c.values, bar_width, alpha=opacity, color='b', label='Original Data')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(category_labels, rotation=90, fontsize=10)
    ax.legend()
    fig.tight_layout()
    # plt.show()
    save_plot_image(plt, filename)
    plt.close(fig)
    # df1_c shows counts for the plot sum is 275


def plot_all_results_individual() -> None:
    """
    Plots all data results from the probal results folder into thier own individual plots.
    
    """
    file_path = '/Users/mitchellborchers/Documents/git/probal/results/'
    file_names = os.listdir(file_path)
    for file_name in file_names:
        if not file_name.endswith('.csv'): continue
        loc = os.path.join(file_path, file_name)
        with open(loc, 'r') as f:
            data = pd.read_csv(f)
            # remove .csv from file name
            name = file_name[:-4]
            # create line plot train and test error using fig and ax
            fig, ax = plt.subplots()
            ax.plot(data['train-error'], label='Train Error')
            ax.plot(data['test-error'], label='Test Error')
            ax.set_xlabel('Budget')
            ax.set_ylabel('Error')
            ax.legend()
            save_plot_image(plt, name)
            plt.close()

def plot_all_results_from_probal() -> None:
    """
    Plots all test data from the probal results folder into one plot based on the kernel. Takes no arguments.
    
    """
    file_path = '/Users/mitchellborchers/Documents/git/probal/results/'
    file_names = os.listdir(file_path)
    file_names = [s for s in file_names if not s.startswith('.')]
    count = 2
    
    kernel_name = ['rbf', 'cosine']
    rbf = []
    cosine = []
    for s in file_names:
        if re.search('rbf.*612|612.*rbf', s):
            rbf.append(s)
        elif re.search('cosine.*612|612.*cosine', s):
            cosine.append(s)
    
    rbf = sorted(rbf)
    cosine = sorted(cosine)

    kernels = [rbf, cosine]
    for n, k in zip(kernel_name, kernels):
        # create figure with four subplots
        fig, axs = plt.subplots(count, count)
        i=0
        for file_name, ax in zip(k, axs.flatten()):
            if not file_name.endswith('.csv'): continue
            loc = os.path.join(file_path, file_name)
            pattern = r'data_([a-zA-Z0-9]+)[\-_]'
            plt_title = re.findall(pattern, file_name)[0]

            with open(loc, 'r') as f:
                data = pd.read_csv(f)
                ax.plot(data['train-error'], label='Train Error')
                ax.plot(data['test-error'], label='Test Error')
                if i ==0: ax.legend()
                if i==0 or i==2: ax.set_ylabel('Error')
                if i==2 or i==3: ax.set_xlabel('Budget')
                ax.set_ylim([0, 1.1])
                ax.set_title(plt_title)
                ax.grid(which='both', linewidth=0.3)
                i+=1

        fig.tight_layout()
        save_plot_image(plt, f'plot_all_results_{n}')
        plt.close()

def plot_explore_classifiers(args, X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    models = [
        LinearSVC(),
        MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=1000),
        SVC(kernel='precomputed'),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
        LogisticRegression(random_state=0),
        MultinomialNB(),
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    ]

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        if model_name == 'SVC':
            train_X_cosine = pairwise_kernels(X, metric='cosine')
            accuracies = cross_val_score(model, train_X_cosine, y, scoring='accuracy', cv=CV)
        else:
            accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
            # use error instead of accuracy
        errors = 1 - accuracies
        for fold_idx, error in enumerate(errors):
            entries.append((model_name, fold_idx, error))

    cv_df = pd.DataFrame(entries, columns=['Classifier', 'fold_idx', 'Error'])
    sns.boxplot(x='Classifier', y='Error', data=cv_df)
    plt.xticks(rotation=90)
    plt.tight_layout()
    save_plot_image(plt, 'plot_explore_classifiers')
    plt.close()


def table_correlated_unigrams(X:np.ndarray, y:np.ndarray, v:TfidfVectorizer, category_to_id:dict, file_name:str) -> None:
    """
    Creates latex .tex table file with of features with the highest chi-squared statistics per target class.

    X: sparse matrix of shape (n_samples, n_features)
    y: array of shape (n_samples,)
    v: vectorizer
    category_to_id: dictionary with category names mapped to their numeric id
    """
    N=3 # number of features to print
    uni_dict = {}
    for label, id in sorted(category_to_id.items()):
        features_chi2 = chi2(X, y == id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(v.get_feature_names_out())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        uni_dict[label] = unigrams[-N:]
        # print("# '{}':".format(label))
        # print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    # Convert the dictionary to a dataframe
    df = pd.DataFrame.from_dict(uni_dict, orient='index', columns=['Keyword 1', 'Keyword 2', 'Keyword 3'])
    df.to_latex(f"{FILE_PATH}{file_name}.tex")


def table_classification_report(test, pred, labels, clf_name_and_info:str) -> None:
    """
    Create latex .tex table file with classification report.
    
    test: array of shape (n_samples,)
    pred: array of shape (n_samples,)
    labels: array of shape (n_samples,)
    clf_name_and_info: string with classifier name and info
    """
    report = metrics.classification_report(test, pred, target_names=labels, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_latex(f"{FILE_PATH}table_classification_report_{clf_name_and_info}.tex", float_format="%.2f")


def table_data_category_counts(data:Dataset, group:List[str] = ['original', 'additional']) -> None:
    """
    Creates latex .tex table file with 'original' or 'additional' data counts.
    
    """
    td_df = data.translated_data
    sd_df = data.site_data
    for g in group:
        # Merge the two DataFrames on the 'id' column
        merged_data = pd.merge(sd_df, td_df, how='inner', left_on='id', right_on='site_data_id')
        # Filter the merged DataFrame to only include rows where the 'origin' column == 'original'
        filtered_data = merged_data[merged_data['origin'] == g]
        # Group the filtered DataFrame by the 'category' column and count the number of rows in each group
        grouped_data = filtered_data.groupby('category').size().reset_index(name='category count')
        # export to latex without indices
        df = pd.DataFrame(grouped_data)
        df.to_latex(f"{FILE_PATH}table_{g}_category_counts.tex", index=False)
