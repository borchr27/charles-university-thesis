import os
import math
from typing import List
from matplotlib import pyplot as plt
import tensorflow as tf
import psycopg2
import re
import numpy as np
import pycld2 as cld2
import seaborn as sns
import pandas as pd
import logging
from dotenv import load_dotenv
import configparser
import requests
from sqlalchemy import create_engine
from analysis import test_model

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics.pairwise import pairwise_kernels, cosine_distances
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.multiclass import OneVsRestClassifier
from matplotlib.colors import LogNorm


# Settings to create pdf plots for thesis
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

CONFIG = configparser.ConfigParser()
CONFIG.read('/Users/mitchellborchers/.config/azure/my_config_file.ini')
# IMG_FILE_PATH = "/Users/mitchellborchers/Documents/git/charles-university-thesis/thesis/vzor-dp/img/"
IMG_FILE_PATH = CONFIG['cognitive_services']['img_file_path']

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
    # change to 9000 ...11/4/2023
    return TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=9000, stop_words="english")

def export_tfidf_to_csv(text_data:np.ndarray, y:np.ndarray, file_name:str = "tfidf_data.csv") -> None:
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

def export_text_data_to_csv(X, y, file_name="TESTER.csv"):
    """
    Converts text data to a CSV file to be used in with Pobal repo.
    
    """
    file_path = '/Users/mitchellborchers/Documents/git/probal/data/' + file_name

    # create dataframe with x and y data
    data = {'x': X, 'y': y}
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    # print(df.head())

def azure_translate(text:str, source_language:str = None, target_language:str ='en') -> str:
    """Use free Azure Cognitive Services to translate text to English.
    text: text to translate
    source_language: language of the text to translate
    target_language: language to translate the text to

    returns: translated text

    Helpful link https://techcommunity.microsoft.com/t5/educator-developer-blog/translate-your-notes-with-azure-translator-and-python/ba-p/3267201
    """
    auth_key = CONFIG['cognitive_services']['auth_key']
    location = CONFIG['cognitive_services']['location']

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

def fixed_data_prep(data:Dataset, origin_filter:str = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Improved data_prep method. Dont vectorize the data here. Lesson learned 11/4. Returns strings.

    Parameters
    ----------
    data : Dataset
        Dataset object
    origin_filter : str, optional
        Filter the data by origin, by default None

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X, y
    """
    sd_df = data.site_data
    td_df = data.translated_data
    data_df = pd.merge(td_df[['site_data_id', 'english_text']], \
                       sd_df[['id', 'category', 'origin']], left_on='site_data_id', right_on='id')
    

    if origin_filter != None:
        filtered_df = data_df[data_df['origin'] == origin_filter]
        data_df = data_df['english_text'].to_numpy()
    else:
        filtered_df = data_df
        data_df = data_df['english_text'].to_numpy()
    
    # filter by the min number of items in a category
    # fdf = filtered_df.groupby('category').filter(lambda x: x['category'].value_counts() > 50)
    # print(filtered_df['category'].value_counts())

    X = filtered_df['english_text'].to_numpy()
    y = filtered_df['category'].to_numpy()
    if len(np.unique(y)) != 23:
        print("All 23 categories are not represented in the data.")

    # for each row in train data replace it with this regex (separates camel case words like howAreYou to how Are You)
    X = np.array([re.sub(r'([a-z])([A-Z])', r'\1 \2', row) for row in X])
    return X, y

def data_prep(data:Dataset, origin_filter:str = 'additional', min_str_len:int=0, max_str_len:int=1000) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    sd_df = data.site_data
    td_df = data.translated_data
    data_df = pd.merge(td_df[['site_data_id', 'english_text']], \
                       sd_df[['id', 'category', 'origin']], left_on='site_data_id', right_on='id')
    
    data_df = data_df[data_df['english_text'].str.len() > min_str_len]
    data_df = data_df[data_df['english_text'].str.len() < max_str_len] # max 1000
    # print the category counts
    # category_counts = data_df['category'].value_counts()
    # print(category_counts)
    
    # y = data_df['category'].to_numpy()

    if origin_filter != None:
        filtered_df = data_df[data_df['origin'] == origin_filter]
        data_df = data_df['english_text'].to_numpy()
    else:
        filtered_df = data_df
        data_df = data_df['english_text'].to_numpy()
    
    y = filtered_df['category'].to_numpy()
    assert len(np.unique(y)) == 23, "All 23 categories are not represented in the data."
    le = LabelEncoder()
    le.fit(y)
    
    # print(filtered_df.value_counts('category'))
    train_data = filtered_df['english_text'].to_numpy()
    # for each row in train data replace it with this regex (separates camel case words like howAreYou to how Are You)
    train_data = np.array([re.sub(r'([a-z])([A-Z])', r'\1 \2', row) for row in train_data])
    vectorizer = tfidf_vectorizer()
    vectorizer.fit(train_data)
    X = vectorizer.transform(train_data)
    # y = le.transform(filtered_df['category'].to_numpy()) # may need to comment this out when making plots to have labels be correct
    return X, y, vectorizer

def build_csv(args, X:np.ndarray, y:np.ndarray, name:str=None) -> None:
    """
    Builds a csv file from the TF-IDF data to be used in the xPAL and other algorithms.
    """
    
    # y_labels = np.unique(y)
    # le = LabelEncoder()
    # y = le.fit_transform(y)
    export_tfidf_to_csv(X.toarray(), y, f"{name}.csv")

def build_LSVC_models(args, X:np.ndarray, y:np.ndarray) -> None:
    """
    Builds three different LinearSVC models with different parameters and compares them.
    
    """
    clf_name = "LinearSVC"

    # lable encoder
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)

    weights = get_weights(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    model1 = LinearSVC(random_state=args.seed, max_iter=10000, multi_class='crammer_singer').fit(train_X, train_y)
    model2 = LinearSVC(random_state=args.seed, max_iter=10000).fit(train_X, train_y)
    model3 = LinearSVC(random_state=args.seed, max_iter=10000, class_weight=weights).fit(train_X, train_y)

    weight_array = np.array([weights[i] for i in test_y])
    error1 = 1 - metrics.accuracy_score(test_y, model1.predict(test_X),)
    error2 = 1 - metrics.accuracy_score(test_y, model2.predict(test_X),)
    error3 = 1 - metrics.accuracy_score(test_y, model3.predict(test_X), sample_weight=weight_array)

    #create df for errors   
    errors = pd.DataFrame({'Model': ['Crammer Singer', 'Boilerplate One-vs-rest ', 'Cosine Decay Weights One-vs-rest'], 'Error': [error1, error2, error3]})
    errors = errors.sort_values(by='Error', ascending=True)
    # save as latex table
    errors.to_latex(f'{IMG_FILE_PATH}table_lsvc_errors.tex', index=False, float_format="%.3f")

def build_tensor_flow_LSTM(args, X:np.ndarray, y:np.ndarray) -> None:
    from nltk.corpus import stopwords
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    Z = tf.strings.regex_replace(X, r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*',"").numpy()
    Z = [z.decode('utf-8') for z in Z]

    max_features = 5000
    max_len = 100
    batch_size = 32

    # Create a tokenizer object
    tokenizer = Tokenizer(num_words=5000)
    # Fit the tokenizer on your text data
    tokenizer.fit_on_texts(Z)
    # Convert the text data to sequences of integers
    sequences = tokenizer.texts_to_sequences(Z)
    # pad sequences
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    vocab_size = len(tokenizer.word_index) + 1
    one_hot_padded_sequences = np.zeros((len(padded_sequences), max_len, vocab_size), dtype=np.float32)
    for i, seq in enumerate(padded_sequences):
        one_hot_padded_sequences[i, np.arange(max_len), seq] = 1

    # Convert the padded sequences to word embeddings
    embedding_dim = 100
    embedding_matrix = np.random.rand(vocab_size, embedding_dim)
    embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), trainable=False)
    embedded_padded_sequences = embedding_layer(padded_sequences)

    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    # one hot encode the y text labels to be used in the model
    y = tf.keras.utils.to_categorical(y)

    # train_X, test_X, train_y, test_y = train_test_split(embedded_padded_sequences, y, test_size=0.25, random_state=args.seed, stratify=y)
    # train_X = train_X.toarray()
    # test_X = test_X.toarray()

    model = tf.keras.models.Sequential([
        embedding_layer,
        tf.keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        tf.keras.layers.Dense(len(y_labels), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(jit_compile=False),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy],)
    model.fit(embedded_padded_sequences, y, batch_size=batch_size, epochs=2)

    # score = model.evaluate(test_X, test_y, batch_size=batch_size)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])



def build_tensor_flow_NN(args, X:np.ndarray, y:np.ndarray) -> float:
    """
    Builds a simple neural network using Tensor Flow.
    """

    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    train_X = train_X.toarray()
    test_X = test_X.toarray()

    try:
        # Try to load tf model
        model = tf.keras.models.load_model(args.model)
        print("Model loaded from disk")
    except:
        print("Model not found, building new model")
        # Build the model
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=[train_X.shape[1]]),
            tf.keras.layers.Dense(args.hidden_layer, activation=tf.nn.sigmoid),
            tf.keras.layers.Dense(len(y_labels), activation=tf.nn.softmax),
        ])

        optimizer = tf.keras.optimizers.Adamax(jit_compile=False,)
        optimizer.learning_rate = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=args.learning_rate,
                decay_steps=train_X.shape[0] // args.batch_size * args.epochs,
                alpha=args.learning_rate_final / args.learning_rate,
            )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy],
        )
        model.fit(train_X, train_y, batch_size=args.batch_size, epochs=args.epochs)
        # # Save the model
        model.save(args.model)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_X, test_y)

    pred = model.predict(test_X)
    pred = np.argmax(pred, axis=1)
    
    plot_confusion_matrix(y_labels, pred, test_y, 'NN')
    error = 1-test_acc
    print(f"NN Error:", error)
    return error


def save_plot_image(plot:plt, file_name:str) -> None:
    """Saves the given plot to a file with the given filename to the thesis directory.
    
    plot: plot to save
    filename: name of the file to save the plot to
    """
    plt.savefig(f'{IMG_FILE_PATH}{file_name}.pdf')


def plot_confusion_matrix(y_labels, pred, test_target, name:str) -> None:
    """
    Plots a confusion matrix for the given classifier.
    
    clf: classifier
    y_labels: labels for the y axis
    pred: predictions
    test_target: test target
    train_target: train target   
    """

    category_bins = [i for i in range(len(y_labels))]
    cm = metrics.confusion_matrix(test_target, pred, labels=category_bins)
    # category_labels = [i[:8] for i in unique_categories]
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    disp.im_.set_norm(LogNorm(vmin=1, vmax=cm.max()))
    plt.xticks(category_bins, y_labels, rotation='vertical', fontsize=8)
    plt.yticks(category_bins, y_labels, fontsize=8)
    plt.tight_layout()
    # plt.show()
    save_plot_image(plt, f"plot_cm_{name}")
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
    ax[0].yaxis.grid(True)
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
    ax[1].yaxis.grid(True)
    ax[1].set_xticks(np.arange(len(languages)))
    ax[1].set_xticklabels(language_labels, fontsize=7, ha='center')
    ax[1].legend()
    fig.tight_layout()
    # plt.show()
    save_plot_image(plt, filename)
    plt.close(fig)


def plot_original_en_histograms(data:Dataset, filename:str) -> None:
    """Plot the language and categorical histograms for the original english site data.
    
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
    ax.bar(df1_c.index, df1_c.values, bar_width, alpha=opacity, color='b', label='Original English Data')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Count')
    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(category_labels, rotation=90, fontsize=10)
    # turn on y grid
    ax.yaxis.grid(True)
    ax.legend()
    fig.tight_layout()
    # plt.show()
    save_plot_image(plt, filename)
    plt.close(fig)
    # df1_c shows counts for the plot sum is 275


def plot_all_results_individual(folder_name:str = 'kernel_rbf') -> None:
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

def plot_all_results_from_probal(folder_name:str = 'kernel_rbf') -> None:
    """
    Plots all TEST and TRAIN data from the probal results folder into one figure based on the KERNEL.

    """
    file_path = f'/Users/mitchellborchers/Documents/git/probal/results/{folder_name}'
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
            if not file_name.endswith('.csv'): 
                continue
            loc = os.path.join(file_path, file_name)
            pattern = r'data_([a-zA-Z0-9]+)[\-_]'
            plt_title = re.findall(pattern, file_name)[0]

            with open(loc, 'r', ) as f:
                data = pd.read_csv(f)
                ax.plot(data['train-error'], label='Train Error')
                ax.plot(data['test-error'], label='Test Error')
                if i ==0: 
                    ax.legend()
                if i==0 or i==2: 
                    ax.set_ylabel('Error')
                if i==2 or i==3: 
                    ax.set_xlabel('Budget')
                ax.set_ylim([0, 1.1])
                ax.set_title(plt_title)
                ax.grid(which='both', linewidth=0.3)
                i+=1

        fig.tight_layout()
        save_plot_image(plt, f'plot_all_results_{n}')
        plt.close()

def plot_test_results(folder:str = 'kernel_cos') -> None:
    """
    Plots all TEST data from the probal results folder into one figure (this is plotting just the single runs).
    
    """
    file_path = f'/Users/mitchellborchers/Documents/git/probal/results/{folder}'
    file_names = os.listdir(file_path)
    file_names = [s for s in file_names if not s.startswith('.')]
    file_names = sorted(file_names)
    pattern = r"data_(.*?)_0"

    fig, ax = plt.subplots()
    for file_name in file_names:
        if not file_name.endswith('.csv'): continue
        loc = os.path.join(file_path, file_name)
        with open(loc, 'r') as f:
            data = pd.read_csv(f)
            # remove .csv from file name
            name = file_name[:-4]
            match = re.search(pattern, name)
            if match:
                captured_text = match.group(1)
                name = captured_text
            ax.plot(data['test-error'], label=name, linewidth=0.4)

    ax.set_xlabel('Budget')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(which='both', linewidth=0.3)
    ax.set_ylim([0, 1.1])
    fig.tight_layout()
    save_plot_image(plt, f'plot_{folder}_test_results')
    plt.close()

def plot_test_results_averaged(folder:str = 'kernel_cos_averaged') -> None:
    """
    Plots all TEST data AVERAGED from the probal results folder into one figure.
    
    """
    file_path = f'/Users/mitchellborchers/Documents/git/probal/results/{folder}'
    file_names = os.listdir(file_path)
    file_names = [s for s in file_names if not s.startswith('.')]
    # remove names with string sample
    file_names = [s for s in file_names if not 'sample' in s]
    group = 1 # group for regex
    if folder == 'kernel_cos_averaged':
        pattern = r"data_all_(.*?)_0" # for 'kernel_cos_averaged' folder
    if folder == 'all_data_new_vectorizer':
        pattern = r"(data_|additional_)(.*?)_0"  # for 'all_data_new_vectorizer' folder
        group = 2
    if folder == 'filtered':
        pattern = r"(filtered_)(.*?)_0"
        group = 2
    if folder == 'text_data_all_proper_vectorizer':
        pattern = r"performances_text_data_all_(.*?)_0"
    assert pattern, 'Pattern for plot averaged test results is not defined'

    # group files together and open in groups and aggregate test data

    groups = ['alce', 'pal', 'xpal', 'log-loss', 'random', 'qbc', 'entropy']
    groups = sorted(groups)
    fig, ax = plt.subplots()
    for g in groups:
        group_data = pd.DataFrame()
        for file_name in file_names:
            if not file_name.endswith('.csv'): continue
            name = file_name[:-4]
            match = re.search(pattern, name) 
            if match:
                captured_text = match.group(group) # 2 was 1 originally
                name = captured_text
                if g == name:
                    print(file_name)
                    loc = os.path.join(file_path, file_name)
                    with open(loc, 'r') as f:
                        data = pd.read_csv(f)
                        group_data = pd.concat([group_data, data['test-error']])

        group_data = group_data.groupby(group_data.index).mean()
        # print(group_data.shape)
        ax.plot(group_data, label=g, linewidth=0.4)

    ax.set_xlabel('Budget')
    ax.set_ylabel('Error')
    ax.legend()
    ax.grid(which='both', linewidth=0.3)
    ax.set_ylim([0, 1.1])
    fig.tight_layout()
    save_plot_image(plt, f'plot_{folder}_test_results')
    plt.close()

def plot_pr_curve(args, X:np.ndarray, y:np.ndarray, data_name=None) -> None:
    """
    Plots and saves the precision recall curve for the given data with LSVC as a pdf.
    
    Parameters
    ----------
    X : np.ndarray
        The data to be used for the plot
    y : np.ndarray
        The labels for the data
    
    """
    if data_name != None: 
        data_name = '_' + data_name
    if data_name == "original":
        data_name = None

    y_labels = np.unique(y)
    n_classes = len(y_labels)
    y_binarized = label_binarize(y, classes=y_labels)
    train_X, test_X, train_y, test_y = train_test_split(X, y_binarized, test_size=0.25, random_state=args.seed, stratify=y)


    classifier = OneVsRestClassifier(
        make_pipeline(LinearSVC(random_state=args.seed, max_iter=100000))
    )
    classifier.fit(train_X, train_y)
    y_score = classifier.decision_function(test_X)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_y[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(test_y[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        test_y.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(test_y, y_score, average="micro")

    _, ax = plt.subplots(figsize=(10, 6))

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

    display = metrics.PrecisionRecallDisplay(
        recall=recall["micro"],
        precision=precision["micro"],
        average_precision=average_precision["micro"],
    )
    display.plot(ax=ax, name="Micro-average PR", color="gold")

    for i, label in zip(range(n_classes), y_labels):
        display = metrics.PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i],
        )
        if i < 10:
            display.plot(ax=ax, name=f"{label}")
        elif i < 20:
            display.plot(ax=ax, name=f"{label}", linestyle=":")
        else:
            display.plot(ax=ax, name=f"{label}", linestyle="--")

    # add the legend for the iso-f1 curves
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])
    # set the legend and the axes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    # ax.legend(handles=handles, labels=labels, loc="best")
    ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_title("Multi-Class Precision-Recall Curve")
    plt.tight_layout()
    save_plot_image(plt, f"plot_pr_curve{data_name}")
    plt.close()

def plot_LSCV_varying_min_category(args, X, y) -> None:
    """
    Split the data into train and test sets, filter the data on the number of occurances in the train set. Build the vectorizer for the train set and then transform the test set each iteration. Plot the test error vs the min number of occurances in the train set and color the points by the number of categories in the test set.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments passed in from the command line
    X : list
        The text data
    y : list
        The labels    
    """

    le = LabelEncoder()
    y = le.fit_transform(y)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)
    vectorizer = tfidf_vectorizer()

    error_list = []
    num_categories = []
    min_occurances_in_train_data = np.arange(0, 30, 3)
    for c in min_occurances_in_train_data:
        train_df = pd.DataFrame({'text': train_X, 'category': train_y})
        train_df = train_df.groupby('category').filter(lambda x: x['category'].value_counts() > c)
        remaining_categories = train_df['category'].unique()
        test_df = pd.DataFrame({'text': test_X, 'category': test_y})
        test_df = test_df[test_df['category'].isin(remaining_categories)]
        # split the test data into x and y
        te_X = test_df['text']
        te_y = test_df['category']
        tr_X = train_df['text']
        tr_y = train_df['category']
        tr_X_v = vectorizer.fit_transform(tr_X)
        te_X_v = vectorizer.transform(te_X)
        # use label encode to change the numbers to the actual category
        # train_df['category'] = le.inverse_transform(train_df['category'])
        # print(train_df['category'].value_counts())
        # split back into x and y
        model = LinearSVC(random_state=args.seed, max_iter=10000)
        model.fit(tr_X_v, tr_y)
        pred = model.predict(te_X_v)
        error = 1 - metrics.accuracy_score(te_y, pred)
        error_list.append(error)
        num_categories.append(len(np.unique(te_y)))

    scatter = plt.scatter(min_occurances_in_train_data, error_list, c=num_categories, cmap='viridis_r')
    plt.legend(*scatter.legend_elements(), loc='upper right', title='Categories')
    plt.xlabel("Min Data Points in Category")
    plt.ylabel("Test Error")
    plt.ylim(0, 1)
    plt.xticks(min_occurances_in_train_data)
    plt.grid()
    plt.savefig(f"{IMG_FILE_PATH}plot_LSVC_varying_min_category.pdf")
    plt.close()

def plot_data_length_grid_search(args, data:Dataset) -> None:
    """
    Plots and saves the error rate for different data lengths for the given data with LSVC as a pdf.
    
    Parameters
    ----------
    data : Dataset
        The data to be used for the plot
    """
    y_bottom_up_error = []
    x_bottom_up_error = []
    # for i in range(0,20,5):
    for i in range(0,200,1):
        X, y, vectorizer = data_prep(data, origin_filter=None, min_str_len=i)
        y_bottom_up_error.append(test_model(args, X, y, vectorizer, clf_name='LinearSVC'))
        x_bottom_up_error.append(i)

    y_top_down_error = []
    x_top_down_error = []
    # for i in range(999, 900, -10):
    for i in range(999, 200, -8): # for min str len = 0
        X, y, vectorizer = data_prep(data, origin_filter=None, max_str_len=i)
        y_top_down_error.append(test_model(args, X, y, vectorizer, clf_name='LinearSVC'))
        x_top_down_error.append(i)

    fig, ax = plt.subplots()

    min_index_top = np.argmin(y_top_down_error)
    min_index_bottom = np.argmin(y_bottom_up_error)
    ax.plot(x_top_down_error, y_top_down_error, linewidth=1)
    ax.plot(x_bottom_up_error, y_bottom_up_error, linewidth=1)
    
    # ax.plot(x_top_down_error[min_index_top], y_top_down_error[min_index_top], 'o', color='red', linewidth=1)
    # ax.plot(x_bottom_up_error[min_index_bottom], y_bottom_up_error[min_index_bottom], 'o', color='red', linewidth=1)

    ax.set_xlabel("String Length")
    ax.set_ylabel("Error")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1000)
    ax.grid(True)

    # ax.annotate(f'({y_top_down_error[min_index_top]:.2f}, {x_top_down_error[min_index_top]})', xy=(x_top_down_error[min_index_top], y_top_down_error[min_index_top]), 
    #             xytext=(x_top_down_error[min_index_top]+1, y_top_down_error[min_index_top]-.05),
    #             )
    # ax.annotate(f'({y_bottom_up_error[min_index_bottom]:.2f}, {x_bottom_up_error[min_index_bottom]})', xy=(x_bottom_up_error[min_index_bottom], y_bottom_up_error[min_index_bottom]),
    #             xytext=(x_bottom_up_error[min_index_bottom]+1, y_bottom_up_error[min_index_bottom]-.05),
    #             )
    ax.legend(['Max string size', 'Min string size'], loc='lower right')
    save_plot_image(plt, "plot_data_length_grid_search")
    plt.close()

    # best results
    # X, y, vectorizer = data_prep(data, origin_filter=None, min_str_len=155)
    # test_model(args, X, y, vectorizer, clf_name='LinearSVC')


def get_weights(y:np.ndarray, method:str='cosine') -> dict:
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
        if method == 'cosine':
            weight = cosine_decay_weight(n, total_steps, initial_weight, final_weight)
        elif method == 'linear':
            weight = 1 - (n/total_steps)
        weights[i] = weight

    # Print the dictionary of weights
    # for w in weights:
    #     print(f"Step {w}: {weights[w]}")
    return weights

def plot_explore_classifiers(args, X, y):
    le = LabelEncoder()
    y = le.fit_transform(y)
    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25, random_state=args.seed, stratify=y)

    weights = get_weights(y)

    models = [
        LinearSVC(random_state=args.seed, max_iter=10000),
        KNeighborsClassifier(n_neighbors=8, metric='cosine'),
        MLPClassifier(random_state=args.seed, max_iter=100, hidden_layer_sizes=500),
        SVC(),
        # GradientBoostingClassifier(n_estimators=100, learning_rate=.09, max_depth=3, random_state=args.seed),
        GaussianNB(),
        LogisticRegression(random_state=args.seed, max_iter=1000),
        RandomForestClassifier(random_state=args.seed, n_estimators=200),
    ]

    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        if model_name == 'SVC':
            train_X_cosine = pairwise_kernels(X, metric='cosine')
            accuracies = cross_val_score(model, train_X_cosine, y, scoring='accuracy', cv=CV)
        if model_name == 'GaussianNB':
            accuracies = cross_val_score(model, X.toarray(), y, scoring='accuracy', cv=CV)
        else:
            accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
        # use error instead of accuracy
        errors = 1 - accuracies
        for fold_idx, error in enumerate(errors):
            entries.append((model_name, fold_idx, error))

    cv_df = pd.DataFrame(entries, columns=['Classifier', 'fold_idx', 'Error'])
    sns.boxplot(x='Classifier', y='Error', data=cv_df)
    plt.xticks(rotation=90)
    plt.ylim([0, 1])
    plt.grid(True, axis='y')
    plt.tight_layout()
    save_plot_image(plt, 'plot_explore_classifiers')
    plt.close()

    # create a pandas data frame from the models and their parameters
    df = pd.DataFrame(columns=['Classifier', 'Parameters'])
    with pd.option_context("max_colwidth", 1000):
        for model in models:
            params = model.get_params()
            # if 'class_weight' in params.keys():
            #     params['class_weight'] = 'precomputed'
            df = df.append({'Classifier': model.__class__.__name__,
                            'Parameters': params}, ignore_index=True)
        # save df as a latex table
        df.to_latex(f"{IMG_FILE_PATH}table_explore.tex", index=False, column_format='p{4.3cm}|p{9cm}')

def table_cosine_decay_weights(args, y) -> None:
    """
    Creates latex .tex table file with the cosine decay weights.
    
    """
    le = LabelEncoder()
    y = le.fit_transform(y)
    w = get_weights(y)
    # turn dict into table
    table = pd.DataFrame.from_dict(w, orient="index")
    # add column names
    table.columns = ["Weight"]
    # turn numbers back into labels
    table.index = le.inverse_transform(table.index)
    # save table as latex
    table.to_latex(f"{IMG_FILE_PATH}table_cosine_decay_weights.tex", index=True, float_format="%.3f")

def table_correlated_unigrams(X:np.ndarray, y:np.ndarray, v:TfidfVectorizer, file_name:str) -> None:
    """
    Creates latex .tex table file with of features with the highest chi-squared statistics per target class.

    X: sparse matrix of shape (n_samples, n_features)
    y: array of shape (n_samples,)
    v: vectorizer
    category_to_id: dictionary with category names mapped to their numeric id
    """
    y_labels = np.unique(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    category_to_id = dict(zip(y_labels, range(len(y_labels))))

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
    df.to_latex(f"{IMG_FILE_PATH}{file_name}.tex")


def table_classification_report(test, pred, labels, clf_name_and_info:str) -> None:
    """
    Create latex .tex table file with classification report.
    
    Parameters
    ----------
    test : array-like
        Ground truth (correct) target values.
    pred : array-like
        Estimated targets as returned by a classifier.
    labels : array-like
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.
    clf_name_and_info : str
        String with classifier name and additional info.

    Returns
    -------
    None
    """
    report = metrics.classification_report(test, pred, labels=range(23), target_names=labels, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_latex(f"{IMG_FILE_PATH}table_classification_report_{clf_name_and_info}.tex", float_format="%.2f")


def table_category_counts(data:Dataset, file_name:str) -> None:
    """
    Creates latex .tex table file with 'original' or 'additional' data counts.

    Parameters
    ----------
    data : Dataset
        Dataset object
    group : List[str], optional
        List of strings with 'original' or 'additional' data, by default ['original', 'additional']
    
    Returns
    -------
    None
    """
    td_df = data.translated_data
    sd_df = data.site_data
    # Merge the two DataFrames on the 'id' column
    merged_data = pd.merge(sd_df, td_df, how='inner', left_on='id', right_on='site_data_id')
    # Filter the merged DataFrame to only include rows where the 'origin' column == 'original'
    original = merged_data[merged_data['origin'] == 'original']
    additional = merged_data[merged_data['origin'] == 'additional']
    
    # Group the filtered DataFrame by the 'category' column and count the number of rows in each group
    group_original = sd_df.groupby('category').size().reset_index(name='Original')
    group_og_english = original[original['original_language'] == 'en'].groupby('category').size().reset_index(name='Original English')
    group_additional = additional.groupby('category').size().reset_index(name='Additonal')
    group_translated = merged_data[merged_data['origin'] == 'original'].groupby('category').size().reset_index(name='Translated')
    # export to latex without indices
    re_merged = pd.merge(group_original, group_og_english, how='outer', left_on='category', right_on='category' )
    re_merged = pd.merge(re_merged, group_translated, how='outer', left_on='category', right_on='category')
    re_merged = pd.merge(re_merged, group_additional, how='outer', left_on='category', right_on='category')
    re_merged = re_merged.fillna(0)
    re_merged.rename(columns={'category': 'Category'}, inplace=True)
    re_merged.to_latex(f"{IMG_FILE_PATH}{file_name}.tex", index=False, float_format="%.0f")

def table_variable_importance(X:np.ndarray, y:np.ndarray, vectorizer:TfidfVectorizer) -> None:
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
    y = le.fit_transform(y)

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
    top_df.rename(columns={'importance': 'Importance'}, inplace=True)
    bottom_df = pd.DataFrame.from_dict(bottom, orient='index', columns=['importance'])
    bottom_df.rename(columns={'importance': 'Importance'}, inplace=True)
    top_df.to_latex(f"{IMG_FILE_PATH}table_top_20_features.tex")
    bottom_df.to_latex(f"{IMG_FILE_PATH}table_bottom_20_features.tex")