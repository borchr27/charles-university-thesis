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

import configparser
import requests
from dotenv import load_dotenv

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

def save_plot_image(plot:plt, filename:str) -> None:
    """Saves the given plot to a file with the given filename to the thesis directory."""
    plt.savefig(f'/Users/mitchellborchers/Documents/git/charles-university-thesis/thesis/vzor-dp/img/{filename}.jpg')

def get_data_histograms(data:Dataset, filename:str) -> None:
    """Get the alnguage and categorical histograms for the original site data and the translated data"""
    td_df = data.translated_data
    sd_df = data.site_data
    m_df = pd.merge(td_df[['site_data_id', 'original_language']], sd_df[['id', 'category','origin']], left_on='site_data_id', right_on='id')
    original = m_df[m_df['origin']=='original']
    additional = m_df[m_df['origin']=='additional']
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Get the unique categories from all three dataframes
    categories = pd.concat([original['category'], additional['category'], sd_df['category']]).unique()
    # Set the bin edges to be the unique categories
    bins = np.arange(len(categories) + 1)

    # fig, ax = plt.subplots()
    ax[0].hist(original['category'], bins=bins, alpha=0.5, label='Original Data')
    ax[0].hist(additional['category'], bins=bins, alpha=0.5, label='Additional Data')
    ax[0].set_xlabel('Category')
    ax[0].set_ylabel('Count')
    ax[0].set_xticks(bins[:-1] + 0.5)
    ax[0].set_xticklabels(categories, rotation=90)
    ax[0].legend(loc='upper right')

    # Get the unique categories from all three dataframes
    languages = pd.concat([original['original_language'], additional['original_language'], td_df['original_language']]).unique()
    labels = np.array([i.upper() for i in languages])
    # Set the bin edges to be the unique categories
    bins = np.arange(len(languages) + 1)

    ax[1].hist(original['original_language'], bins=bins, alpha=0.5, label='Original Data')
    ax[1].hist(additional['original_language'], bins=bins, alpha=0.5, label='Additional Data')
    ax[1].set_xticks(bins[:-1] + 0.5)
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].legend(loc='upper right')
    ax[1].set_xlabel('Language')
    ax[1].set_ylabel('Count')
    fig.tight_layout()
    save_plot_image(plt, f'{filename}')
    plt.show()
    plt.close(fig)

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

def show_confusion_matrix(clf, y_labels, pred, test_target, train_target) -> None:
    """Plots a confusion matrix for the given classifier."""
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
    plt.show()

def tfidf_vectorizer() -> TfidfVectorizer:
    """Function to return a TF-IDF vectorizer."""
    return TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=5000, stop_words="english")

def tfidf_to_csv(text_data:np.ndarray, y:np.ndarray, filename:str = "website_tfidf_data.csv") -> None:
    """Converts TF-IDF data to a CSV file to be used in the xPAL and other algorithms."""
    # get the tfidf vectorizer
    vectorizer = tfidf_vectorizer()

    # Fit and transform the text data
    tfidf = vectorizer.fit_transform(text_data)
    tfidf = tfidf.toarray()
    y = np.array(y)
    # concatenate tfidf and y
    tfidf = np.concatenate((tfidf, y.reshape(-1, 1)), axis=1)

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()
    # create feature names in the form of x_0 to x_n
    feature_names_xs = ["x_" + str(i) for i in range(len(feature_names))]
    # append y to feature names
    feature_names_xs = np.append(feature_names_xs, "y")

    # Convert the TF-IDF data to a Pandas DataFrame
    tfidf_df = pd.DataFrame(tfidf, columns=feature_names_xs)

    # Export the DataFrame to a CSV file
    tfidf_df.to_csv(filename, index=False)


def azure_translate(text:str, source_language:str = None, target_language:str ='en') -> str:
    """Use free Azure Cognitive Services to translate text to English."""
    """Helpful link https://techcommunity.microsoft.com/t5/educator-developer-blog/translate-your-notes-with-azure-translator-and-python/ba-p/3267201"""

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


class PostgresDatabase:
    def __init__(self) -> None:
        self.connection = None
        self.cursor = None

    def connect(self):
        """! Connect to the postgres database. To view the database or debug open up the shell for the 
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

