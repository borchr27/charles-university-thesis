from matplotlib import pyplot as plt
import psycopg2
import re
import numpy as np
import pycld2 as cld2
from sklearn import metrics
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class Website:
    def __init__(self, name, i_url, v_url, text, category, tags):
        self.name = name
        self.i_url = i_url
        self.v_url = v_url
        self.text = text
        self.category = category
        self.tags = tags

def plot_histograms(lang_array:np.ndarray, train_target:np.ndarray, show:bool) -> None:
    """Plots histograms of the languages and categories in the dataset."""
    if show:
        unique_categories, category_counts = np.unique(train_target, return_counts=True)
        # for i in range(len(unique_categories)):
        #     print(f"{unique_categories[i]}: {category_counts[i]}")
        category_bins = [i for i in range(len(category_counts))]
        category_labels = [i[:8] for i in unique_categories]
        plt.bar(category_bins, category_counts, align='center')
        plt.title('Histogram of categories')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(category_bins, category_labels, rotation='vertical', fontsize=8)
        plt.tight_layout()
        plt.show()

        unique_langs, lang_counts = np.unique(lang_array, return_counts=True)
        # for i in range(len(unique_langs)):
        #     print(f"{unique_langs[i]}: {lang_counts[i]}")
        lang_bins = [i for i in range(len(lang_counts))]
        lang_labels = [i for i in unique_langs]
        plt.bar(lang_bins, lang_counts, align='center')
        plt.title('Histogram of languages')
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.xticks(lang_bins, lang_labels, rotation='vertical', fontsize=8)
        plt.tight_layout()
        plt.show()

def preprocessing(train_data:np.ndarray, languages:np.ndarray = ['ENGLISH'], debugging:bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Preprocesses the data and returns the indicies of the selected languages and the language array."""
    lang_array = [None] * len(train_data)
    lang_array = np.array(lang_array)
    # preprocess data with regex, remove numbers and symbols
    for i in range(len(train_data)):
        text = train_data[i]
        text = re.sub(r'[^\w\s]+', ' ', text, flags=re.UNICODE)
        text = re.sub('\d', ' ', text)
        text = re.sub("\s{2,}", ' ', text)
        train_data[i] = text[:3000]

    # try and get language of each data point
    for i in range(len(train_data)):
        isReliable, textBytesFound, details, vectors = cld2.detect(train_data[i], returnVectors=True)
        if len(vectors) != 0 and isReliable:
            for v in vectors:
                if v[2] != 'Unknown':
                    lang_array[i] = v[2]
                    break
        if lang_array[i] == None:
            lang_array[i] = 'Unknown'    

    # get indicies of specified languages data
    selected_language_indicies = []
    for l in languages:
        selected_language_indicies += np.where(lang_array == l)[0].tolist()
    selected_language_indicies = np.array(sorted(selected_language_indicies))
    unknown_indicies = np.where(lang_array == 'Unknown')

    # prints unknown language train data
    if debugging:    
        unk = np.array(train_data)[unknown_indicies]
        for d in unk:
            print(d)
            print(80*"-")

    # returning lang_array just to use in histogram
    return selected_language_indicies, lang_array

def show_confusion_matrix(clf, y_labels, pred, test_target, train_target):
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
    return TfidfVectorizer(analyzer="word", strip_accents="unicode", max_features=500, stop_words="english")

def tfidf_to_csv(text_data:np.ndarray, y:np.ndarray, filename:str = "website_tfidf_data.csv"):
    """Converts TF-IDF data to a CSV file to be used in the xPAL algorithm."""
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
        self.execute("""
        CREATE TABLE IF NOT EXISTS site_data (
            id serial PRIMARY KEY not null, 
            name varchar not null,
            initial_url varchar not null, 
            visited_url varchar not null, 
            text varchar not null,
            category varchar not null,
            tags varchar not null,
            timestamp timestamp default current_timestamp
            );
        """)

    def close(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def commit(self) -> None:
        # assert self.connection, 'Database connection is not established'
        if self.connection:
            self.connection.commit()

    def execute(self, command, values=None) -> None:
        # assert self.connection, 'Database connection is not established'
        if self.connection:
            self.cursor = self.connection.cursor()
            self.cursor.execute(command, values)

    def add_website(self, site: Website) -> None:
        
        if site.i_url == "missing note" or site.v_url == "" or site.text == "":
            return
        
        # Check to see if text is already in database
        self.cursor.execute(
            "SELECT * FROM site_data WHERE initial_url = %s;", (site.i_url,))
        result = self.cursor.fetchone()

        # If it is in DB, create log message
        if result:
            print(f"Item already in database: {site.name}")
        else:
            # Define insert statement
            self.cursor.execute("""INSERT INTO site_data (name, initial_url, visited_url, text, category, tags) VALUES (%s,%s,%s,%s,%s,%s);""", (
                site.name,
                site.i_url,
                site.v_url,
                site.text,
                site.category,
                site.tags,
            ))

        # Execute insert of data into database
        self.commit()

class Dataset:
    def __init__(self):
        self.websites = []
        self.data = []
        self.target = []

        db = PostgresDatabase()
        db.connect()
        if db.connection:
            db.execute('SELECT * FROM site_data;')
            result = db.cursor.fetchall()
            db.close()

            for row in result:
                w = Website(name=row[1], i_url=row[2], v_url=row[3], text=row[4], category=row[5], tags=row[6])
                self.websites.append(w)
                self.data.append(w.text)
                self.target.append(w.category)


# command for probal
# python3 experimental_setup_csv.py \
#   --query_strategy xpal-0.001 \
#   --data_set website_tfidf_data \
#   --results_path ../../results \
#   --test_ratio 0.4 \
#   --bandwidth mean \
#   --budget 200 \
#   --seed 1