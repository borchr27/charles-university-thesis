# coding=utf8
import numpy as np

def language_prediction_test():
    import pycld2 as cld2
    text = """\
    Начало Консултирай се Моля """

    isReliable, textBytesFound, details, vectors = cld2.detect(text, returnVectors=True) 
    for v in vectors:
        print(v)

def tfidf_to_csv_test():
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    # Sample text data
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "The brown fox is quick and the dog is lazy.",
        "The red fox is not as quick as the brown fox.",
        "The lazy dog is sleeping and the quick brown fox is jumping."
    ]

    y = [1, 2, 0, 0]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the text data
    tfidf = vectorizer.fit_transform(corpus)
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
    tfidf_df.to_csv('tfidf_data.csv', index=False)

def where():
    a = np.array(["En", "Es", "En", "Fr"])
    B = np.array(["En", "Es"])
    ind = []
    for b in B:
        ind += np.where(a == b)[0].tolist()
    
    return sorted(ind)

if __name__ == "__main__":
    # language_prediction_test()
    # tfidf_to_csv_test()
    print(where())
