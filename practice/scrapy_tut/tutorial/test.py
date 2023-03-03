# coding=utf8
import numpy as np
import matplotlib.pyplot as plt

def language_prediction_test():
    import pycld2 as cld2
    text = """\
    Начало Консултирай се Моля """
    text2 = "Katuseaknad" # should be estonian

    isReliable, textBytesFound, details, vectors = cld2.detect(text2, returnVectors=True) 
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

def lang_translation():
    import translators as ts
    import translators.server as tss

    chs_text = '季姬感到寂寞，罗集了一些鸡来养，鸡是出自荆棘丛中的野鸡。野鸡饿了唧唧叫，季姬就拿竹箕中的谷物喂鸡。'
    text2 = "Maneki Tea Talk | Dublin" #"Katuseaknad"
    # print(ts.translators_pool)
    result = ts.translate_text(text2, translator='bing', is_detail_result=True)
    print(result["detectedLanguage"]["language"])
    print(result["translations"][0]["text"])
    # print(tss.sogou(chs_text, is_detail_result=True))
    # print(ts.translate_text(chs_text, translator='google', from_language="zh", to_language="en"))
    # print(tss.deepl(chs_text, to_language='en', from_language='zh'))

def save_plot_image(plot:plt, filename:str):
    # Save the plot as a JPEG file in a specific location
    plt.savefig(f'/Users/mitchellborchers/Documents/git/charles-university-thesis/thesis/vzor-dp/img/{filename}.jpg')


if __name__ == "__main__":
    # language_prediction_test()
    # tfidf_to_csv_test()
    # print(where())
    # lang_translation()

    # Create some sample data
    x = [1, 2, 3, 4, 5]
    y = [10, 8, 6, 4, 2]
    # Create a line plot of the data
    plt.plot(x, y)
    # Set the title and axis labels
    plot_name = 'Line Plot'
    filename = plot_name.lower().replace(' ', '_')
    plt.title(f'{plot_name}')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    save_plot_image(plt, filename)
