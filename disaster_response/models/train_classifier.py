# import libraries
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import sys
import pandas as pd
import pickle
import nltk

nltk.download(["stopwords", "wordnet"])


def load_data(database_filepath: str):
    """loads data from database_filepath
    input:
        database_filepath: path to the database
    output:
        X: numpy array of texts
        y: numpy array of target categories
        category_names: list of target category names
    """

    engine = create_engine(database_filepath)
    df = pd.read_sql_table("tbl_yyh_disaster_response_clean_data", engine)
    X = df.head(100)[["message", "genre"]]
    y = df.head(100).drop(["message", "genre"], axis=1).values
    category_names = list(df.columns)[2:]
    return X, y, category_names


def tokenize(text):
    """tokenizes text into lemmatized, stripped, lowercased tokens
    without stopwords

    input:
        text: str - the text to be tokenized
    output:
        clean_tokens: list - of tokens
    """
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [
        lemmatizer.lemmatize(word).lower().strip()
        for word in tokens
        if word not in stop_words
    ]

    return clean_tokens


# build the final model with external feature 'genre'
def build_model():
    """
    input: None
    Output: model pipeline
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(
            tokenizer=tokenize, 
            max_df=0.5,
            ngram_range=(1, 1),
            max_features=None
        )),
        ("tfidf", TfidfTransformer(use_idf=True)),
        ("clf", MultiOutputClassifier(
            estimator=RandomForestClassifier(
                n_estimators=200,
                max_features=None,
                random_state=42
            )
        ))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names: list):
    """prints precision, recall, F1 score of the model on the test set
    input:
        model: trained model object to make predictions
        X_test: numpy array
        Y_test: numpy array
        category_names: list of category names
    Output:
        None
    """
    y_pred = model.predict(X_test)
    y_test = pd.DataFrame(y_test, columns=category_names)
    y_pred = pd.DataFrame(y_pred, columns=category_names)

    for column in y_test:
        print(column)
        print(classification_report(
            y_test[column], y_pred[column], zero_division=0
        ))


def save_model(model, model_filepath: str):
    """saves model as pickle file to filepath"""
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    '''execute if main'''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2
        )

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        # save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
