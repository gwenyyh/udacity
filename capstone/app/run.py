import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask, render_template, request
from plotly.graph_objs import Bar, Box
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine("sqlite:///DisasterResponse.db")
df = pd.read_sql_table("tbl_yyh_disaster_response_clean_data", engine)

# load model
model = joblib.load("models/random_forest.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    cat_counts = df.sum(numeric_only=True).sort_values(ascending=False)
    cat_names = list(cat_counts.index)
    word_counts = df.message.apply(lambda x: len(x.split()))
    train_size = df.shape[0]
    # create visuals
    graphs = [
        {
            "data": [Bar(x=cat_names, y=cat_counts)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category"},
            },
        },
        {
            "data": [Box(x=word_counts)],
            "layout": {
                "title": "Distribution of Message Word Counts",
                "yaxis": {"title": ""},
                "xaxis": {"title": "Word Count"},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template(
        "master.html",
        ids=ids,
        graphJSON=graphJSON,
        train_size=f"{train_size:,}"
    )


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
