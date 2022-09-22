# import libraries
from flask import Flask, render_template, request

import pyspark
from pyspark.sql import SparkSession

from pyspark.ml.tuning import CrossValidatorModel

app = Flask(__name__)

# create spark session
spark = SparkSession.builder.master("local")\
    .appName("Capstone").getOrCreate()

# load model
model = CrossValidatorModel.read().load("gbt")

# feature columns needed for the model to predict
feature_cols = [
    "gender",
    "NextSong",
    "Downgrade",
    "Upgrade",
    "Thumbs Down",
    "Thumbs Up",
    "Submit Upgrade",
    "Add Friend",
    "Add to Playlist",
    "Roll Advert",
    "free",
    "paid"
]

# index webpage that takes user input
@app.route("/")
@app.route("/index")
def index():

    # render web page with plotly graphs
    return render_template(
        "master.html"
    )


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # get user input and make predictions
    feature_vals = []
    features = [
        "feature1", "feature2", "feature3",
        "feature4", "feature5", "feature6",
        "feature7", "feature8", "feature9",
        "feature10", "feature11", "feature12"
    ]
    for f in features:
        feature_vals.append(
            int(request.args.get(f))
        )
    data = [(
        feature_vals[0],
        feature_vals[1],
        feature_vals[2],
        feature_vals[3],
        feature_vals[4],
        feature_vals[5],
        feature_vals[6],
        feature_vals[7],
        feature_vals[8],
        feature_vals[9],
        feature_vals[10],
        feature_vals[11]
    )]
    input_df = spark.createDataFrame(data, feature_cols)

    # use model to predict classification for query
    pred = model.transform(input_df)
    result = "yes" if pred.select("prediction").collect()[0][0] == 1 else "no"
    prob = pred.select("probability").collect()[0][0][0]

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", result=result, prob=prob
    )


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
