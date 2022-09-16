# Disaster Response Pipeline Project

The main goal of this project is to classify messages into categories in order to help emergency agencies to respond during disaster events. 

The training dataset from [Figure Eight](https://appen.com/) uses real messages sent during the disasters' happenings. 

On the web app you can classify a message (text) as well as explore a little about the training set.

### Instructions to run:
1. Go to the `disaster_response` directory to run the following commands for this project.

2. To ensure you have correct environment to run the project, please run the following to install all the packages needed: 
    `pip install -r requirements.txt`

2. Run the following commands to set up your database and model:

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves the model
        `python models/train_classifier.py sqlite:///DisasterResponse.db models/random_forest.pkl`

3. Run the following command to run your web app:
    `python app/run.py`

4. Go to the url returned in the command line

### Files:

- `app` folder: .html files and python script that wraps up the web app
- `data` folder: raw data and python script that preprocesses the data for the training
- `models` folder: saved models and python script that trains, evaluates and saves the model. `random_forest.pkl` is the model we use for the web app for its better performance. 

